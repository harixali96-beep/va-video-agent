#!/usr/bin/env python3
"""
VA Chat - the AI brain inside the video agent.

A local LLM (ollama, default qwen3.5:4b, zero cost, zero API key) reads the user's
free-text brief - "ye drive link se 3 clips banao, 25 second, ye CTA aur hashtags" -
and drives the real editing tools itself: resolve source, plan moments, render, verify.

Tools the model can call:
  resolve_source(link)  download / prepare the footage (Google Drive, Whop, YouTube, direct)
  plan(count, length)   transcribe + score + pick the best moments
  review(index)         read the transcript of a planned moment
  render(index|start,end, hook)   burn one clip
  ask_user(question)    stop and ask the human
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import va

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.environ.get("VA_MODEL", "qwen3.5:4b")
CHAT_WORK = va.WORK / "chat"
CHAT_WORK.mkdir(parents=True, exist_ok=True)

SYSTEM = """You are VA, a video editing agent. You do the work with tools, you do not explain.

Rules:
- Never claim a file exists unless a tool returned it.
- If the user gives a link, call resolve_source first.
- Then call plan to find the best moments, then render each one you want.
- Keep every reply under 40 words. No markdown.
- If something needs a login or a file you cannot get, call ask_user.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "resolve_source",
        "description": "Download or prepare the footage from a link or local path. Supports Google Drive share links, Whop links, YouTube/TikTok/Instagram, direct mp4.",
        "parameters": {"type": "object", "properties": {
            "link": {"type": "string", "description": "the url or file path"}},
            "required": ["link"]}}},
    {"type": "function", "function": {
        "name": "set_style",
        "description": "Set branding: CTA pill text, hashtags, caption highlight colour, hook bar on/off, logo image path.",
        "parameters": {"type": "object", "properties": {
            "cta": {"type": "string"}, "hashtags": {"type": "string"},
            "logo": {"type": "string"}, "hook_bar": {"type": "boolean"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "plan",
        "description": "Transcribe the source and pick the best moments. Returns a numbered list with start, end, score and a short excerpt.",
        "parameters": {"type": "object", "properties": {
            "count": {"type": "integer", "description": "how many moments"},
            "length": {"type": "number", "description": "seconds per clip"}},
            "required": ["count", "length"]}}},
    {"type": "function", "function": {
        "name": "review",
        "description": "Read the transcript lines of one planned moment before rendering it.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"}}, "required": ["index"]}}},
    {"type": "function", "function": {
        "name": "render",
        "description": "Render a vertical 1080x1920 clip with captions and branding. Use index of a planned moment, or explicit start/end.",
        "parameters": {"type": "object", "properties": {
            "index": {"type": "integer"},
            "start": {"type": "number"}, "end": {"type": "number"},
            "hook": {"type": "string", "description": "short hook line burned at the top"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "list_clips",
        "description": "List the clips rendered so far in this job.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": "Stop and ask the human for something you cannot get yourself (login, missing file, which clip).",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"}}, "required": ["question"]}}},
]


# ---------------------------------------------------------------- ollama
def _post(path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ensure_ollama() -> bool:
    try:
        urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3).read()
        return True
    except Exception:
        pass
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        for _ in range(20):
            time.sleep(0.7)
            try:
                urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2).read()
                return True
            except Exception:
                continue
    except FileNotFoundError:
        return False
    return False


def llm(messages: list[dict], tools: list | None = None) -> dict:
    payload = {"model": MODEL, "messages": messages, "stream": False,
               "options": {"temperature": 0.2, "num_ctx": 6144, "num_predict": 400}}
    if tools:
        payload["tools"] = tools
    try:
        return _post("/api/chat", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:400]
        raise RuntimeError(f"ollama error {e.code}: {body}")


# ---------------------------------------------------------------- source resolving
GDRIVE_FILE = re.compile(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([A-Za-z0-9_-]{20,})")
GDRIVE_DIR = re.compile(r"drive\.google\.com/drive/(?:u/\d+/)?folders/([A-Za-z0-9_-]{20,})")


def resolve_source(link: str, outdir: Path = CHAT_WORK) -> str:
    link = link.strip().strip('"').strip("'")
    outdir.mkdir(parents=True, exist_ok=True)

    m = GDRIVE_FILE.search(link)
    if m:
        import gdown
        target = outdir / "drive_source"
        target.mkdir(exist_ok=True)
        out = gdown.download(id=m.group(1), output=str(target / "source.%(ext)s"),
                             quiet=True, fuzzy=True)
        if not out or not Path(out).exists():
            raise RuntimeError("Google Drive download failed - link may be private, "
                               "set sharing to 'Anyone with the link'")
        return out

    d = GDRIVE_DIR.search(link)
    if d:
        import gdown
        folder = outdir / "drive_folder"
        folder.mkdir(exist_ok=True)
        gdown.download_folder(id=d.group(1), output=str(folder), quiet=True, use_cookies=False)
        vids = [p for p in folder.rglob("*") if p.suffix.lower() in
                (".mp4", ".mov", ".mkv", ".webm", ".m4v")]
        if not vids:
            raise RuntimeError("Drive folder has no video files I can read")
        return str(max(vids, key=lambda p: p.stat().st_size))

    if "whop.com" in link:
        try:
            return str(va.ingest(link, outdir / "whop"))
        except Exception:
            raise RuntimeError("Whop links are behind a login. Paste the direct video file "
                               "URL (or upload the file) and I will take it from there.")

    return str(va.ingest(link, outdir / "src"))


# ---------------------------------------------------------------- session / tools
class Session:
    def __init__(self):
        self.dir = CHAT_WORK / f"job-{time.strftime('%H%M%S')}-{os.getpid() % 1000}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.video: Path | None = None
        self.info: dict = {}
        self.transcript: dict = {"words": []}
        self.plan: list[dict] = []
        self.style = {"cta": "", "hashtags": "", "logo": "", "hook_bar": True,
                      "highlight": "&H0000FFFF"}
        self.clips: list[dict] = []
        self.log: list[str] = []

    # ---- tools
    def t_resolve_source(self, link: str) -> str:
        path = resolve_source(link, self.dir)
        self.video = Path(path)
        self.info = va.probe(self.video)
        self.transcript = {"words": []}
        self.plan = []
        return (f"READY path={self.video.name} {self.info['width']}x{self.info['height']} "
                f"{self.info['duration']:.1f}s audio={self.info['has_audio']}")

    def t_set_style(self, cta=None, hashtags=None, logo=None, hook_bar=None) -> str:
        if cta is not None:
            self.style["cta"] = cta
        if hashtags is not None:
            self.style["hashtags"] = hashtags
        if logo is not None:
            self.style["logo"] = logo
        if hook_bar is not None:
            self.style["hook_bar"] = bool(hook_bar)
        return f"STYLE {json.dumps(self.style)}"

    def t_plan(self, count: int = 3, length: float = 30) -> str:
        if not self.video:
            return "ERROR no source yet - call resolve_source first"
        if not self.transcript.get("words"):
            self.transcript = va.transcribe(self.video, "small.en", cache_dir=self.dir)
        scenes = va.detect_scenes(self.video)
        curve = va.energy_curve(self.video)
        self.plan = va.plan(self.video, self.transcript, scenes, curve,
                            target=float(length), count=int(count))
        for i, c in enumerate(self.plan, 1):
            c = va.snap_to_sentences(c, self.transcript["words"], dur=self.info["duration"])
            self.plan[i - 1] = c
        lines = []
        for i, c in enumerate(self.plan, 1):
            ws = [w["word"] for w in self.transcript["words"]
                  if w["start"] >= c["start"] and w["end"] <= c["start"] + 5]
            lines.append(f"{i}. {c['start']:.1f}-{c['end']:.1f}s score={c['score']} | "
                         f"{' '.join(ws)[:90]}")
        return "PLAN\n" + "\n".join(lines)

    def t_review(self, index: int) -> str:
        if not self.plan:
            return "ERROR nothing planned"
        if not 1 <= int(index) <= len(self.plan):
            return f"ERROR index must be 1..{len(self.plan)}"
        c = self.plan[int(index) - 1]
        ws = [w["word"] for w in self.transcript["words"]
              if w["end"] > c["start"] and w["start"] < c["end"]]
        return f"{c['start']:.1f}-{c['end']:.1f}s: " + " ".join(ws)[:400]

    def t_render(self, index=None, start=None, end=None, hook="") -> str:
        if not self.video:
            return "ERROR no source"
        if index is not None and self.plan:
            i = int(index)
            if not 1 <= i <= len(self.plan):
                return f"ERROR index must be 1..{len(self.plan)}"
            c = dict(self.plan[i - 1])
        elif start is not None and end is not None:
            c = {"start": float(start), "end": float(end), "score": 0}
            c = va.snap_to_sentences(c, self.transcript.get("words", []),
                                     dur=self.info["duration"])
        else:
            return "ERROR give index, or start and end"
        if c["end"] - c["start"] < 2:
            return "ERROR clip shorter than 2s"
        if not hook:
            hook = va.make_hook(c, self.transcript.get("words", []))
        brand = {"hook": hook, "cta": self.style["cta"], "hashtags": self.style["hashtags"],
                 "hook_bar": self.style["hook_bar"], "highlight": self.style["highlight"]}
        out = self.dir / f"clip{len(self.clips) + 1:02d}_{va.slug(hook)}.mp4"
        logo = Path(self.style["logo"]) if self.style.get("logo") and Path(self.style["logo"]).exists() else None
        va.render_clip(self.video, c, out, brand, self.transcript.get("words", []), logo=logo)
        qa = va.verify_clip(out)
        pack = va.write_post_pack(out, c, hook.rstrip("…"), f"{hook}\n\n{self.style['hashtags']}",
                                  self.style["hashtags"])
        self.clips.append({"file": out.name, "path": str(out), "pack": str(pack),
                           "hook": hook, "start": c["start"], "end": c["end"],
                           "duration": round(c["end"] - c["start"], 2),
                           "qa_ok": qa["ok"], "qa_issues": qa["issues"]})
        return (f"RENDERED {out.name} {c['start']:.1f}-{c['end']:.1f}s "
                f"qa={'pass' if qa['ok'] else qa['issues']}")

    def t_list_clips(self) -> str:
        if not self.clips:
            return "NO CLIPS yet"
        return "\n".join(f"{i}. {c['file']} {c['duration']}s qa={c['qa_ok']}"
                         for i, c in enumerate(self.clips, 1))

    def call(self, name: str, args: dict) -> str:
        fn = {
            "resolve_source": self.t_resolve_source,
            "set_style": self.t_set_style,
            "plan": self.t_plan,
            "review": self.t_review,
            "render": self.t_render,
            "list_clips": self.t_list_clips,
        }.get(name)
        if name == "ask_user":
            return f"ASKING: {args.get('question', '')}"
        if not fn:
            return f"ERROR unknown tool {name}"
        try:
            return fn(**args)
        except TypeError:
            try:
                return fn(**{k: v for k, v in args.items() if v is not None})
            except Exception as e:
                return f"ERROR {type(e).__name__}: {e}"
        except Exception as e:
            return f"ERROR {type(e).__name__}: {e}"


# ---------------------------------------------------------------- agent turn
def run_turn(session: Session, user_text: str, history: list[dict] | None = None,
             max_steps: int = 8, on_step=None) -> dict:
    """One user message -> the model may chain several tool calls -> final reply."""
    history = history or []
    messages = [{"role": "system", "content": SYSTEM}]
    for m in history[-10:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": str(m.get("content", ""))[:1500]})
    messages.append({"role": "user", "content": user_text})

    steps, reply = [], ""
    for _ in range(max_steps):
        try:
            res = llm(messages, TOOLS)
        except Exception as e:
            reply = f"model error: {e}"
            break
        msg = res.get("message", {}) or {}
        calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        if not calls:
            reply = content or ("Done." if session.clips else "Kuch samajh nahi aaya, dobara bolo.")
            break

        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {}) or {}
            name = fn.get("name", "")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            obs = session.call(name, args)
            steps.append({"tool": name, "args": args, "result": obs[:500]})
            if on_step:
                on_step(name, args, obs)
            messages.append({"role": "tool", "content": obs[:1200], "name": name})
    else:
        reply = "Step limit hit. " + (f"{len(session.clips)} clips ready." if session.clips else "")

    if not reply:
        reply = (f"{len(session.clips)} clips ready." if session.clips
                 else "Ho gaya." if steps else "Kuch nahi hua.")
    return {"reply": reply, "steps": steps, "clips": session.clips,
            "history": history + [{"role": "user", "content": user_text},
                                  {"role": "assistant", "content": reply}]}


# ---------------------------------------------------------------- CLI
def repl(model: str | None = None) -> int:
    if model:
        globals()["MODEL"] = model
    if not ensure_ollama():
        print("Ollama not available. Install it or run `ollama serve`.")
        return 1
    va.log(f"VA Chat - model={MODEL}. Type your brief, or 'exit'.")
    print("VA Chat ready. Example: \"is drive link se 3 clips banao 25 second, "
          "CTA 'WATCH STREAM HOUSE LA ON CWETV', hashtags #streamhouselacwe #streamhousela #cwetv\"\n")
    s = Session()
    hist: list[dict] = []
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit", "q"):
            break
        out = run_turn(s, text, hist, on_step=lambda n, a, o: print(f"   [{n}] {o[:110]}"))
        hist = out["history"]
        print(f"\nva > {out['reply']}\n")
        for c in out["clips"]:
            print(f"   {Path(c['path']).name}  {c['duration']}s  qa={c['qa_ok']}")
    if s.clips:
        print(f"\n{len(s.clips)} clips in {s.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(repl())
