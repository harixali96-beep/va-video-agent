#!/usr/bin/env python3
"""
sh_job.py — Stream House LA campaign job (GitHub Actions).

Reads the request from the issue body or workflow inputs, resolves the source
(Google Drive / YouTube / TikTok / IG / direct mp4 / repo path), transcribes once,
picks the strongest 7-16s spoken windows, renders each with the campaign template
(hook bar, 61% captions with red emphasis, CTA pill, hashtags, logo slot) and
writes shclips.json so the site gallery picks them up.

Env:
  ISSUE_BODY / IN_SOURCE / IN_COUNT / IN_MIN / IN_MAX / IN_CTA / IN_HASHTAGS / IN_HOOK
  WORKER_INDEX, WORKERS (parallel runners), GITHUB_RUN_ID
"""
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent          # pipeline/sh
REPO = HERE.parent.parent                        # repo root
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "pipeline"))

import batch as bt                               # noqa: E402
import make_clip as mc                           # noqa: E402
import build_template as tpl                     # noqa: E402

INDEX = REPO / "shclips.json"
MAX_CLIPS = 10


def field(body, key, default=""):
    m = re.search(rf"^\s*{key}\s*[:\-]\s*(.+)$", body, re.I | re.M)
    return m.group(1).strip() if m else default


def need(key, default=""):
    v = os.environ.get(key)
    return (v.strip() if v and v.strip() else None) or default


def parse():
    body = os.environ.get("ISSUE_BODY") or ""
    src = need("IN_SOURCE") or field(body, "source")
    if not src:
        m = re.search(r"https?://\S+", body) or re.search(
            r"[\w./-]+\.(?:mp4|mov|mkv|webm)", body)
        src = m.group(0) if m else ""
    count = int(need("IN_COUNT") or field(body, "count", "3"))
    lo = float(need("IN_MIN") or field(body, "min", "7"))
    hi = float(need("IN_MAX") or field(body, "max", "16"))
    cta = need("IN_CTA") or field(body, "cta")
    tags = need("IN_HASHTAGS") or field(body, "hashtags")
    hook = need("IN_HOOK") or field(body, "hook")
    return src.strip(), max(1, min(count, MAX_CLIPS)), lo, hi, cta, tags, hook


def resolve_source(src, work):
    """Local repo path, else the agent's own resolver (Drive / yt-dlp / direct)."""
    local = Path(src)
    if local.exists():
        return local
    import va_chat
    return Path(va_chat.resolve_source(src, work / "in"))


def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration,size:stream=codec_type,width,height",
                        "-of", "json", str(path)], capture_output=True, text=True)
    try:
        d = json.loads(p.stdout)
        dur = float(d.get("format", {}).get("duration") or 0)
        size = int(d.get("format", {}).get("size") or 0)
        v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
        a = any(s.get("codec_type") == "audio" for s in d.get("streams", []))
        return {"duration": dur, "size": size,
                "width": v.get("width"), "height": v.get("height"), "audio": a}
    except Exception:
        return {"duration": 0, "size": 0, "width": None, "height": None, "audio": False}


def qa(path):
    info = probe(path)
    issues = []
    if info["duration"] < 3:
        issues.append("too short")
    if not info["audio"]:
        issues.append("no audio")
    if info["width"] != 1080 or info["height"] != 1920:
        issues.append(f"size {info['width']}x{info['height']}")
    if info["size"] < 120_000:
        issues.append("file too small")
    return {"ok": not issues, "issues": issues, **info}


def slug(s, n=48):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:n] or "clip"


def load_index():
    if INDEX.exists():
        try:
            return json.loads(INDEX.read_text())
        except Exception:
            pass
    return {"jobs": []}


def main():
    src, count, lo, hi, cta, tags, hookline = parse()
    job_id = os.environ.get("GITHUB_RUN_ID", "local")
    wi = int(os.environ.get("WORKER_INDEX", "0") or 0)
    wn = max(1, int(os.environ.get("WORKERS", "1") or 1))
    if wn > 1:
        job_id = f"{job_id}-w{wi}"

    entry = {"id": job_id,
             "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "source": src, "count": count, "min": lo, "max": hi,
             "cta": cta or tpl.CTA, "hashtags": tags or tpl.HASHTAGS,
             "clips": [], "error": ""}
    out_dir = REPO / "clips" / f"sh-{job_id}"
    try:
        if not src:
            raise RuntimeError("no source link in the request")
        if {"{", "}"} & set(cta or ""):
            raise RuntimeError("CTA has braces, which ffmpeg would treat as filters")

        # branding is generated on the runner so a request can change CTA/hashtags
        os.environ["SH_CTA"] = entry["cta"]
        os.environ["SH_HASHTAGS"] = entry["hashtags"]
        if hookline:
            os.environ["SH_HOOK"] = hookline
        tpl.build_overlay()
        print(f"[sh] branding built: CTA={entry['cta']!r}", flush=True)

        work = Path(os.environ.get("SH_WORK", "/tmp/sh"))
        work.mkdir(parents=True, exist_ok=True)
        video = resolve_source(src, work)
        info = probe(video)
        print(f"[sh] source {video} {info}", flush=True)
        if not info["audio"]:
            raise RuntimeError("source has no audio track, nothing to caption")

        words = mc.transcribe(str(video))
        if not words:
            raise RuntimeError("no speech detected in the source")
        segs = bt.segments(words)
        wanted = count * wn
        chosen = bt.pick(segs, wanted, lo, hi)
        if not chosen:
            raise RuntimeError(f"no spoken window between {lo:.0f}s and {hi:.0f}s found")
        if wn > 1:
            chosen = chosen[wi::wn][:count]

        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (sc, dur, seg) in enumerate(chosen, 1):
            hook = hookline or bt.hook_from(seg)
            title = bt.title_from(seg)
            start, end = seg[0].start, seg[-1].end
            name = f"clip{i:02d}_{slug(hook)}"
            out = out_dir / f"{name}.mp4"
            print(f"[sh] {i}/{len(chosen)} {start:.2f}-{end:.2f} ({dur:.1f}s) "
                  f"score {sc:.1f} | {hook}", flush=True)
            bt.render(str(video), start, end, hook, title, str(out))
            q = qa(out)
            pack = out.with_suffix(".txt")
            entry["clips"].append({
                "file": f"clips/sh-{job_id}/{out.name}",
                "pack": f"clips/sh-{job_id}/{pack.name}" if pack.exists() else "",
                "hook": hook, "title": title,
                "caption": f"{title} {entry['cta']}",
                "hashtags": entry["hashtags"],
                "start": round(start, 2), "end": round(end, 2),
                "duration": round(dur, 2),
                "qa_ok": q["ok"], "qa_issues": q["issues"]})
            print(f"    -> {out.name} qa={'ok' if q['ok'] else q['issues']}", flush=True)
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job.json").write_text(json.dumps(entry, indent=1))

    idx = load_index()
    jobs = [j for j in idx.get("jobs", []) if j.get("id") != job_id]
    idx["jobs"] = ([entry] + jobs)[:15]
    INDEX.write_text(json.dumps(idx, indent=1))
    print(json.dumps({"id": job_id, "clips": len(entry["clips"]),
                      "error": entry["error"]}, indent=1))
    if entry["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
