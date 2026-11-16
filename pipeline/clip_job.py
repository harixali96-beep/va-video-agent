#!/usr/bin/env python3
"""Render clips on a GitHub Actions runner from an issue body or workflow inputs."""
import json, os, re, shutil, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import va
import va_chat

REPO_ROOT = ROOT.parent
CLIPS_DIR = REPO_ROOT / "clips"
INDEX = REPO_ROOT / "clips.json"


def field(body, key, default=""):
    # spaces/tabs only: \s would swallow the newline and read the next line
    # whenever a field is submitted empty.
    m = re.search(rf"^[ \t]*{key}[ \t]*[:\-][ \t]*(.*)$", body, re.I | re.M)
    return (m.group(1).strip() if m else "") or default


def parse():
    body = os.environ.get("ISSUE_BODY") or ""
    src = (os.environ.get("IN_SOURCE") or field(body, "source")).strip()
    if not src:
        m = re.search(r"https?://\S+", body) or re.search(r"[\w./-]+\.(?:mp4|mov|mkv|webm)", body)
        src = m.group(0) if m else ""
    length = float(os.environ.get("IN_LENGTH") or field(body, "length", "30"))
    count = int(os.environ.get("IN_COUNT") or field(body, "count", "2"))
    cta = os.environ.get("IN_CTA") or field(body, "cta", "")
    tags = os.environ.get("IN_HASHTAGS") or field(body, "hashtags", "")
    hookline = os.environ.get("IN_HOOK") or field(body, "hook", "")
    return src, length, count, cta, tags, hookline


def load_index():
    if INDEX.exists():
        try:
            return json.loads(INDEX.read_text())
        except Exception:
            pass
    return {"jobs": []}


def save_index(idx):
    INDEX.write_text(json.dumps(idx, indent=1))


def main():
    src, length, count, cta, tags, hookline = parse()
    job_id = os.environ.get("GITHUB_RUN_ID", "local")
    wi = int(os.environ.get("WORKER_INDEX", "0") or 0)
    wn = max(1, int(os.environ.get("WORKERS", "1") or 1))
    if wn > 1:
        job_id = f"{job_id}-w{wi}"
    entry = {"id": job_id, "when": os.environ.get("JOB_DATE") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": src,
             "length": length, "count": count, "cta": cta, "hashtags": tags,
             "worker": wi, "workers": wn, "clips": [], "error": ""}
    idx = load_index()
    try:
        if not src:
            raise RuntimeError("no source link found in the request")
        work = Path(os.environ.get("VA_WORK", "/tmp/va"))
        work.mkdir(parents=True, exist_ok=True)
        local = Path(src)
        if local.exists():
            video = local
        else:
            video = Path(va_chat.resolve_source(src, work / "in"))
        va.log(f"source {video} {va.probe(video)}")

        transcript = va.transcribe(video, os.environ.get("VA_ASR", "base.en"), cache_dir=work)
        scenes = va.detect_scenes(video)
        curve = va.energy_curve(video)
        plan = va.plan(video, transcript, scenes, curve, target=length, count=count * wn)
        if wn > 1:
            plan = plan[wi::wn][:count]
        out_dir = CLIPS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, clip in enumerate(plan, 1):
            clip = va.snap_to_sentences(clip, transcript["words"], dur=va.probe(video)["duration"])
            plan[i - 1] = clip
            hook = hookline or va.make_hook(clip, transcript["words"])
            out = out_dir / f"clip{i:02d}_{va.slug(hook)}.mp4"
            brand = {"hook": hook, "cta": cta, "hashtags": tags, "hook_bar": True,
                     "highlight": "&H0000FFFF"}
            va.render_clip(video, clip, out, brand, transcript["words"])
            qa = va.verify_clip(out)
            pack = va.write_post_pack(out, clip, hook.rstrip("\u2026"),
                                      f"{hook}\n\n{tags}", tags)
            entry["clips"].append({
                "file": f"clips/{job_id}/{out.name}", "pack": f"clips/{job_id}/{pack.name}",
                "hook": hook, "duration": round(clip["end"] - clip["start"], 2),
                "start": clip["start"], "end": clip["end"], "qa_ok": qa["ok"],
                "qa_issues": qa["issues"]})
            va.log(f"clip {i} {out.name} qa={qa['ok']}")
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        print(f"VA_ERROR_MESSAGE={entry['error']}")

    part_dir = REPO_ROOT / "clips.partial"
    part_dir.mkdir(exist_ok=True)
    (part_dir / f"{job_id}.json").write_text(json.dumps(entry, indent=1))
    if wn == 1:
        idx["jobs"] = ([entry] + [j for j in idx.get("jobs", []) if j.get("id") != job_id])[:25]
        save_index(idx)
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
