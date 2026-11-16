#!/usr/bin/env python3
"""
batch.py — one command: footage in, N branded Stream House LA clips out.

Transcribes the source once, scores every spoken moment, picks the strongest
non-overlapping windows, and renders each one through make_clip.py's branding +
caption pipeline. Writes a ready-to-paste title/caption file per clip.

Usage:
  python3 batch.py --src episode.mp4 --count 3 --min 18 --max 26 --outdir out
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_clip as mc  # noqa: E402

# words that tend to sit in a clip-worthy reality-TV beat
HOT = {
    "caught", "secret", "lying", "liar", "cheated", "fight", "fought", "screaming",
    "crying", "crazy", "insane", "never", "nobody", "everybody", "stop", "wait",
    "what", "why", "how", "kiss", "kissed", "love", "hate", "dating", "girlfriend",
    "boyfriend", "left", "quit", "kicked", "banned", "arrested", "drunk", "poison",
    "fake", "real", "exposed", "confront", "called", "screamed", "actually",
}


def speech_words(src, lang=None):
    return mc.transcribe(src, lang)


def segments(words, max_gap=0.7, min_words=6, max_words=90):
    """Split word stream into spoken segments on natural pauses."""
    out, cur = [], []
    for w in words:
        if cur and (w.start - cur[-1].end) > max_gap:
            if len(cur) >= min_words:
                out.append(cur)
            cur = []
        cur.append(w)
    if len(cur) >= min_words:
        out.append(cur)
    out = [s for s in out if len(s) <= max_words]
    return out


def score(seg, dur):
    text = " ".join(w.word.strip() for w in seg).lower()
    toks = re.findall(r"[a-z']+", text)
    if not toks:
        return -99
    hits = sum(1 for t in toks if t in HOT)
    density = hits / max(1, len(toks)) * 100
    s = density * 0.9
    # reward segments whose length is near the target window
    # (top clips in this campaign: median 8.5s, 7 of 10 under 12s)
    s -= abs(dur - 10) * 0.8
    # speech rhythm: words per second close to 3.2 reads best on short form
    wps = len(toks) / max(0.1, dur)
    s -= abs(wps - 3.2) * 2.0
    if "?" in text or "!" in text:
        s += 1.5
    return s


def windows(segs, lo, hi, max_gap=2.5):
    """Chain consecutive spoken segments until the window is clip length."""
    out = []
    for i in range(len(segs)):
        acc = list(segs[i])
        J = i
        while True:
            dur = acc[-1].end - acc[0].start
            if lo <= dur <= hi:
                out.append(list(acc))          # every valid end point is a candidate
            if dur >= hi or J + 1 >= len(segs):
                break
            if segs[J + 1][0].start - segs[J][-1].end > max_gap:
                break
            J += 1
            acc += segs[J]
    return out


def pick(segs, count, lo, hi, gap=5.0):
    scored = []
    for seg in windows(segs, lo, hi):
        dur = seg[-1].end - seg[0].start
        if dur < lo:
            continue
        # trim overly long segments to the target window from their start
        if dur > hi:
            keep, end = [], seg[0].start + hi
            for w in seg:
                if w.end <= end:
                    keep.append(w)
            if len(keep) < 6:
                continue
            seg = keep
            dur = seg[-1].end - seg[0].start
            if dur < lo:
                continue
        scored.append((score(seg, dur), dur, seg))
    scored.sort(key=lambda x: -x[0])

    chosen = []
    for sc, dur, seg in scored:
        if len(chosen) >= count:
            break
        a, b = seg[0].start, seg[-1].end
        if all(b + gap < c[2][0].start or a - gap > c[2][-1].end for c in chosen for _ in [0]):
            chosen.append((sc, dur, seg))
    chosen.sort(key=lambda x: x[2][0].start)
    return chosen


TRAIL = {"and", "but", "or", "because", "so", "the", "a", "an", "then",
         "that", "to", "of", "was", "is", "when", "while"}


def best_sentence(seg):
    """Split the window into sentences and return the most charged one."""
    groups, cur = [], []
    for w in seg:
        if cur and (w.start - cur[-1].end) > 0.55:
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    def heat(g):
        toks = re.findall(r"[a-z']+", " ".join(w.word.strip().lower() for w in g))
        return sum(1 for t in toks if t in HOT) + (1 if toks and toks[0] in HOT else 0)
    return max(groups, key=heat) if groups else seg


def hook_from(seg, max_words=7):
    seg = best_sentence(seg)
    ws = [w.word.strip() for w in seg[:max_words]]
    h = " ".join(ws).strip(" ,.!?").upper()
    return h[:60]


def title_from(seg):
    """A clean, postable title built from the strongest sentence in the window."""
    sent = best_sentence(seg)
    txt = " ".join(w.word.strip() for w in sent).strip(" ,.!?")
    if len(txt) > 70:
        txt = txt[:70]
        cut = max(txt.rfind(","), txt.rfind(";"))
        txt = txt[:cut] if cut > 30 else txt.rsplit(" ", 1)[0]
    # never end on a hanging joining word
    while True:
        parts = txt.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].lower() in TRAIL:
            txt = parts[0]
        else:
            break
    txt = txt.strip(" ,.!?")
    return (txt[0].upper() + txt[1:]) if txt else "Stream House LA moment"


def render(src, start, end, hook, title, out, crop="center", lang=None):
    words = mc.transcribe_slice(src, start, end, lang) if hasattr(mc, "transcribe_slice") else None
    argv = ["python3", os.path.join(HERE, "make_clip.py"), "--src", src,
            "--start", f"{start:.2f}", "--end", f"{end:.2f}",
            "--hook", hook, "--title", title, "--out", out, "--crop", crop]
    if lang:
        argv += ["--lang", lang]
    p = subprocess.run(argv, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:])
    return p.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--min", dest="lo", type=float, default=7)
    ap.add_argument("--max", dest="hi", type=float, default=16)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--list-only", action="store_true",
                    help="print the chosen windows without rendering")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    words = speech_words(a.src, a.lang)
    segs = segments(words)
    chosen = pick(segs, a.count, a.lo, a.hi)
    if not chosen:
        raise SystemExit("no usable spoken segments found (is there audio?)")

    manifest = []
    for i, (sc, dur, seg) in enumerate(chosen, 1):
        hook, title = hook_from(seg), title_from(seg)
        out = os.path.join(a.outdir, f"clip{i}.mp4")
        print(f"[{i}] {seg[0].start:7.2f} -> {seg[-1].end:7.2f} ({dur:5.1f}s) "
              f"score {sc:5.1f} | {hook}")
        if not a.list_only:
            print("    " + render(a.src, seg[0].start, seg[-1].end, hook, title,
                                  out, lang=a.lang))
        manifest.append({"clip": i, "start": round(seg[0].start, 2),
                         "end": round(seg[-1].end, 2), "hook": hook,
                         "title": title, "file": os.path.basename(out)})

    with open(os.path.join(a.outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n{len(manifest)} clips -> {a.outdir}/")


if __name__ == "__main__":
    main()
