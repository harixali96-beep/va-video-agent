#!/usr/bin/env python3
"""
make_clip.py — Stream House LA campaign clip renderer (Content Rewards 5f6b6ac5).

Takes a segment from the campaign's reference footage and renders a finished
1080x1920 / 30fps vertical clip with:
  * 16:9 -> 9:16 face-safe crop (centre crop, or --crop left|right|center)
  * word-group captions (Anton, white with yellow emphasis word, black stroke)
  * the branded overlay template (hook bar, CTA pill, hashtags, logo slot)
  * a ready-to-paste title + caption file

Usage:
  python3 make_clip.py --src episode.mp4 --start 88.5 --end 118.5 \
      --hook "SHE THOUGHT NOBODY WAS WATCHING" --title "..." --out clip1.mp4

  python3 make_clip.py --src ... --no-captions        # skip whisper (fast layout test)
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANTON = "/usr/share/fonts/truetype/custom/Anton-Regular.ttf"
HASHTAGS = os.environ.get("SH_HASHTAGS") or "#streamhouselacwe #streamhousela #cwetv"
TYPE_TAIL = os.environ.get("SH_CTA") or "Watch the full ep on @Chicken Wrap ENT"
CAPTION_Y = 1180          # px from top = 61% of frame: measured centre of the winners' caption band
EMPH = r"{\c&H002800E6&}"   # campaign red for the standout word (measured from top clips)
FPS = 30
MODEL = os.environ.get("WHISPER_MODEL", "base")


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        sys.stderr.write(" ".join(cmd) + "\n" + (p.stderr or "")[-3000:] + "\n")
        raise SystemExit(1)
    return p


def ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def group_words(words, max_words=3, max_gap=0.45, max_span=2.2):
    """Group whisper word objects into short caption chunks."""
    chunks, cur = [], []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            span = w.end - cur[0].start
            if gap > max_gap or len(cur) >= max_words or span > max_span:
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def transcribe(path, language=None):
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(path, word_timestamps=True, vad_filter=True,
                               language=language)
    words = []
    for s in segs:
        for w in (s.words or []):
            txt = w.word.strip()
            if txt:
                words.append(w)
    return words


def build_ass(words, offset, path):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Anton,86,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,11,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for chunk in group_words(words):
        start = max(0.0, chunk[0].start - offset)
        end = max(start + 0.35, chunk[-1].end - offset)
        # emphasis = longest word in the chunk
        emph = max(chunk, key=lambda w: len(w.word.strip()))
        parts = []
        for w in chunk:
            t = w.word.strip().upper().replace("{", "(").replace("}", ")")
            if w is emph:
                parts.append(EMPH + t + r"{\c&HFFFFFF&}")
            else:
                parts.append(t)
        text = " ".join(parts)
        lines.append(
            f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,"
            f"{{\\pos(540,{CAPTION_Y})\\fad(60,0)}}{text}"
        )
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--hook", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default="center", choices=["center", "left", "right"])
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--lang", default=None)
    a = ap.parse_args()

    dur = a.end - a.start
    if dur <= 0:
        raise SystemExit("end must be after start")

    tmp = os.path.join(HERE, "_seg.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-ss", str(a.start), "-to", str(a.end),
         "-i", a.src, "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-c:a", "aac", "-b:a", "160k", tmp])

    ass = None
    if not a.no_captions:
        words = transcribe(tmp, a.lang)
        if words:
            ass = os.path.join(HERE, "_caps.ass")
            build_ass(words, a.start, ass)

    # 16:9 -> 9:16 crop, position shifts by --crop
    xexpr = {"center": "(iw-ih*9/16)/2",
             "left": "iw*0.12",
             "right": "iw-ih*9/16-iw*0.12"}[a.crop]
    vf = (f"crop=ih*9/16:ih:{xexpr}:0,"
          f"scale=1080:1920:flags=lanczos,fps={FPS},"
          f"overlay=0:0" if False else
          f"crop=ih*9/16:ih:{xexpr}:0,"
          f"scale=1080:1920:flags=lanczos,fps={FPS}")

    filt = f"[0:v]{vf}[base];[base][1:v]overlay=0:0[v]"
    if ass:
        filt += f";[v]ass={ass}[vout]"
        vmap = "[vout]"
    else:
        vmap = "[v]"

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", tmp,
           "-i", os.path.join(HERE, "overlay.png"),
           "-filter_complex", filt, "-map", vmap, "-map", "0:a?",
           "-c:v", "libx264", "-crf", "19", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", a.out]
    run(cmd)

    meta = os.path.splitext(a.out)[0] + ".txt"
    with open(meta, "w") as fh:
        fh.write(f"TITLE:\n{a.title or a.hook or ''}\n\n")
        fh.write(f"CAPTION:\n{a.title or a.hook or ''} {TYPE_TAIL}\n\n{HASHTAGS}\n")
        fh.write(f"\nSTORY POST (required): IG story with the campaign link, "
                 f"TikTok story with the campaign QR code.\n")
        fh.write(f"SOURCE: {os.path.basename(a.src)} {a.start}-{a.end} "
                 f"({dur:.1f}s)\n")

    for f in (tmp, ass):
        if f and os.path.exists(f):
            os.remove(f)

    print(a.out, f"({dur:.1f}s)", "captions:" + ("yes" if ass else "no"))


if __name__ == "__main__":
    main()
