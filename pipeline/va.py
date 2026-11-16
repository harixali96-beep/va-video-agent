#!/usr/bin/env python3
"""
VA - Ultimate Video Agent
One local CLI that fuses the capabilities of the best open-source video tools:

  ingest      -> yt-dlp                 (any URL / local file)
  transcribe  -> faster-whisper         (word-level timestamps)
  scenes      -> PySceneDetect          (shot boundaries)
  roughcut    -> auto-editor style      (silence / dead-air removal)
  energy      -> librosa                (excitement curve, laughter-ish peaks)
  highlights  -> scoring engine         (moment picking, no paid API)
  reframe     -> OpenCV face tracking   (16:9 -> 9:16 auto pan/crop)
  captions    -> libass + ASS           (word-by-word burned captions)
  brand       -> PIL overlay            (hook bar, logo slot, CTA, hashtags)
  render      -> ffmpeg                 (final 1080x1920 30fps master)
  studio      -> stdlib web server      (browser UI for all of the above)

Everything runs offline on CPU. No API keys.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

HOME = Path(__file__).resolve().parent
ASSETS = HOME / "assets"
WORK = Path(os.environ.get("VA_WORK", str(HOME / "work")))
WORK.mkdir(parents=True, exist_ok=True)

ANTON = "/usr/share/fonts/truetype/custom/Anton-Regular.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FACE_ONNX = str(ASSETS / "face_detection_yunet_2023mar.onnx")
FACE_XML = str(ASSETS / "haarcascade_frontalface_default.xml")

HOOK_WORDS = [
    "but", "never", "why", "how", "actually", "secret", "stopped", "money", "dollars",
    "million", "thousand", "worst", "best", "honestly", "look", "wait", "literally",
    "died", "fired", "quit", "exposed", "truth", "hate", "love", "crazy", "insane",
    "never", "first", "last", "broke", "rich", "fight", "yelled", "quit", "walked",
    "biggest", "mistake", "regret", "warning", "nobody", "everyone", "problem", "wrong",
]


# ---------------------------------------------------------------- utilities
def log(msg: str) -> None:
    print(f"[va] {msg}", file=sys.stderr, flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def run_stream(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def probe(path: str | Path) -> dict:
    out = sh(["ffprobe", "-v", "error", "-print_format", "json",
              "-show_format", "-show_streams", str(path)]).stdout
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    info = {
        "path": str(path),
        "duration": float(data["format"].get("duration") or 0),
        "width": int(v["width"]) if v else 0,
        "height": int(v["height"]) if v else 0,
        "has_audio": a is not None,
        "has_video": v is not None,
        "audio_rate": int(a["sample_rate"]) if a else 0,
    }
    fps = 30.0
    if v and v.get("r_frame_rate") and "/" in v["r_frame_rate"]:
        n, d = v["r_frame_rate"].split("/")
        try:
            fps = float(n) / float(d or 1)
        except ZeroDivisionError:
            pass
    info["fps"] = round(fps, 3) if 0 < fps < 240 else 30.0
    return info


def slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return (s[:n] or "clip")


# ---------------------------------------------------------------- ingest
def ingest(src: str, outdir: Path) -> Path:
    if re.match(r"^https?://", src):
        outdir.mkdir(parents=True, exist_ok=True)
        log(f"downloading {src}")
        sh(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
            "--merge-output-format", "mp4", "-o", str(outdir / "source.%(ext)s"), src])
        files = sorted(outdir.glob("source.*"))
        if not files:
            raise RuntimeError("download produced no file")
        return files[0]
    p = Path(src).expanduser()
    if not p.exists():
        raise FileNotFoundError(src)
    return p


# ---------------------------------------------------------------- transcript
def transcribe(video: Path, model_size: str = "small.en", cache_dir: Path | None = None) -> dict:
    cache = (cache_dir or video.parent) / "transcript.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("words"):
                log("transcript: cache hit")
                return data
        except Exception:
            pass
    from faster_whisper import WhisperModel
    log(f"transcribe: loading {model_size} (int8, cpu)")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(video), word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=1, condition_on_previous_text=False,
    )
    words, segs = [], []
    for seg in segments:
        segs.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        for w in (seg.words or []):
            t = w.word.strip()
            if not t:
                continue
            words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": t})
    data = {"language": getattr(info, "language", "en"), "segments": segs, "words": words}
    cache.write_text(json.dumps(data))
    log(f"transcribe: {len(words)} words, lang={data['language']}")
    return data


# ---------------------------------------------------------------- scenes
def detect_scenes(video: Path, threshold: float = 27.0) -> list[float]:
    try:
        from scenedetect import open_video, SceneManager, ContentDetector
    except Exception as e:  # pragma: no cover
        log(f"scenedetect unavailable: {e}")
        return []
    try:
        v = open_video(str(video))
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=15))
        sm.detect_scenes(v, show_progress=False)
        return [round(s[0].get_seconds(), 3) for s in sm.get_scene_list()]
    except Exception as e:
        log(f"scene detect failed: {e}")
        return []


# ---------------------------------------------------------------- audio energy
def energy_curve(video: Path, hop: float = 0.25) -> dict:
    try:
        import numpy as np
        import librosa
    except Exception as e:
        log(f"librosa unavailable: {e}")
        return {"hop": hop, "values": []}
    tmp = video.parent / "_audio.wav"
    sh(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-ac", "1", "-ar", "16000",
        "-vn", str(tmp)])
    y, sr = librosa.load(str(tmp), sr=16000, mono=True)
    tmp.unlink(missing_ok=True)
    if y.size == 0:
        return {"hop": hop, "values": []}
    n = max(1, int(hop * sr))
    rms = librosa.feature.rms(y=y, frame_length=n * 2, hop_length=n)[0]
    # zcr ~ noisiness (laughter / shouting tends to rise)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n * 2, hop_length=n)[0]
    vals = (rms / (rms.max() or 1.0) * 0.8 + zcr / (zcr.max() or 1.0) * 0.2)
    return {"hop": hop, "values": [round(float(v), 4) for v in vals]}


def energy_at(curve: dict, t0: float, t1: float) -> float:
    vals = curve.get("values") or []
    if not vals:
        return 0.5
    hop = curve["hop"]
    a, b = max(0, int(t0 / hop)), min(len(vals), int(t1 / hop) + 1)
    if b <= a:
        return 0.0
    return sum(vals[a:b]) / (b - a)


# ---------------------------------------------------------------- highlight planning
def _words_between(words, t0, t1):
    return [w for w in words if w["start"] >= t0 - 0.05 and w["end"] <= t1 + 0.35]


def plan(video: Path, transcript: dict, scenes: list[float], curve: dict,
         target: float = 30.0, count: int = 3, min_gap: float = 4.0) -> list[dict]:
    info = probe(video)
    dur = info["duration"]
    words = transcript.get("words") or []
    if target >= dur:
        return [{"start": 0.0, "end": round(dur, 2), "score": 1.0, "reason": "whole file"}]

    def score_window(t0: float, t1: float) -> tuple[float, dict]:
        ws = _words_between(words, t0, t1)
        span = t1 - t0
        wps = len(ws) / span if span else 0
        speech = min(wps / 3.2, 1.0) * 40
        cuts = len([s for s in scenes if t0 <= s <= t1])
        cutting = min(cuts / 5.0, 1.0) * 15
        e = energy_at(curve, t0, t1) * 25
        first_txt = " ".join(w["word"].lower() for w in ws[:14])
        hook = min(sum(1 for h in HOOK_WORDS if h in first_txt), 3) * 6
        # prefer windows that open on a sentence start (pause before it)
        boundary = 0.0
        if ws:
            prev_end = max([w["end"] for w in words if w["end"] <= ws[0]["start"] + 0.01] or [0])
            if ws[0]["start"] - prev_end > 0.25:
                boundary = 8.0
        total = speech + cutting + e + hook + boundary
        return total, {"wps": round(wps, 2), "cuts": cuts, "energy": round(e / 25, 2),
                       "hook": hook > 0, "clean_open": boundary > 0}

    step = 2.0
    cands = []
    t = 0.0
    while t + target <= dur - 0.2:
        s, meta = score_window(t, t + target)
        cands.append({"start": round(t, 2), "end": round(t + target, 2), "score": round(s, 2), "meta": meta})
        t += step
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:
        if len(picked) >= count:
            break
        if any(not (c["end"] + min_gap <= p["start"] or c["start"] - min_gap >= p["end"]) for p in picked):
            continue
        picked.append(c)
    picked.sort(key=lambda c: c["start"])
    for p in picked:
        p["reason"] = f"speech {p['meta']['wps']}wps, energy {p['meta']['energy']}, cuts {p['meta']['cuts']}"
    return picked


def snap_to_sentences(clip: dict, words: list[dict], pad: float = 0.18,
                      dur: float = 0.0, max_shift: float = 1.6) -> dict:
    """Nudge clip edges onto word boundaries so no sentence is cut in half.

    Only small shifts are allowed, and the clip never loses more than 20% of
    its planned length. Leading / trailing dead air is trimmed as a side effect.
    """
    if not words:
        return clip
    orig_len = clip["end"] - clip["start"]
    lo, hi = clip["start"], clip["end"]

    starts = [w for w in words if lo - max_shift <= w["start"] <= hi]
    ends = [w for w in words if lo <= w["end"] <= hi + max_shift]

    new_start, new_end = lo, hi
    if starts:
        w0 = min(starts, key=lambda w: abs(w["start"] - lo))
        if abs(w0["start"] - lo) <= max_shift:
            new_start = max(0.0, w0["start"] - pad)
    if ends:
        w1 = min(ends, key=lambda w: abs(w["end"] - hi))
        if abs(w1["end"] - hi) <= max_shift:
            new_end = w1["end"] + pad

    if new_end - new_start < orig_len * 0.8:
        new_end = new_start + orig_len
    if dur:
        new_end = min(new_end, dur - 0.05)
        new_start = max(0.0, min(new_start, new_end - 1.0))
    clip["start"] = round(new_start, 2)
    clip["end"] = round(new_end, 2)
    return clip


# ---------------------------------------------------------------- face tracking
def _face_detector(w: int, h: int):
    """YuNet (OpenCV 5) with Haar fallback for older OpenCV."""
    import cv2
    if hasattr(cv2, "FaceDetectorYN") and os.path.exists(FACE_ONNX):
        det = cv2.FaceDetectorYN.create(FACE_ONNX, "", (w, h), 0.6, 0.3, 50)
        det.setInputSize((w, h))
        return ("yunet", det)
    cascade = cv2.CascadeClassifier(FACE_XML)
    return ("haar", cascade)


def track_face(video: Path, t0: float, t1: float, sample_fps: float = 5.0) -> dict:
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"points": [], "src_w": 0, "src_h": 0, "found": 0, "fps": 30.0}
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    kind, det = _face_detector(W, H)
    pts, found = [], 0
    t, step = t0, 1.0 / sample_fps
    while t < t1:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        cx = cy = None
        if kind == "yunet":
            _, faces = det.detect(frame)
            if faces is not None and len(faces):
                b = max(faces, key=lambda f: f[2] * f[3])
                cx = (b[0] + b[2] / 2.0) / W
                cy = (b[1] + b[3] / 2.0) / H
        else:
            scale = 480.0 / max(H, 1)
            gray = cv2.equalizeHist(cv2.cvtColor(cv2.resize(frame, (int(W * scale), int(H * scale))),
                                                 cv2.COLOR_BGR2GRAY))
            boxes = det.detectMultiScale(gray, 1.12, 5, minSize=(28, 28))
            if len(boxes):
                b = max(boxes, key=lambda r: r[2] * r[3])
                cx = (b[0] + b[2] / 2.0) / gray.shape[1]
                cy = (b[1] + b[3] / 2.0) / gray.shape[0]
        if cx is not None:
            found += 1
        pts.append({"t": round(t - t0, 3), "cx": cx, "cy": cy})
        t += step
    cap.release()
    return {"points": pts, "src_w": W, "src_h": H, "found": found,
            "sample_fps": sample_fps, "fps": fps, "detector": kind}


def smooth_path(track: dict, clip_dur: float, hold: float = 0.45) -> dict:
    """Median + EMA smoothing over normalised face centres, per output frame."""
    import numpy as np
    pts = track.get("points") or []
    fps = track.get("fps") or 30.0
    n = max(1, int(round(clip_dur * fps)))
    if not pts:
        return {"fps": fps, "n": n, "cx": [0.5] * n, "cy": [0.5] * n}
    ts = np.array([p["t"] for p in pts])
    xs = np.array([p["cx"] if p["cx"] is not None else np.nan for p in pts], dtype=float)
    ys = np.array([p["cy"] if p["cy"] is not None else np.nan for p in pts], dtype=float)
    for arr in (xs, ys):
        idx = np.where(~np.isnan(arr))[0]
        if idx.size == 0:
            arr[:] = 0.5
        else:
            arr[:] = np.interp(np.arange(arr.size), idx, arr[idx])
    # median filter
    def med(a, k=5):
        pad = k // 2
        b = np.pad(a, pad, mode="edge")
        return np.array([np.median(b[i:i + k]) for i in range(a.size)])
    xs, ys = med(xs), med(ys)
    # EMA
    def ema(a, alpha=0.25):
        out = np.empty_like(a)
        acc = a[0]
        for i, v in enumerate(a):
            acc = alpha * v + (1 - alpha) * acc
            out[i] = acc
        return out
    xs, ys = ema(xs), ema(ys)
    frame_t = np.arange(n) / fps
    cx = np.interp(frame_t, ts, xs)
    cy = np.interp(frame_t, ts, ys)
    return {"fps": fps, "n": n,
            "cx": [round(float(v), 5) for v in cx],
            "cy": [round(float(v), 5) for v in cy]}


# ---------------------------------------------------------------- captions (ASS)
def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_captions(words: list[dict], t0: float, t1: float, out: Path,
                   highlight_color: str = "&H0000FFFF", base_color: str = "&H00FFFFFF",
                   font_size: int = 108, margin_v: int = 560, max_words: int = 3) -> int:
    inside = [w for w in words if w["end"] > t0 + 0.02 and w["start"] < t1 - 0.02]
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for w in inside:
        cur.append(w)
        end_punct = bool(re.search(r"[.!?,]$", w["word"]))
        if len(cur) >= max_words or end_punct or (cur and w["end"] - cur[0]["start"] > 1.9):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Anton,{font_size},{base_color},&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,11,0,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for ch in chunks:
        s = max(ch[0]["start"], t0) - t0
        e = min(ch[-1]["end"], t1) - t0
        if e - s < 0.08:
            e = s + 0.09
        best = None
        for w in ch:
            token = re.sub(r"[^a-z$]", "", w["word"].lower())
            if token in HOOK_WORDS or token.isdigit():
                best = w["word"]
                break
        parts = []
        for w in ch:
            txt = w["word"].upper().replace("{", "").replace("}", "")
            if best and w["word"] == best:
                parts.append(r"{\c" + highlight_color + "}" + txt + r"{\c" + base_color + "}")
            else:
                parts.append(txt)
        body = " ".join(parts)
        anim = r"{\fad(45,30)\fscx100\fscy100\t(0,90,\fscx114\fscy114)\t(90,190,\fscx100\fscy100)}"
        lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cap,,0,0,0,,{anim}{body}")
    out.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(chunks)


# ---------------------------------------------------------------- branding overlay
def build_overlay(out: Path, hook: str = "", logo: Path | None = None,
                  cta: str = "", hashtags: str = "", accent: str = "#FFD400",
                  hook_bar: bool = True, width: int = 1080, height: int = 1920) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()

    f_hook = font(ANTON, 74)
    f_pill = font(SANS, 34)
    f_tag = font(SANS, 32)

    # top hook bar (2-line wrap)
    if hook_bar and hook:
        text = hook.upper()
        max_w = width - 220
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if d.textlength(trial, font=f_hook) > max_w and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        lines = lines[:3]
        pad_x, pad_y, gap = 34, 22, 10
        tw = max(d.textlength(l, font=f_hook) for l in lines)
        line_h = f_hook.size + gap
        bh = int(pad_y * 2 + line_h * len(lines))
        bw = int(tw + pad_x * 2)
        x0, y0 = 60, 110
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=26, fill=(0, 0, 0, 215))
        d.rectangle([x0, y0, x0 + 14, y0 + bh], fill=accent)
        for i, l in enumerate(lines):
            d.text((x0 + pad_x, y0 + pad_y + i * line_h), l, font=f_hook,
                   fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))

    # logo slot / actual logo (top-right)
    box = 190
    lx, ly = width - box - 56, 110
    if logo and Path(logo).exists():
        try:
            lg = Image.open(logo).convert("RGBA")
            lg.thumbnail((box, box), Image.LANCZOS)
            img.alpha_composite(lg, (lx + (box - lg.width) // 2, ly + (box - lg.height) // 2))
        except Exception:
            logo = None
    if not (logo and Path(logo).exists()):
        d.rounded_rectangle([lx, ly, lx + box, ly + box], radius=22,
                            outline=(255, 255, 255, 120), width=3)
        f_badge = font(SANS, 26)
        d.text((lx + box / 2, ly + box / 2), "LOGO", font=f_badge,
               fill=(255, 255, 255, 140), anchor="mm")

    # bottom pill CTA
    if cta:
        tw = d.textlength(cta.upper(), font=f_pill)
        pw, ph = int(tw + 76), 74
        px = (width - pw) // 2
        py = height - 330
        d.rounded_rectangle([px, py, px + pw, py + ph], radius=ph // 2,
                            fill=(200, 20, 20, 235))
        d.text((width / 2, py + ph / 2 - 2), cta.upper(), font=f_pill,
               fill=(255, 255, 255, 255), anchor="mm")

    # hashtags
    if hashtags:
        tw = d.textlength(hashtags, font=f_tag)
        d.rounded_rectangle([(width - tw) / 2 - 22, height - 236,
                             (width + tw) / 2 + 22, height - 174],
                            radius=14, fill=(0, 0, 0, 170))
        d.text((width / 2, height - 205), hashtags, font=f_tag,
               fill=(255, 255, 255, 240), anchor="mm")

    img.save(out)
    return out


# ---------------------------------------------------------------- render
def render_clip(video: Path, clip: dict, out: Path, brand: dict,
                words: list[dict], logo: Path | None = None,
                vertical: bool = True, target_w: int = 1080, target_h: int = 1920,
                progress: callable | None = None) -> Path:
    import cv2
    info = probe(video)
    t0, t1 = clip["start"], clip["end"]
    dur = max(0.4, t1 - t0)

    if vertical:
        track = track_face(video, t0, t1)
        path = smooth_path(track, dur)
    else:
        track, path = {"fps": info["fps"], "src_w": info["width"], "src_h": info["height"]}, None

    ass = out.with_suffix(".ass")
    build_captions(words, t0, t1, ass,
                   font_size=brand.get("font_size", 108),
                   highlight_color=brand.get("highlight", "&H0000FFFF"),
                   base_color=brand.get("base_color", "&H00FFFFFF"),
                   margin_v=brand.get("caption_margin_v", 560))
    overlay = None
    if brand.get("hook") or brand.get("cta") or brand.get("hashtags") or logo:
        overlay = out.with_name(out.stem + "_overlay.png")
        build_overlay(overlay, hook=brand.get("hook", ""), logo=logo,
                      cta=brand.get("cta", ""), hashtags=brand.get("hashtags", ""),
                      accent=brand.get("accent", "#FFD400"),
                      hook_bar=brand.get("hook_bar", True))

    W, H = info["width"], info["height"]
    tw, th = (target_w, target_h) if vertical else (W // 2 * 2, H // 2 * 2)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{tw}x{th}",
           "-r", str(int(round(path["fps"] if path else info["fps"]))), "-i", "pipe:0",
           "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(video)]
    if overlay:
        cmd += ["-loop", "1", "-i", str(overlay)]

    vf = f"ass=filename={ass}"
    if overlay:
        vf = f"[0:v]ass=filename={ass}[vc];[vc][2:v]overlay=0:0:shortest=1[vout]"
        maps = ["-map", "[vout]"]
    else:
        vf = f"[0:v]ass=filename={ass}[vout]"
        maps = ["-map", "[vout]"]

    cmd += ["-filter_complex", vf, *maps]
    if info["has_audio"]:
        cmd += ["-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", str(out)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video))
    fps = info["fps"] or 30.0
    if vertical:
        crop_w_src = int(round(H * target_w / target_h))
        crop_w_src = min(crop_w_src, W)
        scale_x = tw / crop_w_src
        half = crop_w_src / 2.0
    n_frames = int(round(dur * (path["fps"] if path else fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    written = 0
    try:
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if vertical:
                cx = path["cx"][min(i, path["n"] - 1)] * W
                cy = path["cy"][min(i, path["n"] - 1)] * H
                x0 = int(round(cx - half))
                x0 = max(0, min(x0, W - crop_w_src))
                # keep some headroom: bias the vertical window when source is taller than crop
                crop_h_src = H
                y0 = 0
                region = frame[y0:y0 + crop_h_src, x0:x0 + crop_w_src]
                if scale_x < 1.001 and region.shape[1] == tw and region.shape[0] == th:
                    pass
                frame = cv2.resize(region, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            else:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            try:
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, ValueError):
                break
            written += 1
            if progress and written % 30 == 0:
                progress(written / max(n_frames, 1))
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode("utf-8", "ignore")
        code = proc.wait()
        cap.release()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}): {err[-1500:]}")
    if overlay:
        overlay.unlink(missing_ok=True)
    return out



# ---------------------------------------------------------------- self check
def verify_clip(path: Path, want_w: int = 1080, want_h: int = 1920) -> dict:
    """Render QA: the agent refuses to call a clip done without checking it."""
    info = probe(path)
    issues = []
    if (info["width"], info["height"]) != (want_w, want_h):
        issues.append(f"resolution {info['width']}x{info['height']} != {want_w}x{want_h}")
    if info["duration"] < 1.0:
        issues.append("duration under 1s")
    if not info["has_audio"]:
        issues.append("no audio stream")
    # captions actually burned in? sample several frames in the caption band
    try:
        import numpy as np
        from PIL import Image
        best = 0
        for frac in (0.25, 0.5, 0.75, 0.15):
            tmp = path.with_suffix(".qa.jpg")
            sh(["ffmpeg", "-y", "-v", "error", "-ss", f"{info['duration'] * frac:.2f}",
                "-i", str(path), "-frames:v", "1", str(tmp)])
            im = np.array(Image.open(tmp).convert("RGB")).astype(int)
            band = im[int(want_h * 0.58):int(want_h * 0.84)]
            best = max(best, int(((band[:, :, 0] > 235) & (band[:, :, 1] > 235) &
                                  (band[:, :, 2] > 235)).sum()))
            tmp.unlink(missing_ok=True)
            if best >= 400:
                break
        info["caption_px"] = best
        if best < 400:
            issues.append("caption text not detected on sampled frames")
    except Exception as e:
        info["caption_px"] = -1
        issues.append(f"frame QA failed: {e}")
    info["ok"] = not issues
    info["issues"] = issues
    return info


# ---------------------------------------------------------------- clip package
def write_post_pack(out: Path, clip: dict, title: str, caption: str, hashtags: str) -> Path:
    pack = out.with_suffix(".txt")
    pack.write_text(
        f"TITLE: {title}\n\nCAPTION:\n{caption}\n\nHASHTAGS: {hashtags}\n\n"
        f"SOURCE RANGE: {clip['start']}s -> {clip['end']}s "
        f"({round(clip['end'] - clip['start'], 1)}s)\n",
        encoding="utf-8")
    return pack


def make_hook(clip: dict, words: list[dict], fallback: str = "") -> str:
    ws = [w for w in words if w["start"] >= clip["start"] - 0.2 and w["end"] <= clip["start"] + 4.5]
    if not ws:
        return fallback
    txt = " ".join(w["word"] for w in ws).strip(" ,.-")
    txt = re.sub(r"\s+", " ", txt)
    return (txt[:62].rsplit(" ", 1)[0] + "…") if len(txt) > 62 else txt


# ---------------------------------------------------------------- pipeline
def auto(args) -> dict:
    job = WORK / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:5]}"
    job.mkdir(parents=True, exist_ok=True)
    video = ingest(args.source, job)
    info = probe(video)
    log(f"source {info['width']}x{info['height']} {info['duration']:.1f}s")
    transcript = transcribe(video, args.model, cache_dir=job)
    scenes = detect_scenes(video)
    curve = energy_curve(video)
    clips = plan(video, transcript, scenes, curve, target=args.length, count=args.count)
    log(f"planned {len(clips)} clips")
    results = []
    logo = Path(args.logo) if args.logo else None
    for i, c in enumerate(clips, 1):
        c = snap_to_sentences(c, transcript.get("words", []), dur=info["duration"])
        hook = args.hook or make_hook(c, transcript.get("words", []))
        brand = {"hook": hook, "cta": args.cta, "hashtags": args.hashtags,
                 "hook_bar": not args.no_hook_bar, "highlight": args.highlight}
        out = job / f"clip{i:02d}_{slug(hook)}.mp4"
        log(f"render {i}/{len(clips)} {c['start']}->{c['end']}")
        render_clip(video, c, out, brand, transcript.get("words", []), logo=logo)
        title = hook.rstrip("…") or f"Clip {i}"
        pack = write_post_pack(out, c, title, f"{title}\n\n{args.hashtags}", args.hashtags)
        qa = verify_clip(out)
        results.append({"clip": str(out), "pack": str(pack), "range": [c["start"], c["end"]],
                        "hook": hook, "duration": round(c["end"] - c["start"], 2),
                        "qa_ok": qa["ok"], "qa_issues": qa["issues"]})
    manifest = job / "manifest.json"
    manifest.write_text(json.dumps({"source": str(video), "info": info, "clips": results,
                                    "scenes": len(scenes)}, indent=2))
    return {"job": str(job), "clips": results, "manifest": str(manifest)}


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="video-agent", description="Ultimate local video agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("auto", help="full pipeline: url/file -> finished vertical clips")
    a.add_argument("source")
    a.add_argument("--length", type=float, default=30.0)
    a.add_argument("--count", type=int, default=3)
    a.add_argument("--model", default="small.en")
    a.add_argument("--logo")
    a.add_argument("--cta", default="")
    a.add_argument("--hashtags", default="")
    a.add_argument("--hook", default="")
    a.add_argument("--highlight", default="&H0000FFFF")
    a.add_argument("--no-hook-bar", action="store_true")

    t = sub.add_parser("transcribe", help="word-level transcript")
    t.add_argument("source")
    t.add_argument("--model", default="small.en")

    an = sub.add_parser("analyze", help="scenes + energy + planned moments")
    an.add_argument("source")
    an.add_argument("--length", type=float, default=30.0)
    an.add_argument("--count", type=int, default=3)

    r = sub.add_parser("render", help="render one exact range")
    r.add_argument("source")
    r.add_argument("--start", type=float, required=True)
    r.add_argument("--end", type=float, required=True)
    r.add_argument("--hook", default="")
    r.add_argument("--cta", default="")
    r.add_argument("--hashtags", default="")
    r.add_argument("--logo")
    r.add_argument("--out", default="")

    s = sub.add_parser("studio", help="browser UI on a local port")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--host", default="127.0.0.1")

    ch = sub.add_parser("chat", help="talk to the AI agent that drives every tool")
    ch.add_argument("--model", default="")
    ch.add_argument("message", nargs="*", help="one-shot brief instead of the REPL")

    sub.add_parser("agents", help="show the upstream agents this CLI fuses")

    args = p.parse_args(argv)

    if args.cmd == "agents":
        manifest = HOME / "agents.json"
        data = json.loads(manifest.read_text())
        print(f"{len(data['agents'])} upstream agents fused into 1 CLI\n")
        for x in data["agents"]:
            print(f"  {x['repo']:<44} -> {x['module']}")
        return 0

    if args.cmd == "auto":
        res = auto(args)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "transcribe":
        v = ingest(args.source, WORK / "tx")
        data = transcribe(v, args.model)
        print(json.dumps({"words": len(data["words"]), "segments": len(data["segments"]),
                          "language": data["language"]}, indent=2))
        return 0

    if args.cmd == "analyze":
        v = ingest(args.source, WORK / "an")
        tr = transcribe(v, "small.en")
        sc = detect_scenes(v)
        cu = energy_curve(v)
        cl = plan(v, tr, sc, cu, target=args.length, count=args.count)
        print(json.dumps({"scenes": len(sc), "clips": cl}, indent=2))
        return 0

    if args.cmd == "render":
        v = Path(args.source)
        tr = transcribe(v, "small.en")
        clip = {"start": args.start, "end": args.end}
        out = Path(args.out) if args.out else WORK / f"range-{int(args.start)}-{int(args.end)}.mp4"
        brand = {"hook": args.hook, "cta": args.cta, "hashtags": args.hashtags}
        render_clip(v, clip, out, brand, tr["words"],
                    logo=Path(args.logo) if args.logo else None)
        print(json.dumps({"out": str(out)}))
        return 0

    if args.cmd == "chat":
        import va_chat
        if args.message:
            text = " ".join(args.message)
            sess = va_chat.Session()
            out = va_chat.run_turn(sess, text,
                                   on_step=lambda n, a, o: print(f"   [{n}] {o[:140]}"))
            print(out["reply"])
            for c in out["clips"]:
                print(f"   {c['path']}  {c['duration']}s  qa={c['qa_ok']}")
            print(json.dumps({"job": str(sess.dir), "clips": out["clips"]}, indent=1))
            return 0
        return va_chat.repl(args.model or None)

    if args.cmd == "studio":
        os.environ["VA_STUDIO_PORT"] = str(args.port)
        import va_studio
        va_studio.serve(host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
