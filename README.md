# VA Video Agent - free hosted edition

Static site on GitHub Pages + rendering on GitHub Actions (2-core runner, free for public repos).
No API keys, no paid services.

**Site:** https://harisaliharis.github.io/va-video-agent/

## How a job runs

1. Fill the form on the site - it opens a prefilled GitHub issue (`CLIP JOB`).
2. Submitting the issue triggers `.github/workflows/clip.yml`.
3. The runner downloads the footage (Google Drive / YouTube / direct mp4), transcribes it,
   scores the best moments, reframes to 9:16 with face tracking, burns word-by-word
   captions + hook bar + CTA + hashtags, renders 1080x1920 30fps and self-checks the output.
4. Clips are committed to `clips/`, the gallery (`clips.json`) refreshes, and the issue
   gets a comment when it is done.

## Manual run

Actions -> "Clip job" -> Run workflow -> paste the source link.

Body format for an issue:

```
source: https://drive.google.com/file/d/XXXX/view
length: 30
count: 2
cta: WATCH STREAM HOUSE LA ON CWETV
hashtags: #streamhouselacwe #streamhousela #cwetv
```

## Pipeline

`pipeline/va.py` (the engine) + `pipeline/clip_job.py` (job runner).
Fuses faster-whisper, PySceneDetect, librosa, OpenCV YuNet, libass, ffmpeg, yt-dlp, gdown.
