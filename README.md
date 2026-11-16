# VA Video Agent - free hosted edition

Static site on GitHub Pages + rendering on GitHub Actions (2-core runner, free for public repos).
No API keys, no paid services.

**Site:** https://harixali96-beep.github.io/va-video-agent/

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

## Editor (chat + timeline, in the browser)

`/editor/` — open it from the "Open editor" button on the site, or from any clip's
**edit karo** link (it loads that clip straight in).

- Type a Hinglish/English command in the chat panel. The command brain is deterministic,
  so it is instant and works offline: `first 12 seconds rakho`, `3 pe split`,
  `caption WAIT FOR IT 0.2 to 2`, `captions word by word for THIS IS CRAZY 0 to 2`,
  `whoosh at 0, pop at 2`, `hook BIG MOMENT`, `cta WATCH NOW`, `1.2x speed`, `9:16`, `export`.
- Or drag/trim/delete blocks on the timeline yourself (video, caption and SFX tracks,
  playhead, undo).
- Export renders in the browser with ffmpeg.wasm (1080x1920, 30fps, hook bar + captions +
  CTA + SFX mix) and gives you a download link. Nothing is uploaded anywhere.

## Parallel runners

The form has a **parallel runners** field (1-10). Each runner is a separate free
GitHub Actions machine rendering a different slice of the best moments, so 5 clips on
3 runners finishes roughly 3x faster. Results merge into the same gallery.
`workers: N` in an issue body works too.
