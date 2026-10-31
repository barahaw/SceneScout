# SceneScout

> A **local, private, on-device** (Windows + NVIDIA) scanner that mines a personal video library to find candidate "scene" ranges, using NudeNet plus an optional whole-frame NSFW classifier, with manual review and `.txt` export.

---

## ✨ Features

- **ماسحة تلقائية (Scan)** — استخراج إطارات بفاصل 1s (قابل للتغيير)، pass1 + pass2 "refine" عند 0.5s حول الضربات.
- **كشف مزدوج**:
  - **NudeNet** (`320n`, `640n`, …) — كشف أجزاء محددة (genitalia, breasts, …) بعتبات score/حجم box.
  - ** whole-frame classifier** (اختياري): `TostAI/nsfw-image-detection-large`, `viddexa/nsfw-detection-2-mini`, `Marqo/nsfw-image-detection-384`.
- \*\*جمع ف candle "تجمع بفجوة" (merge_gap) و×COMPREHENSIVE? expansion (before/after)، تصفية min_hits / high_score.
- **Tail mode scene-beban snaps** (snap to scene cuts via ffmpeg scene detect) when enabled.
- **Manual editing** — apply edit (range + locked), add/delete/toggle candidates, add manual rows.
- **Caching** — hashed cache keyed by path+size+mtime+interval+width+nudenet+classifier in `%LOCALAPPDATA%\SceneScout\cache`; `Re-analyze from cache` re-runs analysis without decoding.
- **Export selected / all** → `<stem>_scenes.txt` with:
  ```
  # duration=123.456
  0:12.345-0:20.000
  ```
- **Multi-video** — إضافة ملفات/مجلدات/Drugging-and-Drop (via FILES path).
- **"Show frame"** — extracts high-res annotated frame (Pillow if available).
- **Subtitle hints** — keyword-based SRT matching (sex,nude,etc.).
- **"Locked" rows preserved on re-scan** (`merge_with_locked`).

## 📦 Prerequisites (Python 3.10+)

Required:

- **Python 3.10+** (tested on 3.13)
- **FFmpeg + FFprobe** on PATH, or set env `FFMPEG` / `FFPROBE`.
- `FreeSimpleGUI`
- `NudeNet==3.4.2` (ships 320n.onnx)
- `onnxruntime` (GPU build preferred: `onnxruntime-gpu`)

Optional (for whole-frame classifier):

- `torch` (CUDA build recommended)
- `torchvision`
- `Pillow`
- `transformers` (TostAI backend)
- `moderators` (viddexa backend)
- `timm` (Marqo backend)

Pillow is also optional for annotated "Show frame" previews.

> Everything runs locally. The only network access is the one-time model download performed by model libraries (Hugging Face Hub) when a model is first loaded.

## 🚀 Quick start (Windows / NVIDIA)

1. Clone:

   ```bash
   git clone https://example.com/SceneScout.git
   cd SceneScout
   ```

2. Create venv:

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install base:

   ```bat
   pip install FreeSimpleGUI nudenet==3.4.2 onnxruntime-gpu
   ```

4. (Optional) Install classifier stack:

   ```bat
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install pillow transformers moderators timm
   ```

5. FFmpeg: download the Windows build from https://ffmpeg.org/download.html, add `ffmpeg.exe` and `ffprobe.exe` to PATH (or set `FFMPEG` / `FFPROBE`).

6. Verify & run:
   ```bat
   python -m py_compile SceneScout.py
   python SceneScout.py
   ```

## 🧪 How it works (data flow)

```
videos ──► probe_duration ──► extract_chunk (ffmpeg, 1s) ──► frames
         │                      ├── NudeNet.detect ───► detections
         │                      └── FrameClassifier.predict ───► explicit/suggestive
         │
         ├── scene_boundaries (optional) ──► snap
         │
         └── analyze_raw_to_candidates ──► hit_frames ──► build_candidates ──► merged candidates
                                                          ├── filter (min_hits / HIGH_SCORE)
                                                          ├── + subtitle hints
                                                          └── merge_with_locked (preserve manual)
                                  ──► GUI table ──► edit / export (.txt)
                                  ──► save_cache ( sha1(path|size|mtime|...) )
```

For each chunk, extract frames at a configurable interval; run NudeNet per frame and optionally the batched classifier on CUDA; cluster hits by `merge_gap`; expand each cluster by `before`/`after` seconds; filter; merge overlapping/subtitle candidates; snap to scene cuts. Cached raw frames enable pure-CPU re-analysis with different parameters (`Re-analyze from cache`).

## 🗂️ Output format

Each exported candidate row becomes a range line with timestamps (hundredth-second precision), plus a header with duration.

```
# duration=1234.567
0:12.345-0:20.000
1:05.000-1:18.250
```

## ⚙️ Key tunables (Review rules)

| Key                 | Default   | Meaning                                |
| ------------------- | --------- | -------------------------------------- |
| Interval (s)        | 1.0       | Frame sampling interval                |
| Frame width         | 960       | Rescale width (px)                     |
| Box score threshold | 0.35      | min NudeNet detection score            |
| Min box             | 0.08      | box height fraction of frame           |
| Min hits            | 2         | hits per candidate cluster             |
| Merge gap           | 12 (s)    | gap beyond which clusters stop merging |
| Before / After      | 3 / 3 (s) | padding around a cluster               |
| Explicit threshold  | 0.60      | explicit class score                   |
| Suggestive          | 0.70      | suggestive class score                 |
| Refine hits @0.5s   | on        | second pass around hits                |
| Snap to scenes      | on        | snap boundaries to scene cuts          |

## 💾 Cache

- Location: `%LOCALAPPDATA%\SceneScout\cache` (or tempdir fallback).
- Filename: `<sha1>.json`, atomic write (`.tmp` + replace).
- Version: `CACHE_VERSION = 3`.
- Key material includes: absolute path, file size, mtime_ns, interval, frame width, nudenet model key, classifier model id.

## 🧩 Architecture

Single-file app (`SceneScout.py` ~2.4 kLoC). Noteworthy sections:

- `AppState` — global runtime state.
- Detectors: `ensure_nudenet`, `FrameClassifier` (transformers / moderators / timm).
- Pipeline: `extract_chunk`, `hit_frames_from_raw`, `select_hit_frames`, `build_candidates_from_hits`, `analyze_raw_to_candidates`, `merge_with_locked`, `snap_ranges`.
- GUI: `FreeSimpleGUI` (Dark base theme) with tables, sliders, threads for scan/re-analyze.
- Cache: `cache_path / build_cache_key / load_cache / save_cache`.

## 🔒 Privacy

- All processing is local.
- Network is used only by model libraries (Hugging Face Hub) to download weights on first model load; cached models stay on disk.
- Video data never leaves the machine.
- Exported `.txt` files are written next to source videos (configurable by editing `export_one`).

## 🐧 Run on other platforms

The code uses `os.startfile` only on Windows; on macOS/Linux it prefers `open`/`xdg-open`. FFmpeg detection works via `shutil.which`. CUDA-specific paths (CUDA decode, CUDAExecutionProvider) gracefully fall back to CPU. Primarily developed/tested on Windows.

## 🤝 License

Unspecified — see repository owner. External components have their own licenses (e.g. `CLASSIFIER_MODELS` entries document classifier licenses: CC BY-NC-SA 4.0 / Apache-2.0).

## 📝 Status

Early/prototype: single-file script, no tests, no CI. Experimental thresholds and detection semantics subject to change.

## ⚠️ Notes

- "Candidates" are heuristic suggestions — not ground truth. Always review in the table.
- Large videos: first scan decodes frames; subsequent runs can use `Re-analyze from cache`.
- Cancelling (Stop) sets a threading.Event; the worker finishes the current frame batch first.
- Only simple `.srt` subtitle format is parsed.
