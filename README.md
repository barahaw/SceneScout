# SceneScout

> **Local. Private. On-device (Windows + NVIDIA).** A scanner that mines a personal video library to find candidate "scene" ranges, using NudeNet plus an optional whole-frame NSFW classifier, with manual review and `.txt` export.

---

## ✨ Features

- **Automatic scan (Scan)** — sample frames every N seconds (default 1.0s); first pass + a refine pass at 0.5s around hits.
- **Dual detection:**
  - **NudeNet** (`320n`, `640n`, …) — body-part detection (genitalia, breasts, …) with score and min-box-height thresholds.
  - **Optional whole-frame classifier**: `TostAI/nsfw-image-detection-large`, `viddexa/nsfw-detection-2-mini`, `Marqo/nsfw-image-detection-384`.
- **Clustering with `merge_gap`, `before`/`after` padding, `min_hits` / `HIGH_SCORE` filtering, optional scene-snap to cuts.**
- **Manual editing** — apply edit (range + lock), add/delete/toggle candidates, add a manual row.
- **Caching** — SHA-1 cache keyed by `path|size|mtime|interval|width|nudenet|classifier` in `%LOCALAPPDATA%\SceneScout\cache`; "Re-analyze from cache" re-runs analysis without decoding.
- **Export selected / all** → `<stem>_scenes.txt`:
  ```
  # duration=123.456
  0:12.345-0:20.000
  ```
- **Multi-video** — add files/folders (paths separated by `;`).
- **Show frame** — extracts a high-res annotated frame via FFmpeg (Pillow for boxes if available).
- **Subtitle hints** — keyword-based SRT matching (default `sex,nude,naked,explicit`).
- **"Locked" rows preserved on re-scan** via `merge_with_locked`.

## 📦 Prerequisites (Python 3.10+)

Required:
- **Python 3.10+** (tested on 3.13)
- **FFmpeg + FFprobe** on PATH, or set env `FFMPEG` / `FFPROBE`.
- `FreeSimpleGUI`
- `NudeNet==3.4.2` (ships `320n.onnx`)
- `onnxruntime` (GPU build preferred: `onnxruntime-gpu`)

Optional (for the whole-frame classifier):
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
   git clone https://github.com/barahaw/SceneScout.git
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
videos ──► probe_duration ──► extract_chunk (ffmpeg) ──► frames
         │                     ├── NudeNet.detect ───► detections
         │                     └── FrameClassifier.predict ──► explicit/suggestive
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

Frames are sampled with a configurable interval; NudeNet runs per frame and the batched classifier runs on CUDA when available; hits are clustered by `merge_gap`; each cluster is padded by `before`/`after` seconds; candidates are filtered/merged/snapped. Cached raw frames enable pure-CPU re-analysis with different parameters ("Re-analyze from cache").

## 🗂️ Output format

Each exported candidate row becomes a range line with timestamps (hundredth-second precision), plus a header with duration.

```
# duration=1234.567
0:12.345-0:20.000
1:05.000-1:18.250
```

## ⚙️ Key tunables (Review rules)

| Key | Default | Meaning |
|-----|---------|---------|
| Interval (s) | 1.0 | Frame sampling interval |
| Frame width | 960 | Rescale width (px) |
| Box score threshold | 0.35 | min NudeNet detection score |
| Min box | 0.08 | box height fraction of frame |
| Min hits | 2 | hits per candidate cluster |
| Merge gap | 12 (s) | gap beyond which clusters stop merging |
| Before / After | 3 / 3 (s) | padding around a cluster |
| Explicit threshold | 0.60 | explicit class score |
| Suggestive | 0.70 | suggestive class score |
| Refine hits @0.5s | on | second pass around hits |
| Snap to scenes | on | snap boundaries to scene cuts |

## 💾 Cache

- Location: `%LOCALAPPDATA%\SceneScout\cache` (or tempdir fallback).
- Filename: `<sha1>.json`, atomic write (`.tmp` + replace).
- Version: `CACHE_VERSION = 3`.
- Key material includes: absolute path, file size, mtime_ns, interval, frame width, nudenet model key, classifier model id.

## 🧩 Architecture

Single-file app (`SceneScout.py`, ~2.4 kLoC). Noteworthy sections:

- `AppState` — global runtime state.
- Detectors: `ensure_nudenet`, `FrameClassifier` (transformers / moderators / timm).
- Pipeline: `extract_chunk`, `hit_frames_from_raw`, `select_hit_frames`, `build_candidates_from_hits`, `analyze_raw_to_candidates`, `merge_with_locked`, `snap_ranges`.
- GUI: `FreeSimpleGUI` (dark base theme) with tables, sliders, threaded scan/re-analyze workers.
- Cache: `cache_path` / `build_cache_key` / `load_cache` / `save_cache`.

## 🔒 Privacy

- All processing is local.
- Network is used only by model libraries (Hugging Face Hub) to download weights on first model load; cached models stay on disk.
- Video data never leaves the machine.
- Exported `.txt` files are written next to source videos (configurable by editing `export_one`).

## 🐧 Other platforms

The code uses `os.startfile` only on Windows; on macOS/Linux it prefers `open`/`xdg-open`. FFmpeg detection works via `shutil.which`. CUDA-specific paths (CUDA decode, CUDAExecutionProvider) gracefully fall back to CPU. Primarily developed/tested on Windows.

## 🤝 Licenses

Unspecified upstream — see repository owner. External components carry their own licenses (classifier entries document: TostAI CC BY-NC-SA 4.0; viddexa & Marqo Apache-2.0).

## 📝 Status

Early/prototype: single-file script, no tests, no CI. Experimental thresholds and detection semantics are subject to change.

## ⚠️ Notes

- Candidates are heuristic suggestions — not ground truth. Always review in the table.
- Large videos: first scan decodes frames; subsequent runs can use "Re-analyze from cache".
- Cancelling (Stop) sets a threading event; the worker finishes the current frame batch first.
- Only a simple `.srt` subtitle format is parsed.
