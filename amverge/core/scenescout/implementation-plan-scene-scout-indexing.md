# Scene Scout: indexing.py implementation plan

Status: plan only. Not implemented yet.
Target file: `amverge/core/scenescout/indexing.py`

## Goal

Turn a video into rows in a Scene Scout database using AMVerge's OWN scene detectors (not PySceneDetect), sample one representative frame per scene, and let the existing `index_video` orchestrator do the rest. This is the other half of the scene-scout feature.

Reference source (upstream scene-scout repo):
- `src/processing.py` (fast_process_and_embed, accurate_process_and_embed)
- detection dispatch must use AMVerge's own modules (see below)

Related plan file (same feature, other half): `implementation-plan-scene-scout-embedding.md`.

## Locked-in decisions

- `--accurate` on `scout add` forces the TransNetV2 path (nelux/ffmpeg decode + TransNetV2 inference). Falls back to keyframe when the `[ml]` extra is not installed.
- Both documented Settings detectors must dispatch to AMVerge's own detectors:
  - `transnetv2_gpu` -> `amverge.core.detection.ai_scene_detection.decode_and_detect_scenes`
  - `keyframe_detection` -> `amverge.core.detection.keyframe.detect_scenes_by_keyframe`
- `scout add` is invoked by the app as `amverge scout add <video> --db <name> --root <dir> --detector <settings-choice> --json`;
  progress JSON lines (`{"stage":..., "done":..., "total":...}`) go to stdout line by line and are streamed to the UI as `scout_progress` events.
- Scene boundaries returned as `(start_ms, end_ms)` milliseconds, because that is what the DB columns and the upstream schema use.

## Data flow (already written, unchanged)

```
detect_scenes(video, method)   -> [(start_ms, end_ms), ...]        N scenes
sample_frames(video, scenes)   -> [uint8 (H, W, 3) RGB, ...]       N frames
embed_frames(frames)           -> float32 (N, D), rows normalised  N vectors
encode_thumbnail(frame)        -> JPEG bytes                       N thumbs
                                  |
                                  v
                          one scene_embeddings row per scene
```

`index_video` (already implemented) zips `scenes`, `frames`, and `vectors` together in order. A dropped frame silently shifts every scene after it onto the wrong timestamp, with nothing to catch it. Length and order are sacred; never return fewer frames than scenes.

The ONLY two functions that need writing are `detect_scenes` and `sample_frames`.

### CURRENT BUG / BEHAVIOUR to preserve, not work around
- `DETECTOR_AI = "transnetv2_gpu"`, `DETECTOR_KEYFRAME = "keyframe_detection"` constants already exist. Keep the exact spellings; the app passes them through verbatim from Settings.
- If `detect_scenes` returns `[]`, `index_video` returns `0` scenes and writes NO video row (returns before `upsert_video`). That is acceptable but the plan prefers a keyframe fallback for AI-detected-empty so users never get a silent "nothing indexed".

## Implementation

### `detect_scenes(video, *, method=DETECTOR_KEYFRAME, accurate=False)`

```
transnetv2   <= (method == "transnetv2_gpu") OR accurate
   |- import amverge.core.detection.ai_scene_detection  (lazily)
   |    TRANSNET_AVAILABLE, decode_and_detect_scenes
   |- if TRANSNET_AVAILABLE:
   |      scenes_secs, _ = decode_and_detect_scenes(video_path)   # (N, 2) seconds, ndarray
   |      scenes = [(int(s * 1000), int(e * 1000)) for s, e in scenes_secs]
   |      if scenes: return scenes          # else fall through to keyframe
   |      (ml pack missing or 0 scenes)
   v
keyframe
   |- import amverge.core.detection.keyframe.detect_scenes_by_keyframe (lazily)
   |- scenes = detect_scenes_by_keyframe(video_path)               # list of [start, end] seconds
   |- return [(int(s * 1000), int(e * 1000)) for s, e in scenes]
```

Details:
- Unknown `method` value -> keyframe (documented behaviour; indexing must keep working).
- `accurate=True` with an unrecognised `method` -> transnetv2 (user decision), same fallback as above.
- `decode_and_detect_scenes` emits `PROGRESS|pct|msg` to stderr via `emit_progress`. Harmless for `--json` (Rust reads stdout only). Optional polish: temporarily monkeypatch `emit_progress` to route into `on_progress`, mirroring `pipeline.py`'s pattern, to keep stderr clean on plain-terminal runs.
- Both detectors work in seconds; this function returns milliseconds. Always round with `int(...)` (truncation toward zero), matching `start_ms`/`end_ms` upstream semantics.

### `sample_frames(video, scenes)` -> list of RGB uint8 numpy frames

Port the decode pattern from `processing.accurate_process_and_embed`, but pick the MIDDLE frame of each scene (docstring mandate; the first frame of a scene often lands in the fade-out of the previous one).

- Validate: if not `scenes`, return `[]`.
- Open: `av.open(str(video), options={"err_detect": "ignore_err"})`; `stream = container.streams.video[0]`, `stream.thread_type = "AUTO"`.
- Targets: `mid_ms = [(s + e) // 2 for s, e in scenes]` (chronological because scenes are ordered).
- Single forward pass: per frame, `ms = int(frame.time * 1000)`; while `ms >= next_mid`, capture THIS frame via `frame.to_ndarray(format="rgb24")`, append, advance pointer. Wrap each frame in `try/except (av.AVError, ValueError): continue` to skip corrupt packets (upstream pattern).
- Scenes whose midpoint lands past the last frame (duration rounding, tiny trailing scene): backfill with the last decoded frame so `len(frames) == len(scenes)` and order holds.
- `finally: container.close()`.
- Frames are RGB by construction (`format="rgb24"`), numpy-uint8, full resolution; the SigLIP processor handles downscaling to its patch budget, so no pre-resize.

### `index_video` edit (single line of wiring)

Pass live embedding progress through the existing `report()` hook:

```python
vectors = embed_frames(
    frames, device=device, max_patches=max_patches, batch_size=batch_size,
    on_progress=lambda d, t: report("embedding", d, t),
)
```

right after the existing `report("embedding", 0, len(frames))`. Nothing else in the orchestrator changes: upsert-video-as-indexing, insert_scenes, mark_video_complete, thumbnail generation, and the final `report("done", ...)` all stay as-is.

## Notes

- `encode_thumbnail(frame)` is already correct (JPEG, max width 320, quality 80) - do not change.
- Do not add OpenCV to this file; `sample_frames` uses PyAV (`av`), which is already a base dependency. If cv2 is ever introduced, remember it returns BGR while embedding expects RGB.
- The `index_video` `--json` path calls `report()` per stage; keep `report("detecting"...)/("sampling"...)/("embedding"...)/("done"...)` frequencies such that a long video does not look frozen (this is why the embedding progress hook matters).

## Gotchas

1. **Frames must be RGB.** PyAV `to_ndarray(format="rgb24")` gives RGB; cv2 gives BGR. Embedding BGR yields vectors and rankings, just wrong colours. Never hand cv2 frames to the embed step.
2. **Embeddings must be L2-normalised** and `embed_text`/`embed_frames` must agree on `D` - see the embedding plan; `search.py` silently skips shape mismatches.
3. **Length/order fidelity**: `sample_frames` must return EXACTLY `len(scenes)` frames in the same order. One dropped frame shifts every later scene onto the wrong timestamp. The mid-target scan + last-frame backfill exists precisely to guarantee this.
4. **Seconds vs milliseconds**: AMVerge detectors return seconds; `scout` DB columns and upstream schema use milliseconds. Convert with `int(sec * 1000)`.
5. **transnetv2 import guard**: importing `amverge.core.detection.ai_scene_detection` is safe without the `[ml]` extra (its `transnetv2_pytorch` import is guarded at module level by try/except and sets `TRANSNET_AVAILABLE`). Branch on that flag; never call `decode_and_detect_scenes` when it is False (it raises ImportError).
6. **Do not import torch at module level** in this file (or anywhere in `core/scenescout`); `paths`, `types`, `db`, and `search` stay pure so `--help` and status work with no AI extra.
7. **Empty result**: if TransNetV2 yields zero scenes, fall back to keyframe so `scout add` still writes rows instead of silently indexing nothing.
8. **`--accurate` semantics**: it FORCES transnetv2 (user decision). It is NOT "PySceneDetect accurate mode" - that is deliberately not ported.

## Testing

See `tests/test_scenescout_indexing.py` (to be written):
- `detect_scenes` dispatch with `monkeypatch`: stub `amverge.core.detection.ai_scene_detection.TRANSNET_AVAILABLE`/`decode_and_detect_scenes` and `amverge.core.detection.keyframe.detect_scenes_by_keyframe`; assert millisecond conversion, `accurate=True -> transnetv2`, and the ml-missing fallback to keyframe.
- `sample_frames` integration (no torch, no ffmpeg binary): write a tiny video with PyAV using the built-in `mpeg4` encoder, distinct solid-colour frames at a known fps, define 2-3 scenes by ms, assert the result is EXACTLY `len(scenes)` frames, ordered, RGB-shaped, and colours match the middle-frame timestamps. This catches off-by-one and drop-shift bugs, the exact failure mode the module warns about.

## Manual verification

```bash
amverge scout create "Test"
amverge scout add ./episode01.mkv --db "Test"            # watch stage/progress json lines on --json
amverge scout videos "Test"                              # shows the scene count
amverge scout search "whatever is in that episode" --db "Test"
```

The last search is the real end-to-end check; it needs no AMVerge build because the app runs exactly these commands.

Also verify:
- `--detector transnetv2_gpu` picks TransNetV2 and falls back to keyframe cleanly when the `ml` pack is absent.
- `--accurate` forces the TransNetV2 path.
- `amverge scout status` still reports `modelAvailable: false` gracefully with no torch installed.