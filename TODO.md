# TODO

Goal: users should never need to import external libraries directly.
Everything exposed through `from amverge import X` with clean classes/functions.

---

## 1. Video Reading / Encoding ✅

~~`AmvergeVideo` class~~ Done. `core/amverge_video.py`, exported from `amverge`.

```python
from amverge import AmvergeVideo
video = AmvergeVideo("episode.mp4")
video.duration, video.fps, video.width, video.height, video.codec, video.is_hevc
video.keyframes, video.audio_streams
video.cut_scene(0, 5, "clip.mp4")
video.copy_segment(0, 5, "clip.mp4")
video.extract_segment(0, 5, "encoded.mp4")
video.extract_thumbnail(5, "thumb.jpg")
video.extract_thumbnails_at([0, 5], "dir/")
video.decode_frames()
video.detect_scenes("keyframe")
video.to_dict()
```

---

## 2. Scene Detection ✅

~~`SceneDetector` class~~ Done. `core/scene_detector.py`, exported from `amverge`.

```python
from amverge import SceneDetector

detector = SceneDetector(method="transnetv2", min_duration=1.0)
result = detector.detect("episode.mp4")
print(f"{len(result.scenes)} scenes in {result.detection_time}s")

# Post-processing
result.filter(min_duration=2.0)
result.merge_similar()
result.to_json("cleaned_scenes.json")
```

Wraps: `detect_scenes()`, `detect_cuts_by_keyframe()`, `detect_cuts_by_edge()`, V2 pipeline.
Also added `filter()`, `merge_similar()`, `to_json()` methods to `DetectResult`.

---

## 3. Export / Encoding ✅

~~`SceneExporter` class~~ Done. `core/scene_exporter.py`, exported from `amverge`.

```python
from amverge import SceneExporter

exporter = SceneExporter(codec="h264_main", audio="aac_320", container="mp4", hardware="auto")
exporter.export(scenes, output_dir="./export")
exporter.export_one(scene, "output.mp4")
exporter.merge(scenes, "merged.mp4")
```

Wraps: manual ffmpeg concat, `CODEC_PROFILES`, `AUDIO_FFMPEG`, `resolve_gpu()`.

---

## 4. Similarity / Comparison ✅

~~`SimilarityChecker` class~~ Done. `core/similarity_checker.py`, exported from `amverge`.

```python
from amverge import SimilarityChecker

checker = SimilarityChecker(threshold=0.10)
similar = checker.are_similar("thumb_a.jpg", "thumb_b.jpg")
pairs = checker.find_in(scenes)
```

Wraps: `check_pair_similar()`, `find_similar_pairs()`.

---

## 5. Image Operations ✅

~~`ImageCrop` class~~ Done. `core/image_crop.py`, extends `CropData` with `apply()`. Exported as `ImageCrop` and `CropData`.
---

## 6. Diagnostics ✅

~~`check_environment()`~~ Done. `core/diagnostics.py`. Returns structured `EnvironmentCheck` with `CheckResult` dataclasses. `amverge doctor` now delegates to this.
---

## 7. Thumbnails ✅

~~`ThumbnailGenerator` class~~ Done. `core/thumbnail_generator.py`, exported from `amverge`.

```python
from amverge import ThumbnailGenerator

gen = ThumbnailGenerator(workers=4)
gen.generate(scenes, output_dir="./thumbs")
gen.generate_one("clip.mp4", "thumb.jpg")
```

Wraps: `make_thumbnail()`, `generate_thumbnails()`.

---

## 8. Cache Management ✅

~~`SceneCache` class~~ Done. `core/scene_cache.py`, exported from `amverge`.

```python
from amverge import SceneCache

cache = SceneCache("./cache_dir")
cache.list(); cache.clear("episode.mp4"); cache.clear_all()
cache.exists("episode.mp4")
cache.load("episode.mp4"); cache.save("episode.mp4", scenes_secs, scenes_frames)
```

Wraps: `build_video_cache_prefix()`, manual .npy file I/O.

---

## 9. TransNetV2 ✅

~~`TransNetConfig` class~~ Done. `core/transnet_config.py`, frozen dataclass exported from `amverge`.
---

## 10. Clean Up Examples ✅

~~`full_pipeline.py` uses `AmvergeVideo`, `SceneCache`, `ThumbnailGenerator`, `SimilarityChecker`. No more raw `torch`/`numpy` imports.~~

---

## Priority Order

| # | Item | Status |
|---|------|--------|
| 1 | `AmvergeVideo` | ✅ done |
| 2 | `SceneDetector` | ✅ done |
| 3 | `SceneExporter` | ✅ done |
| 4 | `SceneCache` | ✅ done |
| 5 | `ThumbnailGenerator` | ✅ done |
| 6 | `SimilarityChecker` | ✅ done |
| 7 | `ImageCrop` rename | ✅ done |
| 8 | `check_environment()` | ✅ done |
| 9 | `TransNetConfig` | ✅ done |
| 10 | Clean examples | ✅ done |
