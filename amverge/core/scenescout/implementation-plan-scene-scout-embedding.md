# Scene Scout: embedding.py implementation plan

Status: plan only. Not implemented yet.
Target file: `amverge/core/scenescout/embedding.py`

## Goal

Port the SigLIP 2 model loading and embedding logic from the standalone scene-scout app into AMVerge-CLI so that `amverge scout search` and `amverge scout add` work end to end. This file turns text, reference images, and video frames into L2-normalised float32 vectors that `search.py` scores with a plain dot product (read as cosine similarity).

Reference source (upstream scene-scout repo):
- `src/model_loader.py` (load_siglip_model, get_compute_device)
- `src/processing.py` (_encode_images, get_query_embedding)
- `src/config.py` (DEFAULT_MODEL, ATTENTION_IMPL, get_hf_token)
- `src/utils.py` (normalize_embedding)

Related plan file (same feature, other half): `implementation-plan-scene-scout-indexing.md`.

## Locked-in decisions

- Model id: `google/siglip2-so400m-patch16-naflex`. Update `EMBEDDING_MODEL_VERSION` in `types.py` from the current org-less `siglip2-so400m-patch16-naflex` to the org-prefixed id. It is used BOTH as the HF repo id (AutoProcessor/AutoModel `from_pretrained`) AND as the DB `model_version` marker. Upstream stores the org-prefixed id in its DB (`config.DEFAULT_MODEL`), so this change also keeps databases compatible with the standalone tool.
- Device support: cuda / mps / cpu only. No DML, XPU, or TensorRT. No new dependencies in the `[scout]` extra: `torch`, `transformers`, `pillow`, `numpy` are already there.
- All `torch` / `transformers` imports stay lazy (inside functions). `paths`, `types`, `db`, `search` and `--help` must keep working with no AI extra installed.

## Public API that file must keep (callers depend on these signatures)

From `search.py`:

```python
embed_text(query, *, device=None, max_patches=256)   -> np.ndarray (D,) float32 L2
embed_image(image_path, *, device=None, max_patches=256) -> np.ndarray (D,) float32 L2
```

From `indexing.index_video`:

```python
embed_frames(frames, *, device=None, max_patches=256, batch_size=16) -> np.ndarray (N, D) float32 L2
```

Also kept: `is_available()`, `resolve_device(preferred=None)`, `model_version()`, `pack_embedding(vector)`, `unpack_embedding(blob)`, `SceneScoutModelUnavailable`, module-level `_MODEL`/`_PROCESSOR`/`_DEVICE` cache globals.

## Implementation

### `_load(device=None, max_patches=256)` -> (model, processor, device)

Port of `model_loader.load_siglip_model`, trimmed to the supported backends.

- Import `torch`, then `AutoProcessor`, `AutoModel` from `transformers` inside the function.
- Resolve device string via the existing `resolve_device(device)` (raises `SceneScoutModelUnavailable` when torch is missing).
- If `_MODEL` is cached and the requested device matches `_DEVICE`, return the cache. Otherwise load fresh and overwrite the globals.
- dtype selection:
  - `cuda`: read `torch.cuda.get_device_capability()`; FP16 when major >= 7 (Volta+), FP32 for older GPUs (Pascal and older do FP16 poorly or not at all).
  - `mps`, `cpu`: FP32.
- Attention implementation: `"sdpa" if hasattr(torch.nn.functional, "scaled_dot_product_attention") else "eager"` (mirrors upstream `ATTENTION_IMPL`).
- HF token: `os.environ.get("HF_TOKEN") or None` (analogue of upstream `config.get_hf_token()`; needed for any gated checkpoint).
- Load processor: `AutoProcessor.from_pretrained(EMBEDDING_MODEL_VERSION, token=token)`.
- Load model: `AutoModel.from_pretrained(EMBEDDING_MODEL_VERSION, token=token, torch_dtype=dtype, attn_implementation=attn_impl, device_map={"": device_str})`.
  - `torch_dtype` (not `dtype`, that key is finicky across transformers versions).
  - `device_map={"": device_str}` for cuda/mps/cpu, exactly as upstream does.
- After load: `model.requires_grad_(False)`, `model.eval()`.
- CPU thread pinning: when the device is `cpu`, cap `torch.set_num_threads` to the physical core count. Guard the whole block with try/except and fall back to `os.cpu_count() // 2` heuristics. This is the difference between a usable CPU index run and a slow one on machines with high logical-core counts.
- Cache and return `(model, processor, device_str)`.

### `_encode_images(images, model, processor, device, max_num_patches)` -> torch.Tensor (B, D) pre-normalisation

Port of `processing._encode_images`. Used by `embed_image` and `embed_frames`.

- `kwargs = {"images": images, "return_tensors": "pt"}`; add `max_num_patches` when not None.
- `inputs = processor(**kwargs).to(device)`.
- If `"pixel_values"` is present and floating point, cast to `model.dtype` (prevents the FP32-vs-FP16 mismatch crash/fp16-blob issue).
- `with torch.no_grad(): out = model.get_image_features(**inputs)`.
- Extract the tensor: `out if isinstance(out, torch.Tensor) else getattr(out, "pooler_output", None) or out[0]`.
- Normalise: `torch.nn.functional.normalize(features, p=2, dim=1)`. (Upstream `utils.normalize_embedding` does exactly this.)

### `embed_text(query, *, device=None, max_patches=256)` -> np.ndarray (D,)

- Load (or reuse) the model via `_load(device)`.
- Lowercase the query, exactly like upstream `get_query_embedding`: `processor(text=[query.lower()], return_tensors="pt", padding="max_length", max_length=64).to(device)`.
  - `max_length=64` matches the SigLIP 2 tokenizer contract upstream relies on. Do not drop it.
- `model.get_text_features(**inputs)` under `torch.no_grad()`; extract tensor, normalise, `.cpu().numpy().astype(np.float32)`.
- Return the single row (squeeze) so shape is `(D,)`.

### `embed_image(image_path, *, device=None, max_patches=256)` -> np.ndarray (D,)

- `Image.open(image_path).convert("RGB")`; wrap in a one-element list; `_encode_images`.
- Return row 0, squeezed to `(D,)`.

### `embed_frames(frames, *, device=None, max_patches=256, batch_size=16, on_progress=None)` -> np.ndarray (N, D)

- Frames are `np.uint8` of shape `(H, W, 3)`, RGB, from `indexing.sample_frames` (PyAV `format="rgb24"`). Convert each to a PIL Image via `Image.fromarray(f)` before batching - PIL guarantees RGB and makes the BGR trap structurally impossible.
- Batch loop `range(0, N, batch_size)`, `_encode_images` per batch, collect normalised rows.
- When `on_progress` is provided, call `on_progress(done, total)` after each batch so the app's progress bar stays alive during the long embed stage (see `implementation-plan-scene-scout-indexing.md` for the caller edit).
- On `cpu`, `gc.collect()` after each batch (upstream does this for RAM pressure during long builds).
- Return `np.vstack(rows)` as float32 `(N, D)`. Return `np.empty((0, D), dtype=np.float32)` for an empty input so the shape contract never breaks.

## Notes

- `is_available()` must stay a cheap probe: try/except import of `torch` and `transformers`, return bool, never load the model.
- `resolve_device()` already covers cuda > mps > cpu; leave it, honouring an explicit `device` only when usable.
- `pack_embedding`/`unpack_embedding` are already correct (raw float32 bytes) - do not change the serialisation; upstream databases are raw float32.
- `model_version()` returns `EMBEDDING_MODEL_VERSION` (after the `types.py` value fix).

## Gotchas (must not reintroduce)

1. **BGR instead of RGB.** OpenCV hands back BGR, PyAV and PIL hand back RGB. Embedding BGR frames still yields vectors and still ranks them, just against the wrong colours. Never pass raw cv2 frames here; always funnel through PIL `convert("RGB")` or PyAV `format="rgb24"`.
2. **Skipping normalisation.** `search.py` scores with a plain dot product and treats the result as cosine similarity, which is only true for unit vectors. Unnormalised vectors silently produce nonsense rankings rather than an error. Every path (text, image, frames) must go through `F.normalize(p=2, dim=1)`.
3. **`embed_text` and `embed_frames` disagreeing on D.** `search.py` skips rows whose shape differs from the query (`vector.shape != query.shape`), so a mismatch makes every search return nothing, silently. All paths come from the same model (D=1152) - keep `max_length=64` text handling and the naflex processor settings identical to upstream.
4. **Module-level `import torch`.** A module-level import makes the whole CLI unusable without the AI extra (even `--help`). All imports stay inside functions.
5. **Model repo id without org prefix** does not resolve. `EMBEDDING_MODEL_VERSION` must be `google/siglip2-so400m-patch16-naflex`.
6. **Model repr/param strings**: `model.requires_grad_(False)` after load, not before, and never rely on parameters being frozen at save time.
7. **FP16 on old GPUs**: always probe `get_device_capability`; hardcoding fp16 for every CUDA GPU breaks Pascal (6.x) and older.
8. **Transformer dtype**: pass `torch_dtype=dtype`; the bare `dtype=` kwarg has been renamed across transformers versions.

## Testing

See `tests/test_scenescout_embedding.py` (to be written):
- Pure unit, no torch: `pack_embedding`/`unpack_embedding` round-trip; L2-normalisation invariant against a tiny fake model object.
- Heavy integration (guarded by `pytest.importorskip("torch")`): `embed_text(...).shape == embed_frames([frame], ...).shape`, dtype float32, unit norm.
- Must skip cleanly on machines without the `[scout]` extra - the repo already uses that pattern in `test_deadframes.py`.

## Manual verification

```bash
pip install -e ".[scout]"
python -c "from amverge.core.scenescout import embedding; v=embedding.embed_text('a girl standing in the rain'); print(v.shape, v.dtype, round(float((v**2).sum()), 3))"
# expect: (1152,) float32 ~1.0
amverge scout search "sunset over water" --db "Test"    # real results after indexing works
```

Also confirm `amverge scout status` and `python -c "import amverge.cli"` still work with no torch installed.