# Vendored code

Third-party source kept verbatim, so its dependency pins never reach the AI environment's resolver.
Nothing here imports from the rest of `amverge`, and only the module named below imports from here.

## depth_anything_v2

- **Source:** the `depth-anything-v2` 0.1.0 wheel on PyPI (Depth Anything V2's model code: the DPT head and DINOv2 backbone).
- **License:** Apache-2.0, see `depth_anything_v2/LICENSE`.
- **Changes:** none. Every file matches the wheel's `RECORD` hashes. It has no `__init__.py`, as published, and imports as a namespace package.
- **Why vendored:** the wheel pins `torchvision<0.23` (PyTorch 2.7) and pulls `gradio-imageslider`, `matplotlib` and full `opencv-python`, none of which the code imports. Its only outside imports are `torch`, `numpy`, `cv2` and `torchvision.transforms.Compose`. The pin made the depth pack impossible to install beside GPU decode (PyTorch 2.13/2.14).
- **Used by:** `amverge/core/depth/depth_map.py` only.
- **Weights:** downloaded separately (`core/depth/registry.py`). vits is Apache-2.0; vitb and vitl are CC-BY-NC-4.0.
