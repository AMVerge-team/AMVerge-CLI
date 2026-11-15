from __future__ import annotations

import os
from typing import Callable, Optional

from ..infra.config import get_amverge_config_dir
from ..infra.model_store import download_with_resume, verify_sha256
from .registry import INTERPOLATION_REGISTRY, get_all_model_keys, get_model


def _get_weights_dir():
    return os.path.join(get_amverge_config_dir(), "models", "interpolation")


def get_weight_path(model_key):
    entry = get_model(model_key)
    if entry is None or "file" not in entry:
        raise ValueError(f"Unknown model: {model_key}")
    return os.path.join(_get_weights_dir(), entry["file"])


def is_weight_downloaded(model_key):
    path = get_weight_path(model_key)
    return os.path.exists(path) and os.path.getsize(path) > 0


def verify_weight_hash(model_key, path=None):
    entry = get_model(model_key)
    expected = entry.get("hash") if entry else None
    return verify_sha256(path or get_weight_path(model_key), expected)


def download_weights(model_key, progress_cb=None, retries=3):
    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model: {model_key}")

    dest = get_weight_path(model_key)
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and verify_weight_hash(model_key, dest):
        return True

    label = entry.get("name", model_key)
    try:
        return download_with_resume(
            entry["url"],
            dest,
            expected_hash=entry.get("hash"),
            retries=retries,
            progress_cb=progress_cb,
            progress_formatter=lambda pct, _dl, _tot: f"Downloading {label}... {pct}%",
            done_message=f"Downloaded {label}",
        )
    except Exception as e:
        raise RuntimeError(f"Download failed for {model_key}: {e}") from e


def _extract_rife_state_dict(checkpoint):
    """Unwrap common checkpoint containers and normalize keys to bare
    ``block0.*`` / ``encode.*`` form (strip ``module.`` and ``flownet.``)."""
    from collections import OrderedDict

    sd = checkpoint
    if isinstance(checkpoint, dict):
        for k in checkpoint:
            if isinstance(k, str) and "model_state_dict" in k and k.startswith("ema_"):
                sd = checkpoint[k]
                break
    if not isinstance(sd, dict):
        return OrderedDict()

    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("flownet."):
            nk = nk[len("flownet."):]
        out[nk] = v
    return out


def _get_flow_weights_path(flow_file):
    return os.path.join(_get_weights_dir(), "flow", flow_file)


def _is_flow_weight_downloaded(flow_file):
    path = _get_flow_weights_path(flow_file)
    return os.path.exists(path) and os.path.getsize(path) > 0


def _download_flow_weights(flow_file, progress_cb=None, retries=3):
    url = f"https://github.com/AniScripts/AniSmooth-Models/releases/download/interpolation/{flow_file}"
    dest = _get_flow_weights_path(flow_file)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    try:
        return download_with_resume(
            url,
            dest,
            retries=retries,
            timeout=300,
            progress_cb=progress_cb,
            progress_formatter=lambda pct, _dl, _tot: f"Downloading GMFlow weights... {pct}%",
            done_message="Downloaded GMFlow weights",
        )
    except Exception as e:
        raise RuntimeError(f"Download failed for GMFlow weights: {e}") from e


def _load_pervfi_model(model_key, device="cpu"):
    import torch
    from .pervfi_arch import Pipeline_infer, build_generator_arch
    from .pervfi_gmflow import GMFlow

    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model: {model_key}")

    if not is_weight_downloaded(model_key):
        raise FileNotFoundError(
            f"Generator weights not found for {model_key}. Run: amverge interpolate --model {model_key}"
        )

    gen_path = get_weight_path(model_key)
    flow_file = entry.get("flow_file")
    generator = entry.get("generator", "v00")

    if flow_file and not _is_flow_weight_downloaded(flow_file):
        _download_flow_weights(flow_file)

    pipeline = Pipeline_infer(generator, gen_path, device=device)
    pipeline.eval()

    if flow_file:
        flow_path = _get_flow_weights_path(flow_file)
        gmflow = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        )
        flow_ckpt = torch.load(flow_path, map_location=device, weights_only=True)
        gmflow.load_state_dict(flow_ckpt["model"])
        gmflow.to(device).eval()

        class InputPadder:
            def __init__(self, size, divide=16, mode="center"):
                self.ht, self.wd = size[-2:]
                pad_ht = (((self.ht // divide) + 1) * divide - self.ht) % divide
                pad_wd = (((self.wd // divide) + 1) * divide - self.wd) % divide
                if mode == "center":
                    self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
                else:
                    self._pad = [0, pad_wd, 0, pad_ht]

            def pad(self, *inputs):
                return [torch.nn.functional.pad(x, self._pad, mode="constant") for x in inputs]

            def unpad(self, *inputs):
                return [x[..., self._pad[2]:self.ht + self._pad[2], self._pad[0]:self.wd + self._pad[0]] for x in inputs]

        @torch.no_grad()
        def compute_flow(img0, img1):
            img0 = img0.to(device) * 255.0
            img1 = img1.to(device) * 255.0
            padder = InputPadder(img0.shape, 16)
            img0_p, img1_p = padder.pad(img0, img1)
            results = gmflow(img0_p, img1_p,
                             attn_splits_list=[2],
                             corr_radius_list=[-1],
                             prop_radius_list=[-1],
                             pred_bidir_flow=False)
            fflow = results["flow_preds"][-1]
            results = gmflow(img1_p, img0_p,
                             attn_splits_list=[2],
                             corr_radius_list=[-1],
                             prop_radius_list=[-1],
                             pred_bidir_flow=False)
            bflow = results["flow_preds"][-1]
            fflow, bflow = padder.unpad(fflow, bflow)
            return fflow, bflow

        pipeline.set_flownet(compute_flow)

    return pipeline


def load_weights_if_available(model_key, device="cpu"):
    import torch

    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model: {model_key}")

    method = entry.get("method", "rife")

    if method == "pervfi":
        return _load_pervfi_model(model_key, device)

    if not is_weight_downloaded(model_key):
        raise FileNotFoundError(
            f"Weights not found for {model_key}. Run: amverge interpolate --model {model_key} --download"
        )

    path = get_weight_path(model_key)
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    from .rife_arch import RIFEModel, infer_config

    state_dict = _extract_rife_state_dict(checkpoint)
    try:
        cfg = infer_config(state_dict)
    except ValueError as e:
        raise RuntimeError(
            f"Model '{model_key}' uses an unsupported architecture ({e}). "
            f"This build supports RIFE IFNet checkpoints only."
        )

    model = RIFEModel(cfg).to(device)
    model.eval()

    prefixed = {
        (k if k.startswith("flownet.") else f"flownet.{k}"): v
        for k, v in state_dict.items()
    }
    incompatible = model.load_state_dict(prefixed, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint for '{model_key}' does not match its inferred architecture "
            f"(missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)})."
        )

    return model
#