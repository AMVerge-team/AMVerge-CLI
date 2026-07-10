from __future__ import annotations

"""RIFE frame-interpolation architectures.

The RIFE family ships several checkpoint shapes that all share the IFNet
skeleton but differ in three ways:

* number of coarse-to-fine blocks (4 or 5),
* the context encoder (a 4-conv ``Head`` for v2, a 2-conv encoder for Elexor,
  or none at all for the v1 line), and
* per-block channel widths and the per-block output width (``out_ch``).

Rather than hardcode one shape, :class:`IFNet` is driven by an
:class:`IFNetConfig`, and :func:`infer_config` derives that config straight
from a checkpoint's tensor shapes. So any compatible RIFE checkpoint loads —
v2 (4.20-4.25, heavy, lite), Elexor, and the v1 line (4.6, ...) — without a
per-model table. Non-RIFE checkpoints (e.g. PerVFI) raise a clear error.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_tenGrid: dict = {}
_tenFlowDiv: dict = {}


def _warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()), str(tenFlow.dtype))
    if k not in _tenGrid:
        H, W = tenFlow.shape[2], tenFlow.shape[3]
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, W, device=tenFlow.device, dtype=torch.float32)
            .view(1, 1, 1, W)
            .expand(tenFlow.shape[0], -1, H, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, H, device=tenFlow.device, dtype=torch.float32)
            .view(1, 1, H, 1)
            .expand(tenFlow.shape[0], -1, -1, W)
        )
        _tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.dtype)
        _tenFlowDiv[k] = torch.tensor(
            [2.0 / (W - 1), 2.0 / (H - 1)],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        ).view(1, 2, 1, 1)

    g = (_tenGrid[k] + tenFlow * _tenFlowDiv[k]).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput, grid=g, mode="bilinear",
        padding_mode="border", align_corners=True,
    )


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True),
    )


class Head4(nn.Module):
    """v2 context encoder (RIFE 4.20-4.25). Emits a 4-channel feature map."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class Head2(nn.Sequential):
    """Elexor context encoder: Conv down -> ConvTranspose up. 4-channel out.

    Subclasses ``Sequential`` so its checkpoint keys are ``encode.0`` /
    ``encode.1`` (matching the Elexor weights), while a custom ``forward``
    applies the LeakyReLU between the two convs.
    """

    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 4, 2, 1),
        )

    def forward(self, x):
        x = F.leaky_relu(self[0](x), 0.2, inplace=True)
        return self[1](x)


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """One coarse-to-fine flow block. ``out_ch`` = channels after PixelShuffle;
    the block emits flow (4) + mask (1) + feat (out_ch - 5)."""

    def __init__(self, in_planes, c=64, out_ch=13):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * out_ch, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
                * 1.0 / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


@dataclass
class IFNetConfig:
    """Shape of a concrete RIFE IFNet, derived from its checkpoint.

    channels:        per-block width ``c`` (length = number of blocks).
    out_ch:          per-block output channels after PixelShuffle (13 for v2's
                     8-channel feat, 6 for the 1-channel-feat variants).
    encode:          ``"head4"`` (v2), ``"head2"`` (Elexor), or ``None`` (v1).
    propagate_feat:  whether each block's feat output feeds the next block
                     (v2) or is dropped (Elexor, v1).
    """

    channels: list
    out_ch: int
    encode: str | None
    propagate_feat: bool

    @property
    def num_blocks(self) -> int:
        return len(self.channels)


# Coarse-to-fine scales; sliced from the tail so a 4-block net uses [8,4,2,1]
# and a 5-block net uses [16,8,4,2,1].
_SCALES = [16, 8, 4, 2, 1]


class IFNet(nn.Module):
    def __init__(self, cfg: IFNetConfig):
        super().__init__()
        self.cfg = cfg

        # Context encoder (optional). Its feature width is fixed at 4 channels.
        encode_feat = 0
        if cfg.encode == "head4":
            self.encode = Head4()
            encode_feat = 4
        elif cfg.encode == "head2":
            self.encode = Head2()
            encode_feat = 4
        else:
            self.encode = None

        feat_ch = cfg.out_ch - 5  # flow(4) + mask(1) + feat(rest)


        in_first = 6 + 2 * encode_feat + 1
        in_rest = 6 + 2 * encode_feat + 1 + 1 + (feat_ch if cfg.propagate_feat else 0) + 4

        self.blocks = []
        for i, c in enumerate(cfg.channels):
            in_planes = in_first if i == 0 else in_rest
            block = IFBlock(in_planes, c=c, out_ch=cfg.out_ch)
            setattr(self, f"block{i}", block)  # register as block0, block1, ...
            self.blocks.append(block)

        self.scale_list = _SCALES[-cfg.num_blocks:]
        self.f0 = None
        self.f1 = None

    def _encode(self, img):
        return self.encode(img[:, :3]) if self.encode is not None else None

    def cacheReset(self, frame):
        self.f0 = self._encode(frame)
        self.f1 = None

    def cachePair(self, img0, img1):
        self.f0 = self._encode(img0)
        self.f1 = self._encode(img1)

    def forward(self, img0, img1, timestep=0.5):
        if self.encode is not None:
            if self.f0 is None:
                self.f0 = self.encode(img0[:, :3])
            if self.f1 is None:
                self.f1 = self.encode(img1[:, :3])

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=img0.dtype, device=img0.device)
        timestep = torch.ones_like(img0[:, :1, :, :]) * timestep.view(1, 1, 1, 1)

        target_shape = img0.shape[2:]

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        feat = None

        for i in range(self.cfg.num_blocks):
            if flow is None:
                if self.encode is not None:
                    inp = torch.cat((img0[:, :3], img1[:, :3], self.f0, self.f1, timestep), 1)
                else:
                    inp = torch.cat((img0[:, :3], img1[:, :3], timestep), 1)
                flow, mask, feat = self.blocks[i](inp, None, scale=self.scale_list[i])
            else:
                parts = [warped_img0[:, :3], warped_img1[:, :3]]
                if self.encode is not None:
                    parts += [_warp(self.f0, flow[:, :2]), _warp(self.f1, flow[:, 2:4])]
                parts += [timestep, mask]
                if self.cfg.propagate_feat:
                    parts.append(feat)
                fd, m0, feat = self.blocks[i](torch.cat(parts, 1), flow, scale=self.scale_list[i])
                mask = m0
                flow = flow + fd

            if flow.shape[2:] != target_shape:
                flow = F.interpolate(flow, size=target_shape, mode="bilinear", align_corners=False)
            if mask.shape[2:] != target_shape:
                mask = F.interpolate(mask, size=target_shape, mode="bilinear", align_corners=False)
            if feat.shape[2:] != target_shape:
                feat = F.interpolate(feat, size=target_shape, mode="bilinear", align_corners=False)

            warped_img0 = _warp(img0, flow[:, :2])
            warped_img1 = _warp(img1, flow[:, 2:4])

        mask = torch.sigmoid(mask)
        result = warped_img0 * mask + warped_img1 * (1 - mask)
        if self.encode is not None:
            self.f0 = self.f1
        return result


def infer_config(state_dict) -> IFNetConfig:
    """Derive an :class:`IFNetConfig` from a checkpoint's tensor shapes.

    Expects keys without the ``flownet.`` prefix (``block0.conv0...``,
    ``encode...``). Raises ``ValueError`` for non-RIFE checkpoints so callers
    can report a clean "unsupported architecture" message.
    """
    block_ids = sorted(
        {int(k.split(".")[0][len("block"):]) for k in state_dict if k.startswith("block")}
    )
    if not block_ids or block_ids != list(range(len(block_ids))):
        raise ValueError("not a RIFE IFNet checkpoint (no contiguous block0..N)")

    channels = []
    for b in block_ids:
        w = state_dict.get(f"block{b}.conv0.1.0.weight")
        if w is None:
            raise ValueError(f"missing block{b}.conv0 weights")
        channels.append(int(w.shape[0]))

    last = state_dict.get("block0.lastconv.0.weight")
    if last is None:
        raise ValueError("missing block0.lastconv weights")
    out_ch = int(last.shape[1]) // 4

    if "encode.cnn0.weight" in state_dict:
        encode = "head4"
    elif "encode.0.weight" in state_dict:
        encode = "head2"
    else:
        encode = None

    encode_feat = 4 if encode else 0
    block1_in = state_dict.get("block1.conv0.0.0.weight")
    if block1_in is None:
        raise ValueError("missing block1.conv0 weights")
    # Feat is propagated when block1 takes more than the no-feat baseline
    # (imgs 6 + warped encode feats 2*ef + timestep 1 + mask 1 + flow 4).
    no_prop_in = 6 + 2 * encode_feat + 1 + 1 + 4
    propagate_feat = int(block1_in.shape[1]) > no_prop_in

    return IFNetConfig(
        channels=channels, out_ch=out_ch, encode=encode, propagate_feat=propagate_feat
    )


class RIFEModel(nn.Module):
    def __init__(self, cfg: IFNetConfig):
        super().__init__()
        self.flownet = IFNet(cfg)

    def cacheReset(self, frame):
        self.flownet.cacheReset(frame)

    def cachePair(self, img0, img1):
        self.flownet.cachePair(img0, img1)

    def forward(self, img0, img1, timestep=0.5):
        return self.flownet(img0, img1, timestep)
