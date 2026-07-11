from __future__ import annotations

import logging
import math

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

_logger = logging.getLogger(__name__)

try:
    from accelerate import Accelerator
    _ACCELERATOR = Accelerator()
except ImportError:
    _ACCELERATOR = None


# ---------------------------------------------------------------------------
# thops helpers
# ---------------------------------------------------------------------------

def thops_sum(tensor, dim=None, keepdim=False):
    if dim is None:
        return tensor.sum()
    return tensor.sum(dim=dim, keepdim=keepdim)


def thops_mean(tensor, dim=None, keepdim=False):
    if dim is None:
        return tensor.mean()
    return tensor.mean(dim=dim, keepdim=keepdim)


def thops_split_feature(tensor, split_type="split"):
    C = tensor.size(1)
    if split_type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif split_type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
    else:
        raise ValueError(f"Unknown split_type: {split_type}")


def thops_cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def thops_pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


# ---------------------------------------------------------------------------
# Pure-PyTorch softsplat (no cupy / CUDA)
# ---------------------------------------------------------------------------

def softsplat(tenIn, tenFlow, tenMetric=None, strMode='avg'):
    B, Cin, H, W = tenIn.shape
    Bf, C2, Hf, Wf = tenFlow.shape
    device = tenIn.device

    if H != Hf or W != Wf:
        scale_y = H / max(1, Hf)
        scale_x = W / max(1, Wf)
        tenFlow = F.interpolate(tenFlow, size=(H, W), mode='bilinear', align_corners=False)
        tenFlow[:, 0] = tenFlow[:, 0] * scale_x
        tenFlow[:, 1] = tenFlow[:, 1] * scale_y
        if tenMetric is not None and (tenMetric.shape[2] != H or tenMetric.shape[3] != W):
            tenMetric = F.interpolate(tenMetric, size=(H, W), mode='bilinear', align_corners=False)

    if strMode == 'avg':
        tenIn = torch.cat([tenIn, tenIn.new_ones(B, 1, H, W)], 1)
    elif strMode == 'soft' and tenMetric is not None:
        w = tenMetric.exp()
        tenIn = torch.cat([tenIn * w, w], 1)

    Cout = tenIn.size(1)
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=tenFlow.dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=tenFlow.dtype),
        indexing='ij',
    )
    base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    flow_norm = tenFlow.permute(0, 2, 3, 1).clone()
    flow_norm[..., 0] = flow_norm[..., 0] * (2.0 / max(1, W - 1))
    flow_norm[..., 1] = flow_norm[..., 1] * (2.0 / max(1, H - 1))

    grid = base_grid + flow_norm
    warped = F.grid_sample(tenIn, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    if strMode in ('avg', 'soft') and (strMode == 'avg' or tenMetric is not None):
        norm = warped[:, -1:, :, :].clamp(min=1e-7)
        warped = warped[:, :-1, :, :] / norm

    return warped


# ---------------------------------------------------------------------------
# warp helper (backward warp)
# ---------------------------------------------------------------------------

_tenGrid_cache: dict = {}
_tenFlowDiv_cache: dict = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()), str(tenFlow.dtype))
    if k not in _tenGrid_cache:
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
        _tenGrid_cache[k] = torch.cat([tenHorizontal, tenVertical], 1).to(tenFlow.dtype)
        _tenFlowDiv_cache[k] = torch.tensor(
            [2.0 / (W - 1), 2.0 / (H - 1)],
            dtype=tenFlow.dtype,
            device=tenFlow.device,
        ).view(1, 2, 1, 1)

    g = (_tenGrid_cache[k] + tenFlow * _tenFlowDiv_cache[k]).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput, grid=g, mode='bilinear',
        padding_mode='border', align_corners=True,
    )


# ---------------------------------------------------------------------------
# resize, binary_hole, warp_pyramid helpers
# ---------------------------------------------------------------------------

def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)


def binary_hole(mask):
    return (mask > 0.5).float()


def warp_pyramid(img, flow):
    warped = warp(img, flow)
    return warped


# ---------------------------------------------------------------------------
# normalizing_flow classes
# ---------------------------------------------------------------------------

def squeeze2d(input, factor=2):
    assert factor >= 1
    if factor == 1:
        return input
    B, C, H, W = input.size()
    assert H % factor == 0 and W % factor == 0, f"{H} {W} not divisible by {factor}"
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1
    if factor == 1:
        return input
    factor2 = factor ** 2
    B, C, H, W = input.size()
    assert C % factor2 == 0
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // factor2, H * factor, W * factor)
    return x


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * math.log(2 * math.pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1.0 / (std + 1e-6))

    def forward(self, x):
        _, _, H, W = x.shape
        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)
        log_abs = thops_sum(torch.log(torch.abs(self.scale)))
        logdet = H * W * log_abs
        if self.logdet:
            return self.scale * (x + self.loc), logdet
        return self.scale * (x + self.loc)

    def reverse(self, y):
        return y / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        _, _, H, W = x.shape
        out = F.conv2d(x, self.weight)
        det = torch.slogdet(self.weight.squeeze())[1] * H * W
        return out, det

    def reverse(self, y):
        weight_inv = torch.inverse(self.weight.squeeze().double()).float().unsqueeze(2).unsqueeze(3)
        return F.conv2d(y, weight_inv)


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = scipy.linalg.qr(weight)
        w_p, w_l, w_u = scipy.linalg.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T
        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        _, _, H, W = x.shape
        weight = self.calc_weight()
        out = F.conv2d(x, weight)
        logdet = H * W * thops_sum(self.w_s)
        return out, logdet

    def reverse(self, y):
        weight = self.calc_weight()
        weight_inv = torch.inverse(weight.squeeze().double()).float().unsqueeze(2).unsqueeze(3)
        return F.conv2d(y, weight_inv)


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, x):
        out = self.conv(x)
        out = out * torch.exp(self.scale * 3)
        return out


class condAffineCoupling(nn.Module):
    def __init__(self, in_channel, cin_channel, hidden=512):
        super().__init__()
        self.cin_channel = cin_channel
        self.nn = nn.Sequential(
            nn.Conv2d(in_channel // 2 + cin_channel, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(hidden, in_channel),
        )

    def forward(self, x, cond, logdet=None, reverse=False):
        if not reverse:
            x_a, x_b = thops_split_feature(x, "split")
            h = torch.cat((x_b, cond), dim=1)
            h = self.nn(h)
            s, t = h.chunk(2, dim=1)
            s = torch.tanh(s)
            scale = torch.exp(s)
            y_a = scale * x_a + t
            y_b = x_b
            y = torch.cat((y_a, y_b), dim=1)
            if logdet is not None:
                logdet = logdet + thops_sum(s, dim=[1, 2, 3])
            return y, logdet
        else:
            y_a, y_b = thops_split_feature(x, "split")
            h = torch.cat((y_b, cond), dim=1)
            h = self.nn(h)
            s, t = h.chunk(2, dim=1)
            s = torch.tanh(s)
            scale = torch.exp(s)
            x_a = (y_a - t) / scale
            x_b = y_b
            x = torch.cat((x_a, x_b), dim=1)
            return x


class condAffineCouplingBN(nn.Module):
    def __init__(self, in_channel, cin_channel, hidden=512):
        super().__init__()
        self.cin_channel = cin_channel
        self.nn = nn.Sequential(
            nn.Conv2d(in_channel // 2 + cin_channel, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            ZeroConv2d(hidden, in_channel),
        )

    def forward(self, x, cond, logdet=None, reverse=False):
        if not reverse:
            x_a, x_b = thops_split_feature(x, "split")
            h = torch.cat((x_b, cond), dim=1)
            h = self.nn(h)
            s, t = h.chunk(2, dim=1)
            s = torch.tanh(s)
            scale = torch.exp(s)
            y_a = scale * x_a + t
            y_b = x_b
            y = torch.cat((y_a, y_b), dim=1)
            if logdet is not None:
                logdet = logdet + thops_sum(s, dim=[1, 2, 3])
            return y, logdet
        else:
            y_a, y_b = thops_split_feature(x, "split")
            h = torch.cat((y_b, cond), dim=1)
            h = self.nn(h)
            s, t = h.chunk(2, dim=1)
            s = torch.tanh(s)
            scale = torch.exp(s)
            x_a = (y_a - t) / scale
            x_b = y_b
            x = torch.cat((x_a, x_b), dim=1)
            return x


class Flow(nn.Module):
    def __init__(self, in_channel, cin_channel, affine_coupling, use_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        if use_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
        self.coupling = affine_coupling(in_channel, cin_channel)

    def forward(self, x, cond, logdet=None, reverse=False):
        if not reverse:
            x, logdet = self.actnorm(x)
            x, det = self.invconv(x)
            logdet = logdet + det
            x, logdet = self.coupling(x, cond, logdet, reverse)
            return x, logdet
        else:
            x = self.coupling(x, cond, reverse=reverse)
            x = self.invconv.reverse(x)
            x = self.actnorm.reverse(x)
            return x


class Block(nn.Module):
    def __init__(self, in_channel, cin_channel, n_flow, affine_coupling, split=True, use_lu=True):
        super().__init__()
        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList()
        for _ in range(n_flow):
            self.flows.append(Flow(squeeze_dim, cin_channel, affine_coupling, use_lu))
        self.split = split

    def forward(self, x, cond, logdet=None, reverse=False, z_stored=None):
        if not reverse:
            x = squeeze2d(x, 2)
            for f in self.flows:
                x, logdet = f(x, cond, logdet, reverse)
            if self.split:
                out, z_new = thops_split_feature(x, "split")
                return out, logdet, z_new
            return x, logdet, None
        else:
            if self.split and z_stored is not None:
                x = torch.cat([x, z_stored], dim=1)
            for f in reversed(self.flows):
                x = f(x, cond, reverse=reverse)
            x = unsqueeze2d(x, 2)
            return x


class CondFlowNet(nn.Module):
    def __init__(self, image_shape, n_levels=3, n_flows=8, hidden_channel=512, use_lu=True, cond_channels=33):
        super().__init__()
        H, W, C_in = image_shape
        self.n_levels = n_levels
        self.blocks = nn.ModuleList()
        n_channel = C_in
        for i in range(n_levels):
            self.blocks.append(Block(n_channel, cond_channels, n_flows, condAffineCouplingBN, split=i < (n_levels - 1), use_lu=use_lu))
            n_channel *= 2
        self.z_shape = None
        self.cond_cache = None

    def encode(self, x, cond, logdet=0.0):
        out = x
        z_outs = []
        for i, block in enumerate(self.blocks):
            cond_i = cond[i] if isinstance(cond, list) else cond
            out, logdet, z_new = block(out, cond_i, logdet, reverse=False)
            if z_new is not None:
                z_outs.append(z_new)
        return out, logdet, z_outs

    def decode(self, z_list, cond, logdet=0.0):
        z = z_list[-1]
        n_blocks = len(self.blocks)
        for i, block in enumerate(reversed(self.blocks)):
            cond_i = cond[n_blocks - 1 - i] if isinstance(cond, list) else cond
            stored = None
            if block.split and i > 0:
                stored = z_list[-(i + 1)]
            z = block(z, cond_i, reverse=True, z_stored=stored)
        return z

    def forward(self, zs=None, inps=None, time=None, code="encode"):
        if code == "encode":
            return self.encode(zs[0], inps)
        elif code == "decode":
            return self.decode(zs, inps)
        return None


# ---------------------------------------------------------------------------
# msfusion classes
# ---------------------------------------------------------------------------

_HRNET_STAGE2 = {
    'NUM_MODULES': 1,
    'NUM_BRANCHES': 3,
    'BLOCK': 'BASIC',
    'NUM_BLOCKS': [2, 2, 2],
    'NUM_CHANNELS': [18, 36, 72],
    'FUSE_METHOD': 'SUM',
}

_HRNET_STAGE3 = {
    'NUM_MODULES': 4,
    'NUM_BRANCHES': 4,
    'BLOCK': 'BASIC',
    'NUM_BLOCKS': [2, 2, 2, 2],
    'NUM_CHANNELS': [18, 36, 72, 144],
    'FUSE_METHOD': 'SUM',
}

_HRNET_STAGE4 = {
    'NUM_MODULES': 3,
    'NUM_BRANCHES': 5,
    'BLOCK': 'BASIC',
    'NUM_BLOCKS': [2, 2, 2, 2, 2],
    'NUM_CHANNELS': [18, 36, 72, 144, 288],
    'FUSE_METHOD': 'SUM',
}

cfg = {"hrnetv2_w18": [_HRNET_STAGE2, _HRNET_STAGE3, _HRNET_STAGE4]}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels,
                 fuse_method, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = list(num_inchannels)
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels, fuse_method)
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            raise ValueError(f"NUM_BRANCHES({num_branches}) != NUM_BLOCKS({len(num_blocks)})")
        if num_branches != len(num_channels):
            raise ValueError(f"NUM_BRANCHES({num_branches}) != NUM_CHANNELS({len(num_channels)})")
        if num_branches != len(num_inchannels):
            raise ValueError(f"NUM_BRANCHES({num_branches}) != num_inchannels({len(num_inchannels)})")

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=0.1),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self, num_branches, num_channels, fuse_method):
        if num_branches == 1:
            return None
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels[j], num_channels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_channels[i], momentum=0.1),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest'),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_channels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_channels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.1),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_channels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_channels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_outchannels_conv3x3, momentum=0.1),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class _TransitionLayer(nn.Sequential):
    def __init__(self, num_channels_pre_layer, num_channels_cur_layer):
        super().__init__()
        for i in range(len(num_channels_cur_layer)):
            if i < len(num_channels_pre_layer):
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    self.add_module(
                        f"transition_{i}",
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i], momentum=0.1),
                            nn.ReLU(inplace=True),
                        ),
                    )
                else:
                    self.add_module(f"transition_{i}", nn.Identity())
            else:
                conv3x3s = []
                for k in range(i + 1 - len(num_channels_pre_layer)):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if k == i - len(num_channels_pre_layer) else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=0.1),
                            nn.ReLU(inplace=True),
                        )
                    )
                self.add_module(f"transition_{i}", nn.Sequential(*conv3x3s))


class MultiscaleFuse(nn.Module):
    def __init__(self, cfg_stages, input_channels=[3, 6], catfeat=12):
        super().__init__()
        self.stages = nn.ModuleList()
        self.conv1 = nn.Conv2d(catfeat, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        num_channels = cfg_stages[0]['NUM_CHANNELS']
        block = blocks_dict[cfg_stages[0]['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        transition_layer = _TransitionLayer([64], num_channels)
        self.transition1 = transition_layer

        stage, pre_stage_channels = self._make_stage(
            cfg_stages[0], num_channels[0], input_channels
        )
        self.stages.append(stage)
        self.pre_stage_channels = pre_stage_channels

        for i in range(1, len(cfg_stages)):
            num_channels_cur = cfg_stages[i]['NUM_CHANNELS']
            block = blocks_dict[cfg_stages[i]['BLOCK']]
            num_channels_cur = [num_channels_cur[i] * block.expansion for i in range(len(num_channels_cur))]
            transition_layer = _TransitionLayer(self.pre_stage_channels, num_channels_cur)
            setattr(self, f"transition{i + 1}", transition_layer)
            stage, pre_stage_channels = self._make_stage(
                cfg_stages[i], num_channels_cur[0], input_channels
            )
            self.stages.append(stage)
            self.pre_stage_channels = pre_stage_channels

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                sum(self.pre_stage_channels),
                256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def _make_stage(self, layer_config, num_inchannels, input_channels):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            if not modules:
                num_inchannels_this = [num_inchannels * block.expansion] if i == 0 else num_inchannels
                num_inchannels_this += [
                    num_channels[j] * block.expansion for j in range(1, num_branches)
                ]
            else:
                num_inchannels_this = self.pre_stage_channels

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels_this,
                    num_channels,
                    fuse_method,
                    multi_scale_output=(i == num_modules - 1),
                )
            )
            if i < num_modules - 1:
                self.pre_stage_channels = modules[-1].get_num_inchannels()

        num_outchannels = [
            num_channels[i] * block.expansion for i in range(num_channels.__len__())
        ]
        self.pre_stage_channels = num_outchannels
        return nn.Sequential(*modules), num_outchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        y_list = [t_branch(x) for t_branch in self.transition1]
        stage = self.stages[0]
        y_list = stage(y_list)
        for i in range(1, len(self.stages)):
            transition_layer = getattr(self, f"transition{i + 1}")
            x_list = [t_branch(y_list[-1] if j == 0 else y_list[j - 1]) for j, t_branch in enumerate(transition_layer)]
            y_list = self.stages[i](x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x = torch.cat(
            [y_list[0]]
            + [
                F.interpolate(y_list[i], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
                for i in range(1, len(y_list))
            ],
            dim=1,
        )
        x = self.last_layer(x)
        return x


# ---------------------------------------------------------------------------
# softsplatnet helpers (only classes used by generators)
# ---------------------------------------------------------------------------

class CropParameters:
    def __init__(self, height, width):
        self.height = height
        self.width = width


class Encode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Softmetric(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 3 + 3 + 2 + 2 + 3 + 2, 32, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, img0, img1, fflow, bflow, merged, metric_in):
        tenHorizontal = torch.linspace(-1.0, 1.0, img0.shape[3], device=img0.device, dtype=img0.dtype).view(1, 1, 1, img0.shape[3]).expand(img0.shape[0], -1, img0.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, img0.shape[2], device=img0.device, dtype=img0.dtype).view(1, 1, img0.shape[2], 1).expand(img0.shape[0], -1, -1, img0.shape[3])
        coords = torch.cat([tenHorizontal, tenVertical], 1)
        x = torch.cat([img0, img1, fflow, bflow, merged, metric_in, coords], 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class Basic(nn.Module):
    def __init__(self, int_channel):
        super().__init__()
        self.moduleConv1 = nn.Sequential(
            nn.Conv2d(in_channels=int_channel, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
        )
        self.moduleConv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
        )
        self.moduleConv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
        )
        self.moduleConv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=False),
        )
        self.moduleConv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        tensor_conv1 = self.moduleConv1(x)
        tensor_conv2 = self.moduleConv2(tensor_conv1)
        tensor_conv3 = self.moduleConv3(tensor_conv2)
        tensor_conv4 = self.moduleConv4(tensor_conv3)
        tensor_conv5 = self.moduleConv5(tensor_conv4)
        return tensor_conv5


# ---------------------------------------------------------------------------
# Generator Components
# ---------------------------------------------------------------------------

class FeaturePyramid(nn.Module):
    def __init__(self, in_channels=3, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.conv_in = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.pyramid = nn.ModuleList()
        for i in range(num_levels):
            self.pyramid.append(
                nn.Sequential(
                    nn.Conv2d(32, 32, 3, 2, 1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.LeakyReLU(0.2, True),
                )
            )

    def forward(self, x):
        feats = []
        f = self.conv_in(x)
        feats.append(f)
        for layer in self.pyramid:
            f = layer(f)
            feats.append(f)
        return feats


class SoftBinary(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


class DCNPack(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x, offset):
        return super().forward(x, offset)


class DeformableAlign(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, 3 * 3 * 2, 3, 1, 1),
        )
        self.dcn = DCNPack(in_channels, in_channels, 3, 1, 1)

    def forward(self, feat_src, feat_ref):
        offset = self.offset_conv(torch.cat([feat_src, feat_ref], dim=1))
        aligned = self.dcn(feat_src, offset)
        return aligned


class AttentionMerge(nn.Module):
    def __init__(self, in_channels, dilate_size=3):
        super().__init__()
        self.dilate = dilate_size
        self.align0 = DeformableAlign(in_channels)
        self.align1 = DeformableAlign(in_channels)
        mid_channels = in_channels * 2
        self.merge = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(mid_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.soft_binary = SoftBinary(mid_channels)

    def forward(self, feat0, feat1, combined):
        aligned0 = self.align0(feat0, combined)
        aligned1 = self.align1(feat1, combined)
        merged = torch.cat([aligned0, aligned1], dim=1)
        gate = self.soft_binary(merged)
        fused = gate * aligned0 + (1.0 - gate) * aligned1
        fused = self.merge(torch.cat([fused, combined], dim=1))
        return fused


# ---------------------------------------------------------------------------
# NetworkV0 -- Generator with CondFlowNet (normalizing flow)
# ---------------------------------------------------------------------------

class NetworkV0(nn.Module):
    def __init__(self, dilate_size=9):
        super().__init__()
        self.dilate_size = dilate_size
        self.feat_pyramid = FeaturePyramid(in_channels=3, num_levels=3)
        self.feat_pyramid_flow = FeaturePyramid(in_channels=4, num_levels=3)

        self.merge0 = AttentionMerge(32, dilate_size)
        self.merge1 = AttentionMerge(32, dilate_size)

        self.softmetric = Softmetric(in_channels=32)

        self.context_encoder = nn.Sequential(
            nn.Conv2d(32 * 4 + 4, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        cond_channels = 32 + 3
        self.condFLownet = CondFlowNet(
            image_shape=(64, 64, 3),
            n_levels=3,
            n_flows=4,
            hidden_channel=256,
            use_lu=True,
            cond_channels=33,
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1),
        )

    def _build_cond(self, img0, img1, fflow, bflow, time):
        feat0 = self.feat_pyramid(img0)
        feat1 = self.feat_pyramid(img1)
        flow_in = torch.cat([fflow, bflow], dim=1)
        feat_flow = self.feat_pyramid_flow(flow_in)

        flow_warped_feat0 = [softsplat(f, fflow) for f in feat0]
        flow_warped_feat1 = [softsplat(f, bflow) for f in feat1]

        combined0 = self.merge0(feat0[0], feat1[0], flow_warped_feat0[0])
        combined1 = self.merge1(feat0[0], feat1[0], flow_warped_feat1[0])

        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, device=img0.device, dtype=img0.dtype)
        time_t = time.view(1, 1, 1, 1).expand(img0.size(0), 1, img0.size(2), img0.size(3))
        warped_img0 = warp(img0, fflow)
        warped_img1 = warp(img1, bflow)
        merged = warped_img0 * (1 - time_t) + warped_img1 * time_t
        metric = self.softmetric(img0, img1, fflow, bflow, merged, combined0)

        cond = self.context_encoder(
            torch.cat([combined0, combined1, feat0[0], feat1[0], fflow, bflow], dim=1)
        )
        time_cond = time.view(1, 1, 1, 1).expand(img0.size(0), 1, cond.size(2), cond.size(3))
        cond = torch.cat([cond, time_cond], dim=1)
        return cond, merged, metric, warped_img0, warped_img1

    def _cond_list(self, cond):
        cond_list = []
        for _ in range(3):
            cond = F.interpolate(cond, scale_factor=0.5, mode='bilinear', align_corners=False)
            cond_list.append(cond)
        return cond_list

    def forward(self, zs=None, inps=None, time=None, code="decode"):
        if code == "decode":
            img0, img1, fflow, bflow = inps
            cond, merged_base, metric, warped_img0, warped_img1 = self._build_cond(img0, img1, fflow, bflow, time)
            cond_list = self._cond_list(cond)
            residual = self.condFLownet(zs=zs, inps=cond_list, code="decode")
            residual = F.interpolate(residual, size=img0.shape[2:], mode='bilinear', align_corners=False)
            pred = merged_base + residual
            return torch.clamp(pred, 0.0, 1.0), None
        return None, None


# ---------------------------------------------------------------------------
# NetworkVb -- Generator with multi-scale Decoder
# ---------------------------------------------------------------------------

class NetworkVb(nn.Module):
    def __init__(self, dilate_size=9):
        super().__init__()
        self.dilate_size = dilate_size
        self.feat_pyramid = FeaturePyramid(in_channels=3, num_levels=3)
        self.feat_pyramid_flow = FeaturePyramid(in_channels=4, num_levels=3)

        self.merge0 = AttentionMerge(32, dilate_size)
        self.merge1 = AttentionMerge(32, dilate_size)

        self.ms_fuse = MultiscaleFuse(cfg["hrnetv2_w18"])

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

        self.context_encoder = nn.Sequential(
            nn.Conv2d(32 * 3 + 6 + 4, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def _build_cond(self, img0, img1, fflow, bflow, time):
        feat0 = self.feat_pyramid(img0)
        feat1 = self.feat_pyramid(img1)
        flow_in = torch.cat([fflow, bflow], dim=1)
        feat_flow = self.feat_pyramid_flow(flow_in)

        warped_feat0 = [softsplat(f, fflow) for f in feat0]
        warped_feat1 = [softsplat(f, bflow) for f in feat1]

        combined = self.merge0(feat0[0], feat1[0], warped_feat0[0])

        cond = self.context_encoder(
            torch.cat([combined, feat0[0], feat1[0], fflow, bflow], dim=1)
        )
        return cond

    def forward(self, zs=None, inps=None, time=None, code="decode"):
        img0, img1, fflow, bflow = inps
        B, _, H, W = img0.shape
        time_tensor = time
        if not isinstance(time_tensor, torch.Tensor):
            time_tensor = torch.tensor(time, device=img0.device, dtype=img0.dtype)
        time_tensor = time_tensor.view(1, 1, 1, 1).expand(B, 1, H, W)

        warped_img0 = warp(img0, fflow)
        warped_img1 = warp(img1, bflow)
        t_img0 = warped_img0 * time_tensor
        t_img1 = warped_img1 * (1.0 - time_tensor)

        ms_input = torch.cat([img0, img1, t_img0, t_img1], dim=1)
        ms_feat = self.ms_fuse(ms_input)

        pred = self.decoder(ms_feat)
        return torch.clamp(pred, 0.0, 1.0), None


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

def build_generator_arch(version):
    if version.lower() == "v00":
        model = NetworkV0(dilate_size=9)
    elif version.lower() == "vb":
        model = NetworkVb(dilate_size=9)
    else:
        raise ValueError(f"Unknown generator version: {version}")
    return model


class Pipeline_infer(nn.Module):
    def __init__(self, generator, model_file, device='cuda'):
        super().__init__()
        self.netG = build_generator_arch(generator)
        state_dict = {k.replace('module.', ''): v for k, v in torch.load(model_file, map_location=device, weights_only=True).items()}
        self.netG.load_state_dict(state_dict, strict=False)
        self.netG.to(device).eval()
        self.device = device
        self._flownet = None

    def set_flownet(self, compute_flow_fn):
        self._compute_flow = compute_flow_fn

    def get_z(self, heat, img_size, batch, device):
        def calc_z_shapes(img_size, n_levels):
            h, w = img_size
            z_shapes = []
            channel = 3
            for _ in range(n_levels - 1):
                h //= 2
                w //= 2
                channel *= 2
                z_shapes.append((channel, h, w))
            h //= 2
            w //= 2
            z_shapes.append((channel * 4, h, w))
            return z_shapes
        z_list = []
        z_shapes = calc_z_shapes(img_size, 3)
        for z in z_shapes:
            z_new = torch.randn(batch, *z, device=device) * heat
            z_list.append(z_new)
        return z_list

    @torch.no_grad()
    def inference_rand_noise(self, img0, img1, heat=0.3, time=0.5):
        zs = self.get_z(heat, img0.shape[-2:], img0.shape[0], img0.device)
        fflow, bflow = self._compute_flow(img0, img1)
        conds = [img0, img1, fflow, bflow]
        pred, _ = self.netG(zs=zs, inps=conds, time=time, code="decode")
        return torch.clamp(pred, 0.0, 1.0)
