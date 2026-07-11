from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    stacks = [x, y]
    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float()
    grid = grid[None].repeat(b, 1, 1, 1)
    if device is not None:
        grid = grid.to(device)
    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None
    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)])
    grid = torch.stack((x, y), -1).transpose(0, 1).float()
    return grid


def normalize_coords(coords, h, w):
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)
    b, _, h, w = sample_coords.shape
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        return img, mask
    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2
    grid = coords_grid(b, h, w).to(flow.device) + flow
    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)
    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)
    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)
    threshold = alpha * flow_mag + beta
    fwd_occ = (diff_fwd > threshold).float()
    bwd_occ = (diff_bwd > threshold).float()
    return fwd_occ, bwd_occ


def normalize_img(img0, img1):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255.0 - mean) / std
    img1 = (img1 / 255.0 - mean) / std
    return img0, img1


def split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, num_splits, h_new, num_splits, w_new, c).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)
        return feature
    else:
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, c, num_splits, h_new, num_splits, w_new).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)
        return feature


def merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(new_b, num_splits * h, num_splits * w, c)
        return merge
    else:
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(new_b, c, num_splits * h, num_splits * w)
        return merge


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    if attn_splits > 1:
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)
        position = pos_enc(feature0_splits)
        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position
        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)
        feature0 = feature0 + position
        feature1 = feature1 + position
    return feature0, feature1


def generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w,
                                    shift_size_h, shift_size_w, device=torch.device('cuda')):
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)
    h_slices = (slice(0, -window_size_h), slice(-window_size_h, -shift_size_h), slice(-shift_size_h, None))
    w_slices = (slice(0, -window_size_w), slice(-window_size_w, -shift_size_w), slice(-shift_size_w, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)
    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def single_head_full_attention(q, k, v):
    assert q.dim() == k.dim() == v.dim() == 3
    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)
    attn = torch.softmax(scores, dim=2)
    out = torch.matmul(attn, v)
    return out


def single_head_split_window_attention(q, k, v, num_splits=1, with_shift=False, h=None, w=None, attn_mask=None):
    assert q.dim() == k.dim() == v.dim() == 3
    assert h is not None and w is not None
    assert q.size(1) == h * w
    b, _, c = q.size()
    b_new = b * num_splits * num_splits
    window_size_h = h // num_splits
    window_size_w = w // num_splits
    q = q.view(b, h, w, c)
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)
    scale_factor = c ** 0.5
    if with_shift:
        assert attn_mask is not None
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2
        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
    q = split_feature(q, num_splits=num_splits, channel_last=True)
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)
    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)) / scale_factor
    if with_shift:
        scores += attn_mask.repeat(b, 1, 1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.view(b_new, -1, c))
    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c), num_splits=num_splits, channel_last=True)
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))
    out = out.view(b, -1, c)
    return out


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MultiScaleTridentConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, strides=1, paddings=0, dilations=1,
                 dilation=1, groups=1, num_branch=1, test_branch_idx=-1, bias=False, norm=None, activation=None):
        super(MultiScaleTridentConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        self.dilation = dilation
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        if isinstance(strides, int):
            strides = [strides] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation) for dilation in dilations]
        self.strides = [_pair(stride) for stride in strides]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation
        assert len({self.num_branch, len(self.paddings), len(self.strides)}) == 1
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch
        if self.training or self.test_branch_idx == -1:
            outputs = [F.conv2d(input, self.weight, self.bias, stride, padding, self.dilation, self.groups)
                       for input, stride, padding in zip(inputs, self.strides, self.paddings)]
        else:
            outputs = [F.conv2d(inputs[0], self.weight, self.bias,
                                self.strides[self.test_branch_idx] if self.test_branch_idx == -1 else self.strides[-1],
                                self.paddings[self.test_branch_idx] if self.test_branch_idx == -1 else self.paddings[-1],
                                self.dilation, self.groups)]
        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d, num_output_scales=1, **kwargs):
        super().__init__()
        self.num_branch = num_output_scales
        feature_dims = [64, 96, 128]
        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)
        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError
            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim, kernel_size=3, strides=strides, paddings=1, num_branch=self.num_branch)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)
        else:
            out = [x]
        return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model=256, nhead=1, attention_type='swin', no_ffn=False, ffn_dim_expansion=4, with_shift=False, **kwargs):
        super(TransformerLayer, self).__init__()
        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.with_shift = with_shift
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target, height=None, width=None, shifted_window_attn_mask=None, attn_num_splits=None, **kwargs):
        query, key, value = source, target, target
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        if self.attention_type == 'swin' and attn_num_splits is not None and attn_num_splits > 1:
            if self.nhead > 1:
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value, num_splits=attn_num_splits, with_shift=self.with_shift, h=height, w=width, attn_mask=shifted_window_attn_mask)
        else:
            message = single_head_full_attention(query, key, value)
        message = self.merge(message)
        message = self.norm1(message)
        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)
        return source + message


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=1, attention_type='swin', ffn_dim_expansion=4, with_shift=False, **kwargs):
        super(TransformerBlock, self).__init__()
        self.self_attn = TransformerLayer(d_model=d_model, nhead=nhead, attention_type=attention_type, no_ffn=True, ffn_dim_expansion=ffn_dim_expansion, with_shift=with_shift)
        self.cross_attn_ffn = TransformerLayer(d_model=d_model, nhead=nhead, attention_type=attention_type, ffn_dim_expansion=ffn_dim_expansion, with_shift=with_shift)

    def forward(self, source, target, height=None, width=None, shifted_window_attn_mask=None, attn_num_splits=None, **kwargs):
        source = self.self_attn(source, source, height=height, width=width, shifted_window_attn_mask=shifted_window_attn_mask, attn_num_splits=attn_num_splits)
        source = self.cross_attn_ffn(source, target, height=height, width=width, shifted_window_attn_mask=shifted_window_attn_mask, attn_num_splits=attn_num_splits)
        return source


class FeatureTransformer(nn.Module):
    def __init__(self, num_layers=6, d_model=128, nhead=1, attention_type='swin', ffn_dim_expansion=4, **kwargs):
        super(FeatureTransformer, self).__init__()
        self.attention_type = attention_type
        self.d_model = d_model
        self.nhead = nhead
        self.layers = nn.ModuleList([TransformerBlock(d_model=d_model, nhead=nhead, attention_type=attention_type, ffn_dim_expansion=ffn_dim_expansion, with_shift=True if attention_type == 'swin' and i % 2 == 1 else False) for i in range(num_layers)])
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1, attn_num_splits=None, **kwargs):
        b, c, h, w = feature0.shape
        assert self.d_model == c
        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)
        if self.attention_type == 'swin' and attn_num_splits > 1:
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits
            shifted_window_attn_mask = generate_shift_window_attn_mask(input_resolution=(h, w), window_size_h=window_size_h, window_size_w=window_size_w, shift_size_h=window_size_h // 2, shift_size_w=window_size_w // 2, device=feature0.device)
        else:
            shifted_window_attn_mask = None
        concat0 = torch.cat((feature0, feature1), dim=0)
        concat1 = torch.cat((feature1, feature0), dim=0)
        for layer in self.layers:
            concat0 = layer(concat0, concat1, height=h, width=w, shifted_window_attn_mask=shifted_window_attn_mask, attn_num_splits=attn_num_splits)
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)
        feature0, feature1 = concat0.chunk(chunks=2, dim=0)
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return feature0, feature1


class FeatureFlowAttention(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(FeatureFlowAttention, self).__init__()
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow, local_window_attn=False, local_window_radius=1, **kwargs):
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow, local_window_radius=local_window_radius)
        b, c, h, w = feature0.size()
        query = feature0.view(b, c, h * w).permute(0, 2, 1)
        query = self.q_proj(query)
        key = self.k_proj(query)
        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)
        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, value)
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)
        return out

    def forward_local_window_attn(self, feature0, flow, local_window_radius=1):
        assert flow.size(1) == 2
        assert local_window_radius > 0
        b, c, h, w = feature0.size()
        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)).reshape(b * h * w, 1, c)
        kernel_size = 2 * local_window_radius + 1
        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)
        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size, padding=local_window_radius)
        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)
        flow_window = F.unfold(flow, kernel_size=kernel_size, padding=local_window_radius)
        flow_window = flow_window.view(b, 2, kernel_size ** 2, h, w).permute(0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, 2)
        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, flow_window).view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()
        return out


def global_correlation_softmax(feature0, feature1, pred_bidir_flow=False):
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)
    feature1 = feature1.view(b, c, -1)
    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)
    init_grid = coords_grid(b, h, w).to(correlation.device)
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)
    correlation = correlation.view(b, h * w, h * w)
    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)
        init_grid = init_grid.repeat(2, 1, 1, 1)
        grid = grid.repeat(2, 1, 1)
        b = b * 2
    prob = F.softmax(correlation, dim=-1)
    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)
    flow = correspondence - init_grid
    return flow, prob


def local_correlation_softmax(feature0, feature1, local_radius, padding_mode='zeros'):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius, -local_radius, local_radius, local_h, local_w, device=feature0.device)
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid
    sample_coords_softmax = sample_coords
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    sample_coords_norm = normalize_coords(sample_coords, h, w)
    window_feature = F.grid_sample(feature1, sample_coords_norm, padding_mode=padding_mode, align_corners=True).permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)
    corr[~valid] = -1e9
    prob = F.softmax(corr, -1)
    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(b, h, w, 2).permute(0, 3, 1, 2)
    flow = correspondence - coords_init
    match_prob = prob
    return flow, match_prob


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs):
        super(GMFlow, self).__init__()
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers, d_model=feature_channels, nhead=num_head, attention_type=attention_type, ffn_dim_expansion=ffn_dim_expansion)
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)
        features = self.backbone(concat)
        features = features[::-1]
        feature0, feature1 = [], []
        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)
            feature0.append(chunks[0])
            feature1.append(chunks[1])
        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor, mode='bilinear', align_corners=True) * upsample_factor
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)
            mask = torch.softmax(mask, dim=2)
            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)
            up_flow = torch.sum(mask * up_flow, dim=2)
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h, self.upsample_factor * w)
        return up_flow

    def forward(self, img0, img1, attn_splits_list=None, corr_radius_list=None, prop_radius_list=None, pred_bidir_flow=False, **kwargs):
        results_dict = {}
        flow_preds = []
        img0, img1 = normalize_img(img0, img1)
        feature0_list, feature1_list = self.extract_feature(img0, img1)
        flow = None
        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
            if pred_bidir_flow and scale_idx > 0:
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)
            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))
            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)
            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            if corr_radius == -1:
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
            flow = flow + flow_pred if flow is not None else flow_pred
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)
            flow = self.feature_flow_attn(feature0, flow.detach(), local_window_attn=prop_radius > 0, local_window_radius=prop_radius)
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)
            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)
        results_dict.update({'flow_preds': flow_preds})
        return results_dict
