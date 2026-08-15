from __future__ import annotations

import torch
import pytest

from amverge.core.interpolation.pervfi_gmflow import (
    GMFlow,
    CNNEncoder,
    FeatureTransformer,
    FeatureFlowAttention,
    PositionEmbeddingSine,
    ResidualBlock,
    TransformerLayer,
    TransformerBlock,
    coords_grid,
    flow_warp,
    normalize_img,
    global_correlation_softmax,
    local_correlation_softmax,
    feature_add_position,
    split_feature,
    merge_splits,
)


class TestCoordsGrid:
    def test_shape(self):
        grid = coords_grid(2, 16, 32)
        assert grid.shape == (2, 2, 16, 32)

    def test_values(self):
        grid = coords_grid(1, 3, 3)
        assert grid[0, 1, 0, 0].item() == 0.0
        assert grid[0, 0, 0, 1].item() == 1.0
        assert grid[0, 1, 2, 2].item() == 2.0

    def test_device(self):
        grid = coords_grid(1, 4, 4, device=torch.device('cpu'))
        assert grid.device.type == 'cpu'


class TestFlowWarp:
    def test_identity_flow(self):
        feature = torch.randn(1, 8, 16, 32)
        flow = torch.zeros(1, 2, 16, 32)
        warped = flow_warp(feature, flow)
        assert warped.shape == feature.shape
        assert torch.allclose(warped, feature, atol=1e-5)

    def test_translation(self):
        feature = torch.zeros(1, 1, 64, 64)
        feature[:, :, 10, 10] = 1.0
        flow = torch.zeros(1, 2, 64, 64)
        flow[:, 0] = 5.0
        flow[:, 1] = 3.0
        warped = flow_warp(feature, flow)
        assert warped.shape == feature.shape


class TestNormalizeImg:
    def test_range(self):
        img0 = torch.rand(1, 3, 64, 64) * 255.0
        img1 = torch.rand(1, 3, 64, 64) * 255.0
        n0, n1 = normalize_img(img0, img1)
        assert n0.shape == img0.shape
        assert n1.shape == img1.shape


class TestSplitMerge:
    def test_split_merge_roundtrip(self):
        f = torch.randn(2, 128, 64, 64)
        splits = split_feature(f, num_splits=2, channel_last=False)
        assert splits.shape == (8, 128, 32, 32)
        merged = merge_splits(splits, num_splits=2, channel_last=False)
        assert merged.shape == f.shape
        assert torch.allclose(merged, f, atol=1e-5)

    def test_split_merge_roundtrip_channel_last(self):
        f = torch.randn(2, 64, 64, 128)
        splits = split_feature(f, num_splits=2, channel_last=True)
        assert splits.shape == (8, 32, 32, 128)
        merged = merge_splits(splits, num_splits=2, channel_last=True)
        assert merged.shape == f.shape


class TestPositionEmbeddingSine:
    def test_output_shape(self):
        pe = PositionEmbeddingSine(num_pos_feats=64)
        x = torch.randn(2, 3, 32, 48)
        pos = pe(x)
        assert pos.shape == (2, 128, 32, 48)

    def test_deterministic(self):
        pe = PositionEmbeddingSine(num_pos_feats=64)
        x = torch.randn(1, 3, 16, 16)
        pos1 = pe(x)
        pos2 = pe(x)
        assert torch.allclose(pos1, pos2)


class TestFeatureAddPosition:
    def test_output_shape(self):
        f0 = torch.randn(1, 128, 16, 16)
        f1 = torch.randn(1, 128, 16, 16)
        out0, out1 = feature_add_position(f0, f1, attn_splits=1, feature_channels=128)
        assert out0.shape == f0.shape
        assert out1.shape == f1.shape


class TestGlobalCorrelationSoftmax:
    def test_output_shape(self):
        f0 = torch.randn(2, 128, 8, 8)
        f1 = torch.randn(2, 128, 8, 8)
        flow, prob = global_correlation_softmax(f0, f1, pred_bidir_flow=False)
        assert flow.shape == (2, 2, 8, 8)

    def test_bidir_shape(self):
        f0 = torch.randn(1, 128, 8, 8)
        f1 = torch.randn(1, 128, 8, 8)
        fwd, bwd = global_correlation_softmax(f0, f1, pred_bidir_flow=True)
        assert fwd.shape == (2, 2, 8, 8)


class TestLocalCorrelationSoftmax:
    def test_output_shape(self):
        f0 = torch.randn(2, 128, 16, 16)
        f1 = torch.randn(2, 128, 16, 16)
        flow, prob = local_correlation_softmax(f0, f1, local_radius=4)
        assert flow.shape == (2, 2, 16, 16)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda(self):
        f0 = torch.randn(1, 128, 16, 16, device='cuda')
        f1 = torch.randn(1, 128, 16, 16, device='cuda')
        flow, prob = local_correlation_softmax(f0, f1, local_radius=2)
        assert flow.device.type == 'cuda'


class TestResidualBlock:
    def test_instantiation(self):
        block = ResidualBlock(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_downsample(self):
        block = ResidualBlock(32, 64, stride=2)
        x = torch.randn(2, 32, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)


class TestCNNEncoder:
    def test_instantiation_single_scale(self):
        encoder = CNNEncoder(output_dim=128, num_output_scales=1)
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0].shape[1] == 128

    def test_instantiation_two_scales(self):
        encoder = CNNEncoder(output_dim=128, num_output_scales=2)
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert len(out) == 2


class TestTransformerLayer:
    def test_instantiation(self):
        layer = TransformerLayer(d_model=128, nhead=1)
        src = torch.randn(2, 64, 128)
        tgt = torch.randn(2, 64, 128)
        out = layer(src, tgt, attn_num_splits=1)
        assert out.shape == src.shape

    def test_swin_with_splits(self):
        layer = TransformerLayer(d_model=128, nhead=1, attention_type='swin')
        src = torch.randn(2, 256, 128)
        tgt = torch.randn(2, 256, 128)
        out = layer(src, tgt, height=16, width=16, attn_num_splits=2)
        assert out.shape == src.shape


class TestFeatureTransformer:
    def test_instantiation(self):
        ft = FeatureTransformer(num_layers=2, d_model=128, nhead=1)
        f0 = torch.randn(2, 128, 8, 8)
        f1 = torch.randn(2, 128, 8, 8)
        o0, o1 = ft(f0, f1, attn_num_splits=1)
        assert o0.shape == f0.shape
        assert o1.shape == f1.shape

    def test_with_splits(self):
        ft = FeatureTransformer(num_layers=2, d_model=128, nhead=1, attention_type='swin')
        f0 = torch.randn(1, 128, 16, 16)
        f1 = torch.randn(1, 128, 16, 16)
        o0, o1 = ft(f0, f1, attn_num_splits=2)
        assert o0.shape == f0.shape


class TestFeatureFlowAttention:
    def test_instantiation(self):
        attn = FeatureFlowAttention(in_channels=128)
        f0 = torch.randn(2, 128, 16, 16)
        flow = torch.randn(2, 2, 16, 16)
        out = attn(f0, flow)
        assert out.shape == flow.shape

    def test_local_window(self):
        attn = FeatureFlowAttention(in_channels=128)
        f0 = torch.randn(2, 128, 8, 8)
        flow = torch.randn(2, 2, 8, 8)
        out = attn(f0, flow, local_window_attn=True, local_window_radius=2)
        assert out.shape == flow.shape


class TestGMFlow:
    def test_instantiation(self):
        model = GMFlow(
            num_scales=1,
            upsample_factor=8,
            feature_channels=128,
            num_transformer_layers=2,
            num_head=1,
        )
        assert isinstance(model, torch.nn.Module)

    def test_forward_single_scale(self):
        model = GMFlow(
            num_scales=1,
            upsample_factor=8,
            feature_channels=128,
            num_transformer_layers=2,
            num_head=1,
        )
        model.eval()
        img0 = torch.randn(1, 3, 128, 128) * 255.0
        img1 = torch.randn(1, 3, 128, 128) * 255.0
        with torch.no_grad():
            result = model(
                img0, img1,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=False,
            )
        flow_preds = result['flow_preds']
        assert len(flow_preds) >= 1
        assert flow_preds[-1].shape == (1, 2, 128, 128)

    def test_forward_two_scales(self):
        model = GMFlow(
            num_scales=2,
            upsample_factor=4,
            feature_channels=128,
            num_transformer_layers=2,
            num_head=1,
        )
        img0 = torch.randn(1, 3, 256, 256) * 255.0
        img1 = torch.randn(1, 3, 256, 256) * 255.0
        result = model(
            img0, img1,
            attn_splits_list=[2, 2],
            corr_radius_list=[-1, -1],
            prop_radius_list=[-1, -1],
            pred_bidir_flow=False,
        )
        flow_preds = result['flow_preds']
        assert len(flow_preds) >= 1

    def test_eval_mode(self):
        model = GMFlow(
            num_scales=1,
            upsample_factor=8,
            feature_channels=128,
            num_transformer_layers=2,
            num_head=1,
        )
        model.eval()
        img0 = torch.randn(1, 3, 64, 64) * 255.0
        img1 = torch.randn(1, 3, 64, 64) * 255.0
        with torch.no_grad():
            result = model(
                img0, img1,
                attn_splits_list=[2],
                corr_radius_list=[-1],
                prop_radius_list=[-1],
                pred_bidir_flow=False,
            )
        assert 'flow_preds' in result
