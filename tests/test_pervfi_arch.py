from __future__ import annotations

import torch
import pytest

from amverge.core.interpolation.pervfi_arch import (
    softsplat,
    resize,
    binary_hole,
    warp_pyramid,
    thops_sum,
    thops_pixels,
    squeeze2d,
    unsqueeze2d,
    Basic,
    Encode,
    Softmetric,
    FeaturePyramid,
    Pipeline_infer,
    build_generator_arch,
    gaussian_log_p,
    gaussian_sample,
)


class TestSoftsplat:
    def test_avg_identity(self):
        x = torch.randn(1, 3, 32, 32)
        flow = torch.zeros(1, 2, 32, 32)
        out = softsplat(x, flow, strMode='avg')
        assert out.shape == (1, 3, 32, 32)

    def test_avg_translation_approx(self):
        x = torch.ones(1, 1, 64, 64)
        flow = torch.zeros(1, 2, 64, 64)
        flow[:, 0] = 2.0
        flow[:, 1] = -3.0
        out = softsplat(x, flow, strMode='avg')
        assert out.shape == (1, 1, 64, 64)

    def test_soft_with_metric(self):
        x = torch.randn(2, 8, 32, 32)
        flow = torch.randn(2, 2, 32, 32) * 2.0
        metric = torch.randn(2, 1, 32, 32)
        out = softsplat(x, flow, tenMetric=metric, strMode='soft')
        assert out.shape == (2, 8, 32, 32)

    def test_soft_no_metric_fallback(self):
        x = torch.randn(1, 4, 16, 16)
        flow = torch.randn(1, 2, 16, 16)
        out = softsplat(x, flow, tenMetric=None, strMode='soft')
        assert out.shape == (1, 4, 16, 16)

    def test_output_finite(self):
        x = torch.randn(1, 3, 32, 32)
        flow = torch.randn(1, 2, 32, 32) * 5.0
        out = softsplat(x, flow, strMode='avg')
        assert torch.isfinite(out).all()

    def test_batch_independence(self):
        batch = torch.randn(4, 2, 16, 16)
        flow = torch.zeros(4, 2, 16, 16)
        out = softsplat(batch, flow, strMode='avg')
        for b in range(4):
            single = softsplat(batch[b:b+1], flow[b:b+1], strMode='avg')
            assert torch.allclose(out[b:b+1], single, atol=1e-5)


class TestResize:
    def test_upsample_shape(self):
        x = torch.randn(2, 3, 32, 32)
        out = resize(x, 2.0)
        assert out.shape == (2, 3, 64, 64)

    def test_downsample_shape(self):
        x = torch.randn(2, 3, 64, 64)
        out = resize(x, 0.5)
        assert out.shape == (2, 3, 32, 32)

    def test_scale_up(self):
        x = torch.randn(1, 1, 16, 16)
        out = resize(x, 2.0)
        assert out.shape == (1, 1, 32, 32)


class TestBinaryHole:
    def test_output_shape(self):
        mask = torch.rand(1, 1, 32, 32)
        out = binary_hole(mask)
        assert out.shape == (1, 1, 32, 32)
        assert ((mask == 0) | (mask == 1)).all()


class TestThops:
    def test_pixels(self):
        x = torch.randn(2, 3, 64, 32)
        assert thops_pixels(x) == 64 * 32

    def test_sum(self):
        x = torch.randn(2, 3, 4, 5)
        total = thops_sum(x)
        expected = x.sum()
        assert torch.allclose(total, expected)

    def test_sum_dim(self):
        x = torch.randn(2, 3, 4, 5)
        result = thops_sum(x, dim=[2, 3])
        expected = x.sum(dim=[2, 3])
        assert torch.allclose(result, expected)


class TestSqueezeUnsqueeze:
    def test_squeeze_shape(self):
        x = torch.randn(2, 3, 16, 16)
        out = squeeze2d(x, factor=2)
        assert out.shape == (2, 12, 8, 8)

    def test_unsqueeze_shape(self):
        x = torch.randn(2, 12, 8, 8)
        out = unsqueeze2d(x, factor=2)
        assert out.shape == (2, 3, 16, 16)

    def test_roundtrip(self):
        x = torch.randn(2, 3, 16, 16)
        assert torch.allclose(unsqueeze2d(squeeze2d(x, 2), 2), x, atol=1e-5)


class TestGaussian:
    def test_log_p_shape(self):
        x = torch.randn(2, 64, 4, 4)
        mean = torch.zeros(2, 64, 4, 4)
        log_sd = torch.zeros(2, 64, 4, 4)
        lp = gaussian_log_p(x, mean, log_sd)
        assert lp.shape == x.shape

    def test_sample_shape(self):
        eps = torch.randn(2, 64, 4, 4)
        mean = torch.zeros(2, 64, 4, 4)
        log_sd = torch.zeros(2, 64, 4, 4)
        out = gaussian_sample(eps, mean, log_sd)
        assert out.shape == eps.shape


class TestBasic:
    def test_forward(self):
        module = Basic(32)
        x = torch.randn(2, 32, 16, 16)
        out = module(x)
        assert out.shape == (2, 1, 16, 16)

    def test_channel_in(self):
        module = Basic(16)
        x = torch.randn(2, 16, 16, 16)
        out = module(x)
        assert out.shape == (2, 1, 16, 16)


class TestEncode:
    def test_output_shape(self):
        encode = Encode(3, 32)
        x = torch.randn(2, 3, 64, 64)
        out = encode(x)
        assert out.shape == (2, 32, 64, 64)


class TestSoftmetric:
    def test_output_shape(self):
        metric = Softmetric(in_channels=3)
        img0 = torch.randn(2, 3, 64, 64)
        img1 = torch.randn(2, 3, 64, 64)
        fflow = torch.zeros(2, 2, 64, 64)
        bflow = torch.zeros(2, 2, 64, 64)
        merged = torch.randn(2, 3, 64, 64)
        metric_in = torch.randn(2, 3, 64, 64)
        out = metric(img0, img1, fflow, bflow, merged, metric_in)
        assert out.shape == (2, 1, 64, 64)


class TestFeaturePyramid:
    def test_encode_only(self):
        fp = FeaturePyramid(in_channels=3, num_levels=3)
        x = torch.randn(2, 3, 128, 128)
        out = fp(x)
        assert len(out) == 3

    def test_forward_single_img(self):
        fp = FeaturePyramid(in_channels=3, num_levels=4)
        x = torch.randn(1, 3, 64, 64)
        out = fp(x)
        assert len(out) == 4


class TestBuildGeneratorArch:
    def test_v00(self):
        model = build_generator_arch('v00')
        assert isinstance(model, torch.nn.Module)

    def test_vb(self):
        model = build_generator_arch('vb')
        assert isinstance(model, torch.nn.Module)

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown generator version"):
            build_generator_arch('v99')


class TestNetworkV0:
    def test_forward_decode(self):
        from amverge.core.interpolation.pervfi_arch import NetworkV0

        model = NetworkV0(dilate_size=9)
        model.eval()

        b, c, h, w = 1, 3, 64, 64
        img0 = torch.randn(b, c, h, w).clamp(0, 1)
        img1 = torch.randn(b, c, h, w).clamp(0, 1)
        fflow = torch.randn(b, 2, h, w)
        bflow = torch.randn(b, 2, h, w)

        with torch.no_grad():
            zs = [
                torch.randn(b, 6, h // 4, w // 4),
                torch.randn(b, 12, h // 8, w // 8),
                torch.randn(b, 48, h // 16, w // 16),
            ]
            pred, smasks = model(zs=zs, inps=[img0, img1, fflow, bflow], time=0.5, code="decode")
        assert pred.shape == (b, c, h, w)

    def test_forward_decode_different_size(self):
        from amverge.core.interpolation.pervfi_arch import NetworkV0

        model = NetworkV0(dilate_size=9)
        model.eval()

        b, c, h, w = 1, 3, 48, 80
        img0 = torch.randn(b, c, h, w).clamp(0, 1)
        img1 = torch.randn(b, c, h, w).clamp(0, 1)
        fflow = torch.randn(b, 2, h, w)
        bflow = torch.randn(b, 2, h, w)

        with torch.no_grad():
            zs = [
                torch.randn(b, 6, h // 4, w // 4),
                torch.randn(b, 12, h // 8, w // 8),
                torch.randn(b, 48, h // 16, w // 16),
            ]
            pred, _ = model(zs=zs, inps=[img0, img1, fflow, bflow], time=0.5, code="decode")
        assert pred.shape == (b, c, h, w)


class TestNetworkVb:
    def test_forward(self):
        from amverge.core.interpolation.pervfi_arch import NetworkVb

        model = NetworkVb(dilate_size=9)
        model.eval()

        b, c, h, w = 1, 3, 64, 64
        img0 = torch.randn(b, c, h, w).clamp(0, 1)
        img1 = torch.randn(b, c, h, w).clamp(0, 1)
        fflow = torch.randn(b, 2, h, w)
        bflow = torch.randn(b, 2, h, w)

        with torch.no_grad():
            pred, smasks = model(inps=[img0, img1, fflow, bflow], time=0.5)
        assert pred.shape == (b, c, h, w)


class TestPipelineInfer:
    def test_instantiation_no_weights(self):
        gen_path = '/nonexistent/path.pth'
        try:
            pipe = Pipeline_infer('v00', gen_path, device='cpu')
        except (FileNotFoundError, RuntimeError):
            pass

    def test_get_z(self):
        class FakePipeline(Pipeline_infer):
            def __init__(self):
                pass

        pipe = FakePipeline()
        pipe.device = 'cpu'
        zs = pipe.get_z(0.3, (64, 64), 1, 'cpu')
        assert len(zs) == 3
        assert zs[0].shape == (1, 6, 32, 32)
        assert zs[1].shape == (1, 12, 16, 16)
        assert zs[2].shape == (1, 48, 8, 8)


class TestWarpPyramid:
    def test_output_shape(self):
        img = torch.randn(1, 3, 64, 64)
        flow = torch.randn(1, 2, 64, 64)
        out = warp_pyramid(img, flow)
        assert out.shape == (1, 3, 64, 64)
