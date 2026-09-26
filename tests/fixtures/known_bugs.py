from __future__ import annotations

import pytest

TRANSNET_MPS = (
    "TransNetV2 picks the MPS device on Apple Silicon, where aten::avg_pool3d is not "
    "implemented; transnetv2_pytorch sets PYTORCH_ENABLE_MPS_FALLBACK on import, which is "
    "too late if the caller imported torch first, so library users who do that crash"
)


def known_bug(reason: str):
    return pytest.mark.xfail(reason=reason, strict=True)
