import sys
import subprocess
from pathlib import Path

import pytest
import torch

import comfy_kitchen as ck


def test_xpu_wheel_exposes_sparse_api_without_cuda_import():
    assert callable(ck.sol_attn)
    assert callable(ck.sol_attn_chunked)
    assert ck.sol_attn_is_available(torch.device("xpu")) is False
    # Other upstream tests may explicitly import their CUDA backend. Check the
    # public import in isolation so collection order does not affect the result.
    subprocess.run(
        [sys.executable, "-B", "-c", "import sys, comfy_kitchen; "
         "assert 'comfy_kitchen.backends.cuda' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_chunked_unavailable_does_not_consume_projection_chunks():
    def chunks():
        raise AssertionError("unsupported chunked attention must not run QKV projection")

    with pytest.raises(NotImplementedError, match="sol_attn_chunked.*not yet supported on XPU"):
        ck.sol_attn_chunked(
            chunks, 12288, 1, None, (None, None), tau=1.3, token_aug=256,
        )
