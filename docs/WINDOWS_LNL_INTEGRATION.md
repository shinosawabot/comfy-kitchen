# Windows Lunar Lake integration

The provider wheel accepts `--xpu-target lnl` for the experimental Windows
Lunar Lake profile. It must be paired with the `lnl`
`omni_xpu_kernel` companion build.

Kitchen keeps the existing oneDNN int4 path for BMG, PTL-H, and DG2. On LNL,
SVDQuant uses Omni dequantization followed by the reference matrix multiply
when the native oneDNN int4 route is unavailable. This preserves the functional
contract while the native LNL GEMM path is validated separately.

The target is experimental until a target-local wheel hash and ComfyUI
workflow receipt are recorded.
