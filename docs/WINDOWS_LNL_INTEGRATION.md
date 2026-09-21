# Windows Lunar Lake integration

The provider wheel accepts `--xpu-target lnl` for the experimental Windows
Lunar Lake profile. It must be paired with the `lnl`
`omni_xpu_kernel` companion build.

Kitchen uses the same SVDQuant API path on LNL as on the other XPU targets.
The `lnl` metadata selects the matching companion build and provider admission;
it is not a runtime operator probe or a fallback selector. Native oneDNN int4
availability must be established by the matching kernel build and its tests.

The target is experimental until a target-local wheel hash and ComfyUI
workflow receipt are recorded.
