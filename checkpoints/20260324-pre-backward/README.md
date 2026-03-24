This checkpoint captures the Qwen3-VL Triton attention codebase state immediately before the backward-path work.

Included:
- `evaluation_wrapper.py`
- `triton_qwen3vl/`
- `test/test_qwen3vl_triton_patch.py`

Notes:
- Files changed by the backward implementation are stored here in their pre-backward form.
- Files not changed by the backward implementation can be copied directly from the current workspace to this snapshot.
