#!/usr/bin/env python3
"""Apply transformers-5 compatibility shims to ComfyUI-Qwen-TTS's vendored qwen_tts.

All shims are CONDITIONAL: on transformers < 5 they are no-ops, so the
transformers 4.57.3 path is unchanged. On transformers >= 5 they repair the
v4-era vendored code that transformers reworked.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path.cwd()

MODELING_FILES = [
    REPO / "qwen_tts" / "core" / "models" / "modeling_qwen3_tts.py",
    REPO / "qwen_tts" / "core" / "tokenizer_12hz" / "modeling_qwen3_tts_tokenizer_v2.py",
]

ROPE_SHIM = """
# --- transformers >=5 compat (patched locally) -------------------------------
# transformers 5 removed the "default" entry from ROPE_INIT_FUNCTIONS; the
# lookups below raise KeyError: 'default' for configs without rope_scaling.
# Restore the v4 default under a rebound local name. No-op on transformers <5.
if "default" not in ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):  # noqa: E501
        import torch as _torch
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            _torch.tensor(base, dtype=_torch.int64)
            ** (
                _torch.arange(0, int(head_dim * partial_rotary_factor), 2, dtype=_torch.int64)
                .float()
                .to(device)
                / head_dim
            )
        )
        return inv_freq, attention_factor

    ROPE_INIT_FUNCTIONS = {**ROPE_INIT_FUNCTIONS, "default": _compute_default_rope_parameters}
# --- end shim ---------------------------------------------------------------
"""

MASK_SHIM = """
# --- transformers >=5 compat (patched locally) -------------------------------
# transformers 5 renamed create_causal_mask argument `input_embeds` to
# `inputs_embeds` and dropped `cache_position`; the vendored calls use the v4
# names. Rebind thin adapters. No-op on transformers <5.
if not getattr(create_causal_mask, "_v5_shim", False):
    _create_causal_mask_v4 = create_causal_mask
    _create_sliding_v4 = create_sliding_window_causal_mask

    def _adapt_mask_kwargs(kwargs):
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if "cache_position" in kwargs and "position_ids" not in kwargs:
            kwargs["position_ids"] = kwargs.pop("cache_position").unsqueeze(0)
        else:
            kwargs.pop("cache_position", None)
        return kwargs

    def create_causal_mask(**kwargs):  # noqa: F811
        return _create_causal_mask_v4(**_adapt_mask_kwargs(kwargs))
    create_causal_mask._v5_shim = True

    def create_sliding_window_causal_mask(**kwargs):  # noqa: F811
        return _create_sliding_v4(**_adapt_mask_kwargs(kwargs))
    create_sliding_window_causal_mask._v5_shim = True
# --- end shim ---------------------------------------------------------------
"""

POSITION_SLICE = """        # transformers >=5 compat (patched locally): generate grows position_ids
        # across decode steps, so a cached step receives the FULL history while
        # inputs_embeds carries only the current token. Slice to the current
        # step or the rotary broadcast inflates the attention output.
        if (
            position_ids is not None
            and inputs_embeds is not None
            and position_ids.shape[-1] != inputs_embeds.shape[1]
        ):
            position_ids = position_ids[..., -inputs_embeds.shape[1] :]

"""


def insert_after_import(s: str, shim: str, import_line: str) -> str:
    """Insert `shim` after the line matching `import_line`."""
    assert import_line in s, f"import line not found: {import_line[:60]}"
    i = s.index(import_line)
    eol = s.index("\n", i)
    return s[: eol + 1] + shim + s[eol + 1 :]


def insert_after_paren_import(s: str, shim: str, start_line: str) -> str:
    """Insert `shim` after the closing paren of a parenthesized import."""
    assert start_line in s, f"paren import start not found: {start_line[:60]}"
    i = s.index(start_line)
    close = s.index(")", i)
    eol = s.index("\n", close)
    return s[: eol + 1] + shim + s[eol + 1 :]


def apply_to_modeling(path: Path) -> bool:
    s = path.read_text()
    if "transformers >=5 compat" in s:
        return False

    # ROPE shim: after the single-line rope import.
    s = insert_after_import(s, ROPE_SHIM, "from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update")

    # MASK shim: after the parenthesized masking import.
    s = insert_after_paren_import(s, MASK_SHIM, "from transformers.masking_utils import (")

    # POSITION slice.
    anchor1 = "        # the hard coded `3` is for temporal, height and width.\n"
    if anchor1 in s:
        s = s.replace(anchor1, POSITION_SLICE + anchor1, 1)
    else:
        anchor2 = "        if position_ids is None:\n            position_ids = cache_position.unsqueeze(0)\n"
        assert anchor2 in s, f"{path.name}: no position anchor"
        s = s.replace(anchor2, POSITION_SLICE + anchor2, 1)

    path.write_text(s)
    return True


def patch_config(path: Path) -> bool:
    s = path.read_text()
    if "pad_token_id=None,  # transformers" in s:
        return False
    a = "        codec_eos_token_id=4198,\n        codec_think_id=4202,"
    assert a in s, "config codec_eos anchor"
    s = s.replace(a, "        codec_eos_token_id=4198,\n        pad_token_id=None,  # transformers >=5 compat; defaults to codec_eos\n        codec_think_id=4202,", 1)
    a = "        self.codec_eos_token_id = codec_eos_token_id\n        self.codec_think_id = codec_think_id"
    assert a in s, "config assign anchor"
    s = s.replace(a, "        self.codec_eos_token_id = codec_eos_token_id\n        self.pad_token_id = pad_token_id if pad_token_id is not None else codec_eos_token_id\n        self.codec_think_id = codec_think_id", 1)
    path.write_text(s)
    return True


def patch_tokenizer_decorator(path: Path) -> bool:
    s = path.read_text()
    a = "from transformers.utils.generic import check_model_inputs"
    if a in s:
        s = s.replace(a, "# " + a + "  # dead on transformers >=5 (needs a func arg); unused", 1)
    d = "    @check_model_inputs()"
    if d in s:
        s = s.replace(d, "    # " + d.strip() + "  # dead decorator, unused", 1)
    path.write_text(s)
    return True


def main() -> None:
    for f in MODELING_FILES:
        print(f"{f.name}:", "patched" if apply_to_modeling(f) else "already patched")
        if f.name == "modeling_qwen3_tts_tokenizer_v2.py":
            print(f"{f.name}: decorator", "patched" if patch_tokenizer_decorator(f) else "already ok")
    cfg = REPO / "qwen_tts" / "core" / "models" / "configuration_qwen3_tts.py"
    print(f"{cfg.name}:", "patched" if patch_config(cfg) else "already patched")


if __name__ == "__main__":
    main()
