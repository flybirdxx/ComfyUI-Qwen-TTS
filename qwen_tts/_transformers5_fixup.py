"""Runtime transformers-5 fixups applied on package import.

On transformers >= 5 the vendored Qwen3TTSTalkerRotaryEmbedding initializes
on the meta device, producing an all-zero inv_freq whose original_inv_freq
stays a meta reference; every forward then re-applies it and cos/sin go to
zero -> generation diverges (no EOS, runaway loop). Recompute both buffers on
the real device after from_pretrained. No-op on transformers < 5.
"""
from __future__ import annotations


def install() -> None:
    import sys
    import transformers

    if transformers.__version__.startswith("4."):
        return  # v4 path is correct already

    from .core.models import modeling_qwen3_tts as modeling_module
    from .inference.qwen3_tts_model import Qwen3TTSModel

    orig = Qwen3TTSModel.from_pretrained.__func__

    def from_pretrained_fixed(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = orig(cls, pretrained_model_name_or_path, *args, **kwargs)
        try:
            fn = modeling_module.ROPE_INIT_FUNCTIONS["default"]
            device = next(model.model.talker.model.parameters()).device
            # Talker rotary (Qwen3TTSTalkerRotaryEmbedding).
            rot = model.model.talker.model.rotary_emb
            inv_freq, attention_scaling = fn(rot.config, device)
            rot.inv_freq = inv_freq
            rot.original_inv_freq = inv_freq.clone()
            rot.attention_scaling = attention_scaling
        except Exception:
            pass
        try:
            # Code-predictor (subtalker) rotary (Qwen3TTSRotaryEmbedding) is a
            # separate instance with the SAME meta-device problem; unfixed it
            # made the subtalker logits wrong under transformers 5.
            fn = modeling_module.ROPE_INIT_FUNCTIONS["default"]
            rot2 = model.model.talker.code_predictor.model.rotary_emb
            inv_freq2, scaling2 = fn(rot2.config, device)
            rot2.inv_freq = inv_freq2
            rot2.original_inv_freq = inv_freq2.clone()
            rot2.attention_scaling = scaling2
        except Exception:
            pass
        return model

    Qwen3TTSModel.from_pretrained = classmethod(from_pretrained_fixed)
