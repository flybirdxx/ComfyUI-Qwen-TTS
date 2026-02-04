# Update Log

## 2026-02-04
- **Global Pause Control**: Introduced `QwenTTSConfigNode` to finely control silence duration after punctuation.
  - Supports separate settings for Linebreaks, Periods (.), Commas (,), Question marks (?), and Hyphens (-).
  - Supported nodes: `VoiceCloneNode`, `CustomVoiceNode`, `VoiceDesignNode`, `DialogueInferenceNode`.
- **Path Configuration**: Added support for ComfyUI's `extra_model_paths.yaml`.
  - You can now define a custom model directory using the `qwen-tts` key.

## 2026-01-29
- **Fine-tuning Support**: Added support for loading custom fine-tuned models in `VoiceCloneNode` and `CustomVoiceNode`.
  - Added `custom_model_path` input: specific absolute path to model folder.
  - Added `custom_speaker_name` input to `CustomVoiceNode`: allows calling specific speaker IDs from fine-tuned models.

![Custom Model Loading UI](/doc/example3.png)

> **⚠️ Note / 说明**:
> Currently, the fine-tuning feature is experimental. Official Qwen3-TTS only supports fine-tuning the Base model, and the results are often inferior to direct zero-shot cloning with `VoiceCloneNode`. For experimentation only.
>
> 目前微调功能仅为占位/实验性支持。官方目前仅支持微调 Base 模型，实际效果往往不如直接使用 `VoiceCloneNode` 进行零样本克隆。仅供感兴趣的用户尝试。

## 2026-01-27
- **Optimization**: Streamlined the `LoadSpeaker` node and removed redundant parameters.
- **Compatibility**: Fixed loading errors for local `.qvp` files when using PyTorch 2.6+.
- **Security**: Added feature dimension validation (0.6B/1.7B) to prevent crashes from model mismatches.

## 2026-01-26
- **Persistence Upgrade**: Introduced JSON metadata management to replace old hidden text storage, making voice libraries more transparent and editable.
- **New Nodes**: Officially added `SaveVoice` and `LoadSpeaker` persistence nodes.

![新增节点截图](/doc/example2.png)

## 2026-01-24

### Added Generation Parameters
- Added `top_p`, `top_k`, `temperature`, and `repetition_penalty` parameters to all TTS nodes
- These parameters allow fine-tuning of speech generation quality and diversity
- Available in: VoiceDesignNode, VoiceCloneNode, CustomVoiceNode, DialogueInferenceNode

**Parameter Details:**
- `top_p` (0.0-1.0, default 0.8): Nucleus sampling probability
- `top_k` (0-100, default 20): Top-k sampling parameter
- `temperature` (0.1-2.0, default 1.0): Sampling temperature (higher = more random)
- `repetition_penalty` (1.0-2.0, default 1.05): Penalty to reduce repeated tokens

### Added Attention Mechanism Selection
- Added `attention` parameter dropdown to all TTS nodes with options: auto, sage_attn, flash_attn, sdpa, eager
- Implements automatic detection of best available attention mechanism
- Provides graceful fallback when requested attention is unavailable
- Console logging shows which attention mechanism is being used

**Available Attention Mechanisms:**
- `sage_attn`: SAGE attention (fastest, requires installation)
- `flash_attn`: Flash Attention 2 (fast, requires installation)
- `sdpa`: Scaled Dot Product Attention (PyTorch built-in, medium speed)
- `eager`: Standard attention (always available, slowest)
- `auto`: Automatically selects best available option (recommended)

**Auto-Detection Priority:**
When `attention: "auto"` is selected, checks in this order:
1. sage_attn → if installed
2. flash_attn → if installed
3. sdpa → always available
4. eager → always available (fallback)

**Graceful Fallback:**
If requested attention is unavailable, falls back to sdpa → eager with console warning.

**Model Caching:**
- Cache key includes attention implementation to prevent cross-contamination
- Changing attention mechanism automatically clears cache and reloads model
- Same model with different attention mechanisms coexists in cache

### Added Model Memory Management
- Added `unload_model_after_generate` toggle to all TTS nodes
- Enables model cache clearing and GPU memory freeing after generation
- Clears model cache (`_MODEL_CACHE.clear()`)
- Empties GPU memory (`torch.cuda.empty_cache()`)
- Synchronizes CUDA operations (`torch.cuda.synchronize()`)
- Runs garbage collection (`gc.collect()`)

**Use Cases:**
- Users with limited VRAM (< 8GB) to free memory after generation
- Running multiple different models sequentially
- Freeing memory after generation completes

**Bug Fix:**
- Fixed missing `_MODEL_CACHE` initialization at module level
- Added `_MODEL_CACHE = {}` after line 61 in `nodes.py`

### Bug Fixes
- Fixed `check_model_inputs()` TypeError caused by decorator usage in `transformers==4.57.0`
- Removed parentheses from `@check_model_inputs` decorator in `modeling_qwen3_tts_tokenizer_v2.py`

### Assets & Documentation
- Added `example/example.png` - comprehensive node screenshot
- Added `example/example.json` - example workflow
- Added `example/Multi-character dialogue.json` - multi-role dialogue workflow
- Updated README with new nodes: VoiceClonePromptNode and DialogueInferenceNode

### Repository Maintenance
- Deleted `dev` branch (merged to main)
- Version bumped to 1.0.2

---

## 2026-01-23

### Dependency Compatibility & Mac Support
- **Fixed**: Resolved `transformers` version conflicts with `qwen-tts` dependency
- **Improvement**: Now supports local package import without requiring `pip install qwen-tts`
- **New**: Add MPS (Mac Apple Silicon) support for device detection
- **Note**: The official `qwen-tts` package requires `transformers==4.57.3`, which may conflict with other ComfyUI nodes. This version uses bundled local code to avoid dependency issues.

### New Nodes
- **VoiceClonePromptNode**: Extract and reuse voice features from reference audio
  - Enables "once extracted, multiple uses" workflow
  - Improves efficiency for batch generation with same voice
