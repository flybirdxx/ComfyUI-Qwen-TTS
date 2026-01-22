# ComfyUI-Qwen-TTS

English | [中文版](README_CN.md)

![Nodes Screenshot](example/qwen3-tts.png)

ComfyUI custom nodes for speech synthesis, voice cloning, and voice design, based on the open-source **Qwen3-TTS** project by the Alibaba Qwen team.

## Key Features

- 🎵 **Speech Synthesis**: High-quality text-to-speech conversion.
- 🎭 **Voice Cloning**: Zero-shot voice cloning from short reference audio.
- 🎨 **Voice Design**: Create custom voice characteristics based on natural language descriptions.
- 🚀 **Efficient Inference**: Supports both 12Hz and 25Hz speech tokenizer architectures.
- 🎯 **Multilingual**: Native support for 10 languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian).
- ⚡ **Integrated Loading**: No separate loader nodes required; model loading is managed on-demand with global caching.
- ⏱️ **Ultra-Low Latency**: Supports high-fidelity speech reconstruction with low-latency streaming.

## Nodes List

### 1. Qwen3-TTS Voice Design (`VoiceDesignNode`)
Generate unique voices based on text descriptions.
- **Inputs**:
  - `text`: Target text to synthesize.
  - `instruct`: Description of the voice (e.g., "A gentle female voice with a high pitch").
  - `model_choice`: Currently locked to **1.7B** for VoiceDesign features.
- **Capabilities**: Best for creating "imaginary" voices or specific character archetypes.

### 2. Qwen3-TTS Voice Clone (`VoiceCloneNode`)
Clone a voice from a reference audio clip.
- **Inputs**:
  - `ref_audio`: A short (5-15s) audio clip to clone.
  - `ref_text`: Text spoken in the `ref_audio` (helps improve quality).
  - `target_text`: The new text you want the cloned voice to say.
  - `model_choice`: Choose between **0.6B** (fast) or **1.7B** (high quality).

### 3. Qwen3-TTS Custom Voice (`CustomVoiceNode`)
Standard TTS using preset speakers.
- **Inputs**:
  - `text`: Target text.
  - `speaker`: Selection from preset voices (Aiden, Eric, Serena, etc.).
  - `instruct`: Optional style instructions.


## Installation

Ensure you have the required dependencies:
```bash
pip install torch torchaudio transformers librosa accelerate
```

## Tips for Best Results
- **Cloning**: Use clean, noise-free reference audio (5-15 seconds).
- **VRAM**: Use `bf16` precision to save significant memory with minimal quality loss.
- **Local Models**: Pre-download weights to `models/qwen-tts/` to prioritize local loading and avoid HuggingFace timeouts.

## Acknowledgments

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): Official open-source repository by Alibaba Qwen team.

## License

- This project is licensed under the **Apache License 2.0**.
- Model weights are subject to the [Qwen3-TTS License Agreement](https://github.com/QwenLM/Qwen3-TTS#License).
