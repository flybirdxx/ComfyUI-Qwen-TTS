# ComfyUI-Qwen-TTS

[English](README.md) | 中文版

![节点截图](example/qwen3-tts.png)

基于阿里巴巴 Qwen 团队开源的 **Qwen3-TTS** 项目，为 ComfyUI 实现的语音合成自定义节点。

## 功能特性

- 🎵 **语音合成**: 高质量的文本转语音功能。
- 🎭 **声音克隆**: 支持从短音频示例进行零样本（Zero-shot）声音克隆。
- 🎨 **声音设计**: 支持通过自然语言描述自定义声音特质。
- 🚀 **高效推理**: 支持 12Hz 和 25Hz 的语音 Tokenizer 架构。
- 🎯 **多语言支持**: 原生支持 10 种主要语言（中文、英文、日文、韩文、德文、法文、俄文、葡萄牙文、西班牙文和意大利文）。
- ⚡ **集成加载**: 无需独立的加载器节点；模型加载按需管理，并带有全局缓存。
- ⏱️ **超低延迟**: 基于创新架构，支持极速语音重建与流式生成。

## 节点列表

### 1. Qwen3-TTS 声音设计 (`VoiceDesignNode`)
根据文本描述生成独有的声音。
- **输入**:
  - `text`: 要合成的目标文本。
  - `instruct`: 声音描述指令（例如：“一个温和的高音女声”）。
  - `model_choice`: 目前声音设计功能锁定为 **1.7B** 模型。
- **能力**: 最适合创建“想象中的”声音或特定的人设。

### 2. Qwen3-TTS 声音克隆 (`VoiceCloneNode`)
从参考音频剪辑中克隆声音。
- **输入**:
  - `ref_audio`: 一段短的（5-15秒）参考音频。
  - `ref_text`: 参考音频中的文本内容（有助于提高质量）。
  - `target_text`: 你希望克隆声音说出的新文本。
  - `model_choice`: 可选择 **0.6B**（速度快）或 **1.7B**（质量高）。

### 3. Qwen3-TTS 预设声音 (`CustomVoiceNode`)
使用预设说话人的标准 TTS。
- **输入**:
  - `text`: 目标文本。
  - `speaker`: 从预设声音中选择（Aiden, Eric, Serena 等）。
  - `instruct`: 可选的风格指令。

## 安装

确保已安装以下依赖：
```bash
pip install torch torchaudio transformers librosa accelerate
```

## 最佳实践技巧
- **克隆**: 使用清晰、无背景噪音的参考音频（5-15 秒）。
- **显存**: 使用 `bf16` 精度可以在几乎不损失质量的情况下大幅节省内存。
- **本地模型**: 预先将权重下载到 `models/qwen-tts/` 以优先进行本地加载，避免 HuggingFace 连接超时。

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS): 阿里巴巴 Qwen 团队官方开源仓库。

## 许可证

- 本项目采用 **Apache License 2.0** 许可证。
- 模型权重请参考 [Qwen3-TTS 许可协议](https://github.com/QwenLM/Qwen3-TTS#License)。
