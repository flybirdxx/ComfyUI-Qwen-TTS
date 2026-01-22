# coding=utf-8
# Qwen3-TTS Gradio Demo for HuggingFace Spaces with Zero GPU
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)
import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)
import os
import spaces
import gradio as gr
import numpy as np
import torch

# Global model holders - keyed by (model_type, model_size)
loaded_models = {}

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return os.path.join("Qwen", f"Qwen3-TTS-12Hz-{model_size}-{model_type}")


def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = get_model_path(model_type, model_size)
        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
#           attn_implementation="flash_attention_2",
        )
    return loaded_models[key]


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["自动", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


@spaces.GPU(duration=120)
def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "错误：需要文本。"
    if not voice_description or not voice_description.strip():
        return None, "错误：需要语音描述。"

    try:
        tts = get_model("VoiceDesign", "1.7B")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "语音设计生成成功完成！"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


@spaces.GPU(duration=180)
def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "错误：需要目标文本。"

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "错误：需要参考音频。"

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "错误：当未启用'仅使用x-vector'时，需要参考文本。"

    try:
        tts = get_model("Base", model_size)
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "语音克隆生成成功完成！"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


@spaces.GPU(duration=120)
def generate_custom_voice(text, language, speaker, instruct, model_size):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "错误：需要文本。"
    if not speaker:
        return None, "错误：需要说话人。"

    try:
        tts = get_model("CustomVoice", model_size)
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "生成成功完成！"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS 演示") as demo:
        gr.Markdown(
            """
# Qwen3-TTS 演示

统一的文本到语音演示，提供三种强大的模式：
- **语音设计**：使用自然语言描述创建自定义语音
- **语音克隆（基础）**：从参考音频克隆任何语音
- **TTS（自定义语音）**：使用预定义说话人和可选风格指令生成语音

由阿里巴巴Qwen团队基于[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)构建。
"""
        )

        with gr.Tabs():
            # Tab 1: Voice Design (Default, 1.7B only)
            with gr.Tab("语音设计"):
                gr.Markdown("### 使用自然语言创建自定义语音")
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="待合成的文本",
                            lines=4,
                            placeholder="输入您要转换为语音的文本...",
                            value="它在最上面的抽屉里...等等，它是空的？不可能，这不可能！我确定我把它放在那里了！"
                        )
                        design_language = gr.Dropdown(
                            label="语言",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                        design_instruct = gr.Textbox(
                            label="语音描述",
                            lines=3,
                            placeholder="描述您想要的语音特征...",
                            value="以怀疑的语气说话，但带有逐渐浮现的恐慌感。"
                        )
                        design_btn = gr.Button("使用自定义语音生成", variant="primary")

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="生成的音频", type="numpy")
                        design_status = gr.Textbox(label="状态", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone (Base)
            with gr.Tab("语音克隆（基础）"):
                gr.Markdown("### 从参考音频克隆语音")
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="参考音频（上传要克隆的语音样本）",
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="参考文本（参考音频的转录文本）",
                            lines=2,
                            placeholder="输入参考音频中确切说出的文本...",
                        )
                        clone_xvector = gr.Checkbox(
                            label="仅使用x-vector（不需要参考文本，但质量较低）",
                            value=False,
                        )

                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="目标文本（使用克隆语音合成的文本）",
                            lines=4,
                            placeholder="输入您想要克隆语音说的文本...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="语言",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            clone_model_size = gr.Dropdown(
                                label="模型大小",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        clone_btn = gr.Button("克隆并生成", variant="primary")

                with gr.Row():
                    clone_audio_out = gr.Audio(label="生成的音频", type="numpy")
                    clone_status = gr.Textbox(label="状态", lines=2, interactive=False)

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size],
                    outputs=[clone_audio_out, clone_status],
                )

            # Tab 3: TTS (CustomVoice)
            with gr.Tab("TTS（自定义语音）"):
                gr.Markdown("### 使用预定义说话人的文本到语音")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="待合成的文本",
                            lines=4,
                            placeholder="输入您要转换为语音的文本...",
                            value="您好！欢迎使用文本到语音系统。这是我们TTS能力的演示。"
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="语言",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                            tts_speaker = gr.Dropdown(
                                label="说话人",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                        with gr.Row():
                            tts_instruct = gr.Textbox(
                                label="风格指令（可选）",
                                lines=2,
                                placeholder="例如：以欢快和充满活力的语气说话",
                            )
                            tts_model_size = gr.Dropdown(
                                label="模型大小",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        tts_btn = gr.Button("生成语音", variant="primary")

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="生成的音频", type="numpy")
                        tts_status = gr.Textbox(label="状态", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size],
                    outputs=[tts_audio_out, tts_status],
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=2).launch()
