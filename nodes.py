# ComfyUI-Qwen-TTS Node Implementation
# Based on the open-source Qwen3-TTS project by Alibaba Qwen team

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import folder_paths
import types


# Common languages list for UI
DEMO_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]

# Language mapping dictionary to engine codes
LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

# Model family options for UI (0.6B / 1.7B)
MODEL_FAMILIES = ["0.6B", "1.7B"]
# Mapping of family to default HuggingFace repo ID
MODEL_FAMILY_TO_HF = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# Add qwen_tts to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_path = os.path.join(current_dir, 'qwen_tts')
if qwen_tts_path not in sys.path:
    sys.path.insert(0, qwen_tts_path)

try:
    from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
except ImportError as e:
    print(f"❌ Failed to import Qwen3-TTS: {e}")
    print("Please ensure the qwen_tts package is correctly installed.")
    Qwen3TTSModel = None
    VoiceClonePromptItem = None


# Global model cache
_MODEL_CACHE = {}

def load_qwen_model(model_type: str, model_choice: str, device: str, precision: str):
    """Shared model loading logic with caching and local path priority"""
    global _MODEL_CACHE
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set precision
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    # VoiceDesign restriction
    if model_type == "VoiceDesign" and model_choice == "0.6B":
        raise RuntimeError("❌ VoiceDesign only supports 1.7B models!")
        
    # Cache key
    cache_key = (model_type, model_choice, device, precision)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    # Clear old cache
    _MODEL_CACHE.clear()
    
    # --- 1. Determine search directories ---
    base_paths = []
    try:
        # Resolve ComfyUI root
        import folder_paths
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        qwen_tts_dir = os.path.join(comfy_root, "models", "qwen-tts")
        if os.path.exists(qwen_tts_dir):
            base_paths.append(qwen_tts_dir)
        else:
            # Compatibility check: models/qwen-tts in parent dir
            alt_root = os.path.dirname(comfy_root)
            alt_qwen_tts_dir = os.path.join(alt_root, "models", "qwen-tts")
            if os.path.exists(alt_qwen_tts_dir):
                base_paths.append(alt_qwen_tts_dir)
    except Exception:
        pass
    
    # Check registered TTS paths in folder_paths
    try:
        registered_tts = folder_paths.get_folder_paths("TTS") or []
        for p in registered_tts:
            if p not in base_paths: base_paths.append(p)
    except Exception: pass

    # --- 2. Search for matching models ---
    HF_MODEL_MAP = {
        ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }
    
    final_source = HF_MODEL_MAP.get((model_type, model_choice)) or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    found_local = None
    
    for base in base_paths:
        try:
            if not os.path.isdir(base): continue
            subdirs = os.listdir(base)
            for d in subdirs:
                cand = os.path.join(base, d)
                if os.path.isdir(cand):
                    # Match logic: contains model size and type keyword
                    if model_choice in d and model_type.lower() in d.lower():
                        found_local = cand
                        break
            if found_local: break
        except Exception: pass
    
    if found_local:
        final_source = found_local
        print(f"✅ [Qwen3-TTS] Loading local model: {os.path.basename(final_source)}")
    else:
        print(f"🌐 [Qwen3-TTS] Loading remote model: {final_source}")

    model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype, attn_implementation="flash_attention_2")
    
    _MODEL_CACHE[cache_key] = model
    return model



    
    


class VoiceDesignNode:
    """
    VoiceDesign Node: Generate custom voices based on text descriptions.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, I am a custom voice created by description.", "placeholder": "Enter text to synthesize"}),
                "instruct": ("STRING", {"multiline": True, "default": "A cute girl voice with a high pitch and expressive tone.", "placeholder": "Enter voice description"}),
                "model_choice": (["1.7B"], {"default": "1.7B", "tooltip": "VoiceDesign only supports 1.7B models"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceDesign: Generate custom voices from descriptions."

    def generate(self, text: str, instruct: str, model_choice: str, device: str, precision: str, language: str, seed: int = 0, max_new_tokens: int = 2048) -> Tuple[Dict[str, Any]]:
        if not text or not instruct:
            raise RuntimeError("Text and instruction description are required")

        # Load model
        model = load_qwen_model("VoiceDesign", model_choice, device, precision)

        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed % (2**32))

        # Perform generation
        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        wavs, sr = model.generate_voice_design(
            text=text,
            language=mapped_lang,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}
            return (audio_data,)
        raise RuntimeError("Invalid audio data generated")


class VoiceCloneNode:
    """
    VoiceClone (Base) Node: Create clones from reference audio and synthesize target text.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional)"}),
                "target_text": ("STRING", {"multiline": True, "default": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceClone: Clone voice from reference audio."

    def _audio_tensor_to_tuple(self, audio_tensor: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        # Accept multiple possible ComfyUI audio formats
        waveform = None
        sr = None
        try:
            if isinstance(audio_tensor, dict):
                # Common keys: 'waveform'/'sample_rate' or 'data'/'sampling_rate'
                if "waveform" in audio_tensor:
                    waveform = audio_tensor.get("waveform")
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr") or audio_tensor.get("sampling_rate")
                elif "data" in audio_tensor and "sampling_rate" in audio_tensor:
                    waveform = np.asarray(audio_tensor.get("data"))
                    sr = audio_tensor.get("sampling_rate")
                elif "audio" in audio_tensor and isinstance(audio_tensor["audio"], (tuple, list)):
                    # maybe {'audio': (sr, data)}
                    a = audio_tensor["audio"]
                    if isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], (int, float)):
                        sr, waveform = int(a[0]), np.asarray(a[1])
                else:
                    # fallback: try common numeric fields
                    for k in ("samples", "y", "wave"):
                        if k in audio_tensor:
                            waveform = np.asarray(audio_tensor.get(k))
                            break
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr") or audio_tensor.get("sampling_rate") or audio_tensor.get("rate")
            elif isinstance(audio_tensor, tuple) and len(audio_tensor) == 2:
                # (waveform, sr) or (sr, waveform)
                a0, a1 = audio_tensor
                if isinstance(a0, (int, float)) and isinstance(a1, (list, np.ndarray, torch.Tensor)):
                    sr = int(a0); waveform = np.asarray(a1)
                elif isinstance(a1, (int, float)) and isinstance(a0, (list, np.ndarray, torch.Tensor)):
                    sr = int(a1); waveform = np.asarray(a0)
            elif isinstance(audio_tensor, list):
                # maybe [waveform, sr]
                if len(audio_tensor) == 2 and isinstance(audio_tensor[0], (list, np.ndarray)) and isinstance(audio_tensor[1], (int, float)):
                    waveform = np.asarray(audio_tensor[0]); sr = int(audio_tensor[1])
        except Exception:
            pass
        # Normalize to 1-D numpy float32 array (model expects 1-D waveforms)
        if isinstance(waveform, torch.Tensor):
            # ComfyUI audio is often [batch, channels, samples] or [channels, samples]
            if waveform.dim() > 1:
                # Squeeze out any unit dimensions (like batch=1, channel=1)
                waveform = waveform.squeeze()
                # If still multi-dimensional (e.g. stereo), average to mono
                if waveform.dim() > 1:
                    waveform = torch.mean(waveform, dim=0)
            waveform = waveform.cpu().numpy()

        if isinstance(waveform, np.ndarray):
            # Double check for numpy version of same logic
            if waveform.ndim > 1:
                waveform = np.squeeze(waveform)
                if waveform.ndim > 1:
                    # Heuristic: the smaller dimension is likely channels
                    if waveform.shape[0] < waveform.shape[1]:
                        waveform = np.mean(waveform, axis=0)
                    else:
                        waveform = np.mean(waveform, axis=1)
            waveform = waveform.astype(np.float32)

        # Final safety flatten to ensure it's 1-D
        if waveform is not None and waveform.ndim > 1:
            waveform = waveform.flatten()

        # Basic validation
        if waveform is None or not isinstance(waveform, np.ndarray) or waveform.size == 0:
            raise RuntimeError("Failed to parse reference audio waveform")
        
        min_samples = 1024
        if waveform.size < min_samples:
            # Pad with zeros to avoid upstream padding errors
            pad_to = min_samples
            pad_amount = pad_to - waveform.size
            waveform = np.concatenate([waveform, np.zeros(pad_amount, dtype=np.float32)])

        # Return as tuple (waveform, sr) with 1-D numpy waveform as expected by the tokenizer
        return (waveform, int(sr))

    def generate(self, ref_audio: Dict[str, Any], ref_text: str, target_text: str, model_choice: str, device: str, precision: str, language: str, seed: int = 0, x_vector_only: bool = False, max_new_tokens: int = 2048) -> Tuple[Dict[str, Any]]:
        if ref_audio is None:
            raise RuntimeError("Reference audio is required")
        
        # Load model
        model = load_qwen_model("Base", model_choice, device, precision)

        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed % (2**32))
        
        audio_tuple = None
        if isinstance(ref_audio, dict):
            audio_tuple = self._audio_tensor_to_tuple(ref_audio)
        else:
            raise RuntimeError("Unsupported reference audio format")

        # Monkeypatch model._normalize_audio_inputs to return mutable lists to avoid upstream tuple assignment bug
        orig_normalize = getattr(model, "_normalize_audio_inputs", None)
        def _safe_normalize(self, audios):
            # Adapted from upstream but returns list entries as lists [waveform, sr]
            if isinstance(audios, list):
                items = audios
            else:
                items = [audios]

            out = []
            for a in items:
                if isinstance(a, str):
                    wav, sr = self._load_audio_to_np(a)
                    out.append([wav.astype(np.float32), int(sr)])
                elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                    out.append([a[0].astype(np.float32), int(a[1])])
                elif isinstance(a, np.ndarray):
                    raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
                else:
                    raise TypeError(f"Unsupported audio input type: {type(a)}")

            # ensure mono
            for i in range(len(out)):
                wav, sr = out[i][0], out[i][1]
                if wav.ndim > 1:
                    out[i][0] = np.mean(wav, axis=-1).astype(np.float32)
            return out

        if orig_normalize is not None:
            try:
                model._normalize_audio_inputs = types.MethodType(_safe_normalize, model)
            except Exception:
                pass

        try:
            mapped_lang = LANGUAGE_MAP.get(language, "auto")
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                ref_audio=audio_tuple,
                ref_text=ref_text if ref_text and ref_text.strip() else None,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=max_new_tokens,
            )
        finally:
            # restore original
            if orig_normalize is not None:
                try:
                    model._normalize_audio_inputs = orig_normalize
                except Exception:
                    pass

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            # Ensure waveform is 1D [samples]
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            # Convert to ComfyUI format: [batch, channels, samples]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}
            return (audio_data,)
        


class CustomVoiceNode:
    """
    CustomVoice (TTS) Node: Generate text-to-speech using preset speakers.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello world", "placeholder": "Enter text to synthesize"}),
                "speaker": (["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"], {"default": "Ryan"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (optional)"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "CustomVoice: Generate speech using preset speakers."

    def generate(self, text: str, speaker: str, model_choice: str, device: str, precision: str, language: str, seed: int = 0, instruct: str = "", max_new_tokens: int = 2048) -> Tuple[Dict[str, Any]]:
        if not text or not speaker:
            raise RuntimeError("Text and speaker are required")
        
        # Load model
        model = load_qwen_model("CustomVoice", model_choice, device, precision)

        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed % (2**32))

        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=mapped_lang,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct if instruct and instruct.strip() else None,
            max_new_tokens=max_new_tokens,
        )

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            # Ensure waveform is 1D [samples]
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            # Convert to ComfyUI format: [batch, channels, samples]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}
            return (audio_data,)
        raise RuntimeError("Invalid audio data generated")
