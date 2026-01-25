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

# All required models for batch download
ALL_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
]

_MODELS_CHECKED = False
_MODEL_CACHE = {}

# Handle qwen_tts package import
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_dir = os.path.join(current_dir, "qwen_tts")

# CRITICAL: Add current_dir to sys.path FIRST so Python can resolve 'qwen_tts' as a package
# This allows qwen_tts internal files to use relative imports like 'from ..core.models import ...'
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also ensure the qwen_tts folder itself is importable
if qwen_tts_dir not in sys.path:
    sys.path.insert(0, qwen_tts_dir)

try:
    # 1. Try absolute import first (if user installed via pip)
    import qwen_tts
    Qwen3TTSModel = qwen_tts.Qwen3TTSModel
    VoiceClonePromptItem = qwen_tts.VoiceClonePromptItem
except ImportError:
    try:
        # 2. Fallback to local package import (relative or absolute via sys.path)
        from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    except ImportError as e:
        import traceback
        print(f"\n❌ [Qwen3-TTS] Critical Import Error: {e}")
        if not os.path.exists(qwen_tts_dir):
            print(f"   Missing directory: {qwen_tts_dir}")
            print("   Please clone the repository with submodules or ensure 'qwen_tts' folder exists.")
        else:
            print("   Traceback for debugging:")
            traceback.print_exc()
            print("\n   Common fix: run 'pip install -r requirements.txt' in your ComfyUI environment.")
        
        Qwen3TTSModel = None
        VoiceClonePromptItem = None


ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]

def check_attention_implementation():
    """Check available attention implementations and return in priority order."""
    available = []
    
    try:
        from sageattention import sageattn
        available.append("sage_attn")
        print(f"✅ [Qwen3-TTS] sage_attn available (sageattention package)")
    except ImportError:
        pass
    
    try:
        import flash_attn
        available.append("flash_attn")
        print(f"✅ [Qwen3-TTS] flash_attn available")
    except ImportError:
        pass
    
    available.append("sdpa")
    print(f"✅ [Qwen3-TTS] sdpa available (PyTorch built-in)")
    
    available.append("eager")
    print(f"✅ [Qwen3-TTS] eager attention available (always available)")
    
    return available

def get_attention_implementation(selection: str) -> str:
    """Get the actual attention implementation based on selection and availability."""
    available = check_attention_implementation()
    
    if selection == "auto":
        priority = ["sage_attn", "flash_attn", "sdpa", "eager"]
        for attn in priority:
            if attn in available:
                print(f"🔍 [Qwen3-TTS] Auto-selected attention: {attn}")
                return attn
        return "eager"
    else:
        if selection in available:
            print(f"🔍 [Qwen3-TTS] Using requested attention: {selection}")
            return selection
        else:
            print(f"⚠️ [Qwen3-TTS] Requested attention '{selection}' not available, falling back to sdpa")
            if "sdpa" in available:
                return "sdpa"
            return "eager"


def unload_cached_model():
    """Unload all cached models and clear GPU memory."""
    global _MODEL_CACHE
    if _MODEL_CACHE:
        print(f"🗑️ [Qwen3-TTS] Unloading {_MODEL_CACHE.__len__()} cached model(s)...")
        _MODEL_CACHE.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()
    print(f"✅ [Qwen3-TTS] Model cache and GPU memory cleared")


def apply_qwen3_patches(model):
    """Apply stability and compatibility patches to the model instance"""
    if model is None:
        return
    
    # 1. Monkeypatch model._normalize_audio_inputs to return mutable lists to avoid upstream tuple assignment bug
    # and to support (waveform, sr) tuple format from ComfyUI
    orig_normalize = getattr(model, "_normalize_audio_inputs", None)
    
    def _safe_normalize(self, audios):
        # Adapted from upstream but returns list entries as lists [waveform, sr]
        # and correctly handles the (waveform, sr) tuple format
        if isinstance(audios, list):
            items = audios
        elif isinstance(audios, tuple) and len(audios) == 2 and isinstance(audios[0], np.ndarray):
            # This is a single audio in tuple format (waveform, sr)
            items = [audios]
        else:
            items = [audios]

        out = []
        for a in items:
            if a is None:
                continue
            if isinstance(a, str):
                wav, sr = self._load_audio_to_np(a)
                out.append([wav.astype(np.float32), int(sr)])
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, list) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                # If we still can't identify it, it might be the cause of our NoneType or other issues
                print(f"⚠️ [Qwen3-TTS] Unknown audio input type: {type(a)}")
                continue

        # ensure mono
        for i in range(len(out)):
            wav, sr = out[i][0], out[i][1]
            if wav.ndim > 1:
                out[i][0] = np.mean(wav, axis=-1).astype(np.float32)
        return out

    try:
        model._normalize_audio_inputs = types.MethodType(_safe_normalize, model)
    except Exception as e:
        print(f"⚠️ [Qwen3-TTS] Failed to apply audio normalization patch: {e}")


def download_model_if_needed(model_id: str, qwen_root: str) -> str:
    """Download a specific model if not found locally"""
    folder_name = model_id.split("/")[-1]
    target_dir = os.path.join(qwen_root, folder_name)
    
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Model already exists
        return target_dir
    
    print(f"\n📥 [Qwen3-TTS] Downloading {model_id}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=target_dir)
        print(f"✅ [Qwen3-TTS] {folder_name} downloaded successfully.\n")
        return target_dir
    except ImportError:
        print("⚠️ [Qwen3-TTS] 'huggingface_hub' not found. Please install it to use auto-download.")
        return None
    except Exception as e:
        print(f"❌ [Qwen3-TTS] Failed to download {model_id}: {e}")
        return None


def check_and_download_tokenizer():
    """Check and download tokenizer (shared by all models)"""
    global _MODELS_CHECKED
    if _MODELS_CHECKED:
        return
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # ComfyUI root search
    comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
    qwen_root = os.path.join(comfy_models_path, "qwen-tts")
    os.makedirs(qwen_root, exist_ok=True)

    # Download tokenizer (required by all models)
    tokenizer_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    download_model_if_needed(tokenizer_id, qwen_root)
    
    _MODELS_CHECKED = True


def load_qwen_model(model_type: str, model_choice: str, device: str, precision: str, attention: str = "auto", unload_after: bool = False, previous_attention: str = None):
    """Shared model loading logic with caching and local path priority"""
    global _MODEL_CACHE
    
    if previous_attention is not None and previous_attention != attention:
        print(f"🔄 [Qwen3-TTS] Attention changed from '{previous_attention}' to '{attention}', clearing cache...")
        unload_cached_model()
    
    attn_impl = get_attention_implementation(attention)
    
    # Check and download tokenizer (shared by all models)
    check_and_download_tokenizer()
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  # 针对 Mac 的关键修复
        else:
            device = "cpu"
    
    # 强制 Mac 使用 float16 或 bfloat16 (MPS 跑 float32 会很慢)
    if device == "mps" and precision == "bf16":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    # Set precision
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    # VoiceDesign restriction
    if model_type == "VoiceDesign" and model_choice == "0.6B":
        raise RuntimeError("❌ VoiceDesign only supports 1.7B models!")
        
    # Cache key includes attention implementation
    cache_key = (model_type, model_choice, device, precision, attn_impl)
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
        # Try to download the specific model if not found locally
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
        qwen_root = os.path.join(comfy_models_path, "qwen-tts")
        
        downloaded_path = download_model_if_needed(final_source, qwen_root)
        if downloaded_path:
            final_source = downloaded_path
            print(f"✅ [Qwen3-TTS] Loading downloaded model: {os.path.basename(final_source)}")
        else:
            # Fall back to remote loading if download failed
            print(f"🌐 [Qwen3-TTS] Loading remote model: {final_source}")

    if Qwen3TTSModel is None:
        raise RuntimeError(
            "❌ [Qwen3-TTS] Model class is not loaded because the 'qwen_tts' package failed to import. "
            "Please check the ComfyUI console for the detailed 'Critical Import Error' above."
        )

    # Map attention implementation to model loading parameter
    attn_param = None
    use_sage_attn = False
    
    if attn_impl == "flash_attn":
        attn_param = "flash_attention_2"
    elif attn_impl == "sage_attn":
        use_sage_attn = True
    elif attn_impl == "sdpa":
        attn_param = "sdpa"
    elif attn_impl == "eager":
        attn_param = "eager"
    
    # Handle sage_attn (sageattention package)
    if use_sage_attn:
        try:
            from sageattention import sageattn
            print(f"🔧 [Qwen3-TTS] Loading model with sage_attn (sageattention)")
            
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
            
            # Patch attention modules to use sageattention
            patched_count = 0
            for name, module in model.model.named_modules():
                if hasattr(module, 'forward') and 'Attention' in type(module).__name__ or 'attn' in name.lower():
                    try:
                        original_forward = module.forward
                        def make_sage_forward(orig_forward, mod):
                            def sage_forward(*args, **kwargs):
                                # Extract q, k, v from attention call
                                if len(args) >= 3:
                                    q, k, v = args[0], args[1], args[2]
                                else:
                                    return orig_forward(*args, **kwargs)
                                
                                # Handle attention_mask
                                attn_mask = kwargs.get('attention_mask', None)
                                
                                # Call sageattention
                                out = sageattn(q, k, v, is_causal=False, attn_mask=attn_mask)
                                return out
                            return sage_forward
                        
                        module.forward = make_sage_forward(original_forward, module)
                        patched_count += 1
                    except Exception:
                        pass
            
            print(f"🔧 [Qwen3-TTS] Patched {patched_count} attention modules with sage_attn")
            
        except (ImportError, Exception) as e:
            print(f"⚠️ [Qwen3-TTS] Failed with sage_attn, falling back to default attention: {e}")
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
    else:
        try:
            if attn_param:
                print(f"🔧 [Qwen3-TTS] Loading model with attention: {attn_impl}")
                model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype, attn_implementation=attn_param)
            else:
                print(f"🔧 [Qwen3-TTS] Loading model with attention: {attn_impl}")
                model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
        except (ImportError, ValueError, Exception) as e:
            print(f"⚠️ [Qwen3-TTS] Failed with {attn_impl}, falling back to default attention: {e}")
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype)
    
    # Apply patches
    apply_qwen3_patches(model)
    
    _MODEL_CACHE[cache_key] = model
    
    if unload_after:
        def unload_callback():
            unload_cached_model()
        model._unload_callback = unload_callback
    else:
        model._unload_callback = None
    
    return model

class VoiceDesignNode:
    """
    VoiceDesign Node: Generate custom voices based on text descriptions.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello world", "placeholder": "Enter text to synthesize"}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (required for VoiceDesign)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceDesign: Generate custom voices from descriptions."

    def generate(self, text: str, instruct: str, model_choice: str, device: str, precision: str, language: str, seed: int = 0, max_new_tokens: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not text or not instruct:
            raise RuntimeError("Text and instruction description are required")

        # Track previous attention for cache invalidation
        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "VoiceDesign":
                previous_attention = key[4] if len(key) > 4 else None
                break

        # Load model
        model = load_qwen_model("VoiceDesign", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

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
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}
            
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
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
                "target_text": ("STRING", {"multiline": True, "default": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional)"}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT", {"tooltip": "Reusable voice clone prompt from VoiceClonePromptNode"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
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
                    sr = int(a0)
                    waveform = np.asarray(a1)
                elif isinstance(a1, (int, float)) and isinstance(a0, (list, np.ndarray, torch.Tensor)):
                    sr = int(a1)
                    waveform = np.asarray(a0)
            elif isinstance(audio_tensor, list):
                # maybe [waveform, sr]
                if len(audio_tensor) == 2 and isinstance(audio_tensor[0], (list, np.ndarray)) and isinstance(audio_tensor[1], (int, float)):
                    waveform = np.asarray(audio_tensor[0])
                    sr = int(audio_tensor[1])
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

    def generate(self, target_text: str, model_choice: str, device: str, precision: str, language: str, 
                 ref_audio: Optional[Dict[str, Any]] = None, ref_text: str = "", 
                 voice_clone_prompt: Optional[Any] = None, seed: int = 0, 
                 max_new_tokens: int = 2048,
                 top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05,
                 x_vector_only: bool = False, attention: str = "auto",
                 unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if ref_audio is None and voice_clone_prompt is None:
            raise RuntimeError("Either reference audio or voice clone prompt is required")
        
        # Track previous attention for cache invalidation
        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break
        
        # Load model
        model = load_qwen_model("Base", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed % (2**32))
        
        audio_tuple = None
        if ref_audio is not None:
            if isinstance(ref_audio, dict):
                audio_tuple = self._audio_tensor_to_tuple(ref_audio)
            else:
                raise RuntimeError("Unsupported reference audio format")

        try:
            mapped_lang = LANGUAGE_MAP.get(language, "auto")
            
            # Check if voice_clone_prompt is provided
            voice_clone_prompt_param = None
            ref_audio_param = None
            if voice_clone_prompt is not None:
                voice_clone_prompt_param = voice_clone_prompt
            elif ref_audio is not None:
                ref_audio_param = audio_tuple
            else:
                raise RuntimeError("Either 'ref_audio' or 'voice_clone_prompt' must be provided")

            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                ref_audio=ref_audio_param,
                ref_text=ref_text if ref_text and ref_text.strip() else None,
                voice_clone_prompt=voice_clone_prompt_param,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            # Ensure waveform is 1D [samples]
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            # Convert to ComfyUI format: [batch, channels, samples]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}
            
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)
        raise RuntimeError("Invalid audio data generated")


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
                "device": (["auto", "cuda","mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (optional)"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "CustomVoice: Generate speech using preset speakers."

    def generate(self, text: str, speaker: str, model_choice: str, device: str, precision: str, language: str, seed: int = 0, instruct: str = "", max_new_tokens: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not text or not speaker:
            raise RuntimeError("Text and speaker are required")

        # Track previous attention for cache invalidation
        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "CustomVoice":
                previous_attention = key[4] if len(key) > 4 else None
                break
        
        # Load model
        model = load_qwen_model("CustomVoice", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

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
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            # Ensure waveform is 1D [samples]
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            # Convert to ComfyUI format: [batch, channels, samples]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}
            
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)
        raise RuntimeError("Invalid audio data generated")


class VoiceClonePromptNode:
    """
    VoiceClonePrompt Node: Extract voice features from reference audio for reuse.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (highly recommended for better quality)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip": "If True, only speaker embedding is extracted (ref_text not needed)"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "VoiceClonePrompt: Extract and cache voice features for reuse in VoiceClone node."

    def create_prompt(self, ref_audio: Dict[str, Any], ref_text: str, model_choice: str, device: str, precision: str, attention: str, x_vector_only: bool = False, unload_model_after_generate: bool = False) -> Tuple[Any]:
        if ref_audio is None:
            raise RuntimeError("Reference audio is required")
        
        # Track previous attention for cache invalidation
        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break
        
        # Load model (usually Base model)
        model = load_qwen_model("Base", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

        # Reuse VoiceCloneNode's audio parsing logic
        vcn = VoiceCloneNode()
        audio_tuple = vcn._audio_tensor_to_tuple(ref_audio)

        # Extract prompt
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text if ref_text and ref_text.strip() else None,
            x_vector_only_mode=x_vector_only,
        )

        # Unload model if requested
        if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
            model._unload_callback()

        return (prompt_items,)


class RoleBankNode:
    """
    RoleBank Node: Manage a collection of voice prompts mapped to names.
    Supports up to 8 roles per node.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs = {
            "required": {},
            "optional": {}
        }
        for i in range(1, 9):
            inputs["optional"][f"role_name_{i}"] = ("STRING", {"default": f"Role{i}"})
            inputs["optional"][f"prompt_{i}"] = ("VOICE_CLONE_PROMPT",)
        return inputs

    RETURN_TYPES = ("QWEN3_ROLE_BANK",)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "create_bank"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "RoleBank: Collect multiple voice prompts into a named registry for dialogue generation."

    def create_bank(self, **kwargs) -> Tuple[Dict[str, Any]]:
        bank = {}
        for i in range(1, 9):
            name = kwargs.get(f"role_name_{i}", "").strip()
            prompt = kwargs.get(f"prompt_{i}")
            if name and prompt is not None:
                bank[name] = prompt
        return (bank,)


class DialogueInferenceNode:
    """
    DialogueInference Node: Generate multi-role continuous dialogue from a script.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": "Role1: Hello, how are you?\nRole2: I am fine, thank you.", "placeholder": "Format: RoleName: Text"}),
                "role_bank": ("QWEN3_ROLE_BANK",),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
                "pause_seconds": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration between sentences"}),
                "merge_outputs": ("BOOLEAN", {"default": True, "tooltip": "Merge all dialogue segments into a single long audio"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1, "tooltip": "Number of lines to process in parallel. Larger = faster but more VRAM."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens_per_line": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus sampling probability"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Top-k sampling parameter"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Sampling temperature"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "Penalty for repetition"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "Attention implementation"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "Unload model from memory after generation"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "DialogueInference: Execute a script with multiple roles and generate continuous speech."

    def generate_dialogue(self, script: str, role_bank: Dict[str, Any], model_choice: str, device: str, precision: str, language: str, pause_seconds: float, merge_outputs: bool, batch_size: int, seed: int = 0, max_new_tokens_per_line: int = 2048, top_p: float = 0.8, top_k: int = 20, temperature: float = 1.0, repetition_penalty: float = 1.05, attention: str = "auto", unload_model_after_generate: bool = False) -> Tuple[Dict[str, Any]]:
        if not script or not role_bank:
            raise RuntimeError("Script and Role Bank are required")

        # Track previous attention for cache invalidation
        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break

        # Load model
        model = load_qwen_model("Base", model_choice, device, precision, attention, unload_model_after_generate, previous_attention)

        # Set random seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import numpy as np
        np.random.seed(seed % (2**32))

        lines = script.strip().split("\n")
        
        texts_to_gen = []
        prompts_to_gen = [] # This must be a flat list of VoiceClonePromptItem
        langs_to_gen = []
        
        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        print(f"🎬 [Qwen3-TTS] Preparing batched inference for {len(lines)} lines...")

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or ":" not in line and "：" not in line:
                continue
            
            # Robust parser: find the first occurring colon (English or Chinese)
            pos_en = line.find(":")
            pos_cn = line.find("：")
            
            if pos_en == -1 and pos_cn == -1:
                continue
            
            if pos_en != -1 and (pos_cn == -1 or pos_en < pos_cn):
                role_name, text = line.split(":", 1)
            else:
                role_name, text = line.split("：", 1)
            
            role_name = role_name.strip()
            text = text.strip()
            
            if role_name not in role_bank:
                print(f"⚠️ [Qwen3-TTS] Role '{role_name}' not found in Role Bank, skipping line: {line[:20]}...")
                continue

            texts_to_gen.append(text)
            langs_to_gen.append(mapped_lang)
            
            # role_bank[role_name] is a List[VoiceClonePromptItem] (from create_voice_clone_prompt)
            # We need to extract the items and keep them in a flat list for the model's batch call
            role_prompts = role_bank[role_name]
            if isinstance(role_prompts, list):
                # Use the first one (most common case as users design one voice per role)
                prompts_to_gen.append(role_prompts[0])
            else:
                prompts_to_gen.append(role_prompts)

        if not texts_to_gen:
            raise RuntimeError("No valid dialogue lines found matching Role Bank.")

        try:
            results = []
            num_lines = len(texts_to_gen)
            sr = 24000
            
            # Micro-matching: process in chunks to save VRAM
            for i in range(0, num_lines, batch_size):
                chunk_texts = texts_to_gen[i:i + batch_size]
                chunk_prompts = prompts_to_gen[i:i + batch_size]
                chunk_langs = langs_to_gen[i:i + batch_size]
                
                print(f"🎙️ [Qwen3-TTS] Running batched inference for chunk {i//batch_size + 1}, lines {i+1} to {min(i+batch_size, num_lines)}...")
                
                wavs_list, sr = model.generate_voice_clone(
                    text=chunk_texts,
                    language=chunk_langs,
                    voice_clone_prompt=chunk_prompts,
                    max_new_tokens=max_new_tokens_per_line,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )
                
                for wav in wavs_list:
                    waveform = torch.from_numpy(wav).float()
                    # Enforce mono [1, 1, samples]
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)
                    elif waveform.ndim == 2:
                        waveform = waveform.unsqueeze(0)
                        if waveform.shape[1] > 1:
                            waveform = torch.mean(waveform, dim=1, keepdim=True)
                    
                    results.append(waveform)

                    # Add inter-sentence pause
                    if pause_seconds > 0:
                        silence_len = int(pause_seconds * sr)
                        silence = torch.zeros((1, 1, silence_len))
                        results.append(silence)
                
                # Cleanup cache after each chunk to maximize VRAM availability
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            raise RuntimeError(f"Dialogue generation failed during chunked inference: {e}")

        if not results:
            raise RuntimeError("No dialogue lines were successfully generated.")

        if merge_outputs:
            # Concatenate along the sample dimension (last one)
            merged_waveform = torch.cat(results, dim=-1)
            audio_data = {"waveform": merged_waveform, "sample_rate": sr}
            
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)
        else:
            # Pad to longest for batch format
            max_len = max(w.shape[-1] for w in results)
            padded_results = []
            for w in results:
                curr_len = w.shape[-1]
                if curr_len < max_len:
                    padding = torch.zeros((w.shape[0], w.shape[1], max_len - curr_len))
                    w = torch.cat([w, padding], dim=-1)
                padded_results.append(w)
            
            batched_waveform = torch.cat(padded_results, dim=0)
            audio_data = {"waveform": batched_waveform, "sample_rate": sr}
            
            # Unload model if requested
            if unload_model_after_generate and hasattr(model, '_unload_callback') and model._unload_callback:
                model._unload_callback()
            
            return (audio_data,)
