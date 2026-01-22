# ComfyUI-Qwen-TTS Custom Nodes
# Based on the open-source Qwen3-TTS project by Alibaba Qwen team

import os
import sys
import torch

# Add current directory to path for qwen_tts package
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_path = os.path.join(current_dir, 'qwen_tts')
if qwen_tts_path not in sys.path:
    sys.path.insert(0, qwen_tts_path)

# Import nodes
from .nodes import (
    VoiceDesignNode,
    VoiceCloneNode,
    CustomVoiceNode,
)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "Qwen3TTSVoiceClone": VoiceCloneNode,
    "Qwen3TTSVoiceDesign": VoiceDesignNode,
    "Qwen3TTSCustomVoice": CustomVoiceNode,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSVoiceClone": "Qwen3-TTS VoiceClone",
    "Qwen3TTSVoiceDesign": "Qwen3-TTS VoiceDesign",
    "Qwen3TTSCustomVoice": "Qwen3-TTS CustomVoice",
}

# Version information
__version__ = "1.0.0"

print(f"✅ ComfyUI-Qwen-TTS v{__version__} loaded")
