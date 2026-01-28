
import os
import io
import json
import base64
import shutil
import logging
import torch
import numpy as np
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

import folder_paths
from comfy.utils import ProgressBar
from comfy import model_management
from server import PromptServer

# Handle qwen_tts import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from qwen_tts.finetuning.dataset import TTSDataset
    from safetensors.torch import save_file
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    import scipy.io.wavfile as wav
except ImportError as e:
    print(f"Training node missing dependencies: {e}")
    TTSDataset = None

logger = logging.getLogger("ComfyUI-Qwen-TTS-Train")

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

def send_training_update(node_id, data):
    if PromptServer.instance is not None:
        PromptServer.instance.send_sync(
            "qwen3tts_training_update",
            {"node": str(node_id), **data}
        )

def audio_to_base64(audio_np, sample_rate):
    buffer = io.BytesIO()
    audio_np = np.asarray(audio_np).flatten()
    
    if audio_np.dtype in (np.float32, np.float64, float):
        if np.any(~np.isfinite(audio_np)):
            audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_np = (audio_np * 32767).astype(np.int16)
    elif audio_np.dtype != np.int16:
        audio_np = audio_np.astype(np.int16)
        
    wav.write(buffer, sample_rate, audio_np)
    buffer.seek(0)
    return "data:audio/wav;base64," + base64.b64encode(buffer.read()).decode("utf-8")

class Qwen3TTS_Train_Node:
    @classmethod
    def INPUT_TYPES(cls):
        default_output = os.path.join(folder_paths.output_directory, "qwen3tts_finetune")
        
        # Import ALL_MODELS from nodes.py to populate the list
        try:
            from .nodes import ALL_MODELS
            # Filter for Base models usually, but let's allow all 1.7B variants as potential starting points
            # Though strictly training requires speaker_encoder which is in Base.
            # Let's verify if we should restrict list. 
            # Reference used AVAILABLE_QWEN3TTS_MODELS keys.
            # We will use ALL_MODELS for simplicity as it contains the repo IDs.
            model_list = ALL_MODELS
        except ImportError:
            model_list = ["Qwen/Qwen3-TTS-12Hz-1.7B-Base"]

        return {
            "required": {
                "init_model": (model_list, {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}),
                "tokenizer": (["Qwen/Qwen3-TTS-Tokenizer-12Hz"], {"default": "Qwen/Qwen3-TTS-Tokenizer-12Hz"}),
                "audio_folder": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": default_output}),
                "speaker_name": ("STRING", {"default": "new_speaker"}),
                "test_text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of my new voice."
                }),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean"], {"default": "English"}),
                "learning_rate": ("FLOAT", {"default": 2e-5, "min": 1e-7, "max": 1e-3, "step": 1e-7}),
                "num_epochs": ("INT", {"default": 10, "min": 1, "max": 100}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "gradient_accumulation_steps": ("INT", {"default": 4, "min": 1, "max": 64}),
                "validate_every": ("INT", {"default": 2, "min": 1, "max": 10}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_path",)
    FUNCTION = "train"
    CATEGORY = "Qwen3TTS"
    OUTPUT_NODE = True

    @torch.inference_mode(False)
    def train(self, init_model, tokenizer, audio_folder, output_dir, speaker_name, test_text, language, learning_rate, num_epochs, batch_size, gradient_accumulation_steps, validate_every, unique_id=None):
        torch.set_grad_enabled(True)
        
        if TTSDataset is None:
            raise RuntimeError("Training dependencies missing. Please check requirements.")

        if not os.path.isdir(audio_folder):
            raise ValueError(f"Audio folder not found: {audio_folder}")

        # Basic setup
        os.makedirs(output_dir, exist_ok=True)
        send_training_update(unique_id, {"type": "status", "message": "Initializing..."})
        
        # 1. Load Model Fresh
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        
        # Determine model path from input
        model_name = init_model.split("/")[-1] # Use the end part of the repo ID as folder name

        
        # Check standard ComfyUI location
        # Use simple default path
        base_path = os.path.join(folder_paths.models_dir, "qwen-tts")
        models_dir = base_path

        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(os.path.join(model_path, "config.json")):
             # Check if it's an official repo ID from the list, or just try to download whatever string is passed
             repo_id = init_model # Default assumption
             # Try to find if it matches a known family mapping
             from .nodes import MODEL_FAMILY_TO_HF
             if init_model in MODEL_FAMILY_TO_HF.values():
                 repo_id = init_model
             
             logger.info(f"Downloading {model_name} from {repo_id}...")
             send_training_update(unique_id, {"type": "status", "message": f"Downloading {model_name}..."})
             try:
                snapshot_download(repo_id=repo_id, local_dir=model_path)
             except Exception as e:
                 # If download fails, maybe it's a local path relative to models_dir?
                 # But for now, we assume it's a Repo ID if not found locally.
                 raise ValueError(f"Model not found locally and download failed: {e}")

        send_training_update(unique_id, {"type": "status", "message": "Loading Base Model..."})
        
        # Load Main Model
        tts_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa" # Use sdpa for broad compatibility
        )
        
        # Load Tokenizer using input selection
        tokenizer_name = tokenizer.split("/")[-1]
        tokenizer_path = os.path.join(models_dir, tokenizer_name)
        
        # Determine repo_id for tokenizer - defaulting to the input string if we can't infer otherwise, 
        # or checking against knowns. For now, since we only offer one, we use the input string directly as repo_id if download is needed.
        if not os.path.exists(os.path.join(tokenizer_path, "config.json")):
             logger.info(f"Downloading {tokenizer_name}...")
             snapshot_download(repo_id=tokenizer, local_dir=tokenizer_path)
             
        tts_tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_path)
        if hasattr(tts_tokenizer, 'model'):
             tts_tokenizer.model.to("cuda")
             tts_tokenizer.device = torch.device("cuda")

        # 2. Prepare Dataset
        entries = self._prepare_dataset(audio_folder, tts_tokenizer, language, unique_id)
        if not entries:
            raise ValueError("No valid audio/txt pairs found in folder.")
            
        train_dataset = TTSDataset(entries, tts_model.processor, tts_model.model.config)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        
        # 3. Training Loop Setup
        tts_model.model.train()
        for param in tts_model.model.parameters():
            param.requires_grad = True
            
        optimizer = AdamW(tts_model.model.parameters(), lr=learning_rate)
        model = tts_model.model # Access internal HuggingFace model
        device = next(model.parameters()).device
        
        target_speaker_embedding = None
        total_steps = num_epochs * len(train_dataloader)
        pbar = ProgressBar(total_steps)
        
        send_training_update(unique_id, {"type": "status", "message": "Starting Training..."})
        
        final_checkpoint = None
        optimizer.zero_grad() # Initialize gradients
        
        for epoch in range(num_epochs):
            if model_management.processing_interrupted():
                break
                
            epoch_loss = 0
            for step, batch in enumerate(train_dataloader):
                if model_management.processing_interrupted():
                    break
                    
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                codec_ids = batch['codec_ids'].to(device)
                ref_mels = batch['ref_mels'].to(device).to(torch.bfloat16)
                text_embedding_mask = batch['text_embedding_mask'].to(device).to(torch.bfloat16)
                codec_embedding_mask = batch['codec_embedding_mask'].to(device).to(torch.bfloat16)
                attention_mask = batch['attention_mask'].to(device)
                codec_0_labels = batch['codec_0_labels'].to(device)
                codec_mask = batch['codec_mask'].to(device).to(torch.bfloat16)
                codec_mask_bool = batch['codec_mask'].to(device).to(torch.bool)

                # Speaker Embedding
                speaker_embedding = model.speaker_encoder(ref_mels).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                # Embeddings Calculation
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]
                
                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding # Inject speaker
                
                input_embeddings = input_text_embedding + input_codec_embedding
                
                # Add codec layers
                for i in range(1, 16):
                    layer_embed = model.talker.code_predictor.get_input_embeddings()[i-1](codec_ids[:, :, i])
                    input_embeddings += layer_embed * codec_mask.unsqueeze(-1)
                    
                # Forward
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )
                
                # Sub-talker loss
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask_bool[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask_bool]
                
                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
                
                loss = outputs.loss + sub_talker_loss
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Accumulation Step
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item()
                pbar.update(1)
                
                if step % 5 == 0:
                    send_training_update(unique_id, {
                        "type": "progress", 
                        "epoch": epoch+1, 
                        "loss": loss.item()
                    })

            # Checkpoint
            if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
                
                # Copy Base Model structure
                shutil.copytree(model_path, checkpoint_dir, dirs_exist_ok=True)
                
                # Update Config
                config_path = os.path.join(checkpoint_dir, "config.json")
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                
                cfg["tts_model_type"] = "custom_voice"
                cfg.setdefault("talker_config", {})["spk_id"] = {speaker_name.lower(): 3000}
                cfg["talker_config"]["spk_is_dialect"] = {speaker_name.lower(): False}
                
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                
                # Save Weights (Filtering speaker_encoder)
                state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("speaker_encoder")}
                
                # Inject learned speaker embedding
                if target_speaker_embedding is not None:
                    weight_key = 'talker.model.codec_embedding.weight'
                    state_dict[weight_key][3000] = target_speaker_embedding[0].detach().cpu().to(torch.bfloat16)

                save_file(state_dict, os.path.join(checkpoint_dir, "model.safetensors"))
                final_checkpoint = checkpoint_dir
                
                # Validation
                self._run_validation(checkpoint_dir, test_text, speaker_name, unique_id, epoch+1)

        send_training_update(unique_id, {"type": "status", "message": "Done!"})
        return (final_checkpoint,)

    def _prepare_dataset(self, audio_folder, tokenizer, language, unique_id):
        folder = Path(audio_folder)
        files = sorted([f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS])
        entries = []
        
        # Use first file as ref audio for all (consistency)
        ref_audio = str(files[0].absolute()) if files else None
        
        for f in files:
            txt_path = f.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as tf:
                    text = tf.read().strip()
                if text:
                    entries.append({
                        "audio": str(f.absolute()),
                        "text": text,
                        "language": language, 
                        "ref_audio": ref_audio
                    })
        
        # Encode
        batch_size = 8
        send_training_update(unique_id, {"type": "status", "message": f"Encoding {len(entries)} samples..."})
        
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i+batch_size]
            paths = [e["audio"] for e in batch]
            try:
                enc = tokenizer.encode(paths)
                for j, codes in enumerate(enc.audio_codes):
                    entries[i+j]["audio_codes"] = codes.cpu().tolist()
            except Exception as e:
                logger.error(f"Encode error: {e}")
                raise e
                
        return entries

    def _run_validation(self, checkpoint_path, text, speaker, unique_id, epoch):
        try:
            val_model = Qwen3TTSModel.from_pretrained(
                checkpoint_path, 
                dtype=torch.bfloat16, 
                device_map="cuda", 
                attn_implementation="sdpa"
            )
            wavs, sr = val_model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language="English",
                do_sample=True,
                max_new_tokens=2048
            )
            if wavs:
                b64 = audio_to_base64(wavs[0], sr)
                send_training_update(unique_id, {
                    "type": "validation",
                    "epoch": epoch,
                    "audio_base64": b64
                })
            del val_model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Validation failed: {e}")
