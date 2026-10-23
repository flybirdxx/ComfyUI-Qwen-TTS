
import os
import io
import json
import base64
import shutil
import logging
import gc
import torch
import numpy as np
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

import folder_paths
from comfy.utils import ProgressBar
from comfy import model_management
from server import PromptServer

# ===== XPU 稳定性补丁：修复 level_zero 设备丢失 =====
import transformers.pytorch_utils
import importlib
import torch

# XPU stability patch: redirect isin_mps_friendly to CPU to prevent Intel XPU level_zero crashes.
# isin_mps_friendly was removed in newer transformers versions, so guard the patch.
if hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    _original_isin = transformers.pytorch_utils.isin_mps_friendly

    def _safe_isin(elements, test_elements):
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                if isinstance(elements, torch.Tensor) and elements.device.type == 'xpu':
                    elements_cpu = elements.cpu()
                    test_cpu = test_elements.cpu() if isinstance(test_elements, torch.Tensor) else test_elements
                    result_cpu = torch.isin(elements_cpu, test_cpu)
                    return result_cpu.to(elements.device)
        except Exception:
            pass
        return _original_isin(elements, test_elements)

    transformers.pytorch_utils.isin_mps_friendly = _safe_isin

    import transformers.generation.logits_process
    importlib.reload(transformers.generation.logits_process)


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
    server = getattr(PromptServer, "instance", None)
    if server is not None:
        server.send_sync(
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
                "device": (["auto", "cuda", "xpu", "mps", "cpu"], {
                    "default": "auto",
                    "tooltip": "Which device trains. auto = xpu > cuda > mps > cpu. "
                               "A device this build does not have stops with a clear error instead "
                               "of switching silently.",
                }),
                "optimizer_state": (["bf16", "8bit"], {
                    "default": "bf16",
                    "tooltip": "Precision of the Adam moments: bf16 ~7.2 GiB for a 1.9B model, "
                               "8bit ~3.8 GiB. On XPU the 8-bit path needs bitsandbytes; without it "
                               "the node falls back to bf16 and says so.",
                }),
                "optimizer_placement": (["vram", "ram"], {
                    "default": "vram",
                    "tooltip": "Where the Adam moments live. vram is the original behaviour; on a "
                               "16 GiB card a 1.9B full fine-tune runs out of memory there, so pick "
                               "ram (states in system RAM, updates on the CPU) or 8bit.",
                }),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Recompute each layer in the backward pass: measured ~5.7 GiB less "
                               "peak VRAM on the 1.9B model, at the cost of extra compute.",
                }),
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
    def train(self, init_model, tokenizer, audio_folder, output_dir, speaker_name, test_text, language, learning_rate, num_epochs, batch_size, gradient_accumulation_steps, validate_every, device="auto", optimizer_state="bf16", optimizer_placement="vram", gradient_checkpointing=False, unique_id=None):
        holder = {}
        try:
            return self._train_impl(
                init_model, tokenizer, audio_folder, output_dir, speaker_name, test_text,
                language, learning_rate, num_epochs, batch_size, gradient_accumulation_steps,
                validate_every, device, optimizer_state, optimizer_placement,
                gradient_checkpointing, unique_id, holder,
            )
        finally:
            # Leaving this to the garbage collector kept the previous run alive: a second
            # training in the same ComfyUI process started +3.7 GB of VRAM and +17.5 GB of
            # RAM higher (measured with two back-to-back runs), which is how a card runs
            # out of memory after a few rounds. The optimizer registers a backward hook on
            # every parameter, and that hook is stored on the C++ side of the autograd
            # metadata - invisible to Python's collector - so the optimizer, its host
            # buffers and the whole model stay reachable long after the run returns.
            # Tear those tensors down explicitly instead of waiting for a collection that
            # never happens.
            self._free_run_memory(holder)
            gc.collect()
            self._empty_cache()

    @staticmethod
    def _free_run_memory(holder):
        """Drop the weights, optimizer states and pinned host copies of a finished run."""
        optimizer = holder.get("optimizer")
        model = holder.get("model")
        if optimizer is not None:
            host = getattr(optimizer, "param_d2h_map", None)
            if host:
                for p in list(host.values()):
                    p.grad = None
                    p.data = torch.empty(0)
                host.clear()
            for inner in list(getattr(optimizer, "optim_dict", {}).values()):
                inner.state.clear()
            d_opt = getattr(optimizer, "d_opt", None)
            if d_opt is not None:
                d_opt.state.clear()
            if hasattr(optimizer, "queue"):
                optimizer.queue.clear()
        if model is not None:
            for p in model.parameters():
                p.grad = None
                p.data = torch.empty(0)
        holder.clear()

    def _train_impl(self, init_model, tokenizer, audio_folder, output_dir, speaker_name, test_text, language, learning_rate, num_epochs, batch_size, gradient_accumulation_steps, validate_every, device="auto", optimizer_state="bf16", optimizer_placement="vram", gradient_checkpointing=False, unique_id=None, holder=None):
        torch.set_grad_enabled(True)
        
        if TTSDataset is None:
            raise RuntimeError("Training dependencies missing. Please check requirements.")

        if not os.path.isdir(audio_folder):
            raise ValueError(f"Audio folder not found: {audio_folder}")
			
        # ----- 设备：节点可选（auto = xpu > cuda > mps > cpu），XPU 走专属分支 -----
        train_device = self._resolve_device(device)
        device_map_arg = {"": f"{train_device}:0"} if train_device in ("xpu", "cuda") else train_device

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
            device_map=device_map_arg, 
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
             tts_tokenizer.model.to(train_device)                     # 改为 train_device
             tts_tokenizer.device = torch.device(train_device)

        # 2. Prepare Dataset
        entries = self._prepare_dataset(audio_folder, tts_tokenizer, language, unique_id)
        if not entries:
            raise ValueError("No valid audio/txt pairs found in folder.")

        # Long single clips are the classic XPU OOM. The sub-talker expands every codec
        # frame into num_code_groups tokens (16), so its attention is quadratic in the clip
        # length. Measured peaks for one sample (1.9B, gradient checkpointing on):
        # 20s -> 7.5 GiB, 60s -> 8.9 GiB, 150s -> 16.5 GiB (OOM on a 16 GiB card) and a
        # 431s file ran out of memory during codec encoding. Warn before the run starts.
        long_files = []
        for e in entries:
            frames = len(e.get("audio_codes") or [])
            if frames and frames / 12.5 > 75.0:
                long_files.append((os.path.basename(e["audio"]), frames / 12.5))
        if long_files:
            detail = ", ".join(f"{n} ({s:.0f}s)" for n, s in sorted(long_files, key=lambda x: -x[1])[:5])
            msg = (f"long training audio: {detail}. One clip above ~60-75s costs far more "
                   f"VRAM than the same audio split up (measured: 20s=7.5 GiB, 60s=8.9 GiB, "
                   f"150s=16.5 GiB peak), and multi-minute files can OOM while encoding. "
                   f"Split into 20-30s segments with matching transcripts.")
            logger.warning(f"[Qwen3TTS][train] {msg}")
            send_training_update(unique_id, {"type": "status", "message": msg})

        # The codec tokenizer is only needed to turn audio into codes. It used to stay on
        # the accelerator for the whole run, holding ~0.65 GiB that the training pass needs
        # more (measured: the 12Hz tokenizer checkpoint is 650.7 MB).
        try:
            if hasattr(tts_tokenizer, "model"):
                tts_tokenizer.model.to("cpu")
            del tts_tokenizer
            self._empty_cache(train_device)
            logger.info("[Qwen3TTS][train] codec tokenizer parked on cpu after encoding")
        except Exception as e:  # noqa: BLE001 - training can continue without this win
            logger.warning(f"[Qwen3TTS][train] could not park the codec tokenizer: {e}")
            
        train_dataset = TTSDataset(entries, tts_model.processor, tts_model.model.config)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        
        # 3. Training Loop Setup
        tts_model.model.train()
        for param in tts_model.model.parameters():
            param.requires_grad = True
            
        model = tts_model.model # Access internal HuggingFace model
        if holder is not None:
            holder["model"] = model
        optimizer = self._build_optimizer(
            optimizer_state,
            optimizer_placement,
            model.parameters(),
            learning_rate,
            train_device,
            gradient_accumulation_steps,
            unique_id,
        )
        if holder is not None:
            holder["optimizer"] = optimizer
        if gradient_checkpointing:
            status = self._enable_gradient_checkpointing(model)
            logger.info(f"[Qwen3TTS][train] gradient checkpointing: {status}")
            send_training_update(unique_id, {"type": "status", "message": f"Gradient checkpointing: {status}"})
        else:
            # Off by default, but on a 16 GiB card it is usually the difference between
            # fitting and an XPU OOM: measured 10.0 -> 4.3 GiB peak on the 1.9B model.
            msg = ("gradient checkpointing is OFF - fine on a large card, but a 1.9B "
                   "fine-tune needs it when VRAM is tight (measured ~5.7 GiB less peak).")
            logger.warning(f"[Qwen3TTS][train] {msg}")
            send_training_update(unique_id, {"type": "status", "message": msg})
        device = next(model.parameters()).device
        
        target_speaker_embedding = None
        total_steps = num_epochs * len(train_dataloader)
        pbar = ProgressBar(total_steps)
        
        send_training_update(unique_id, {"type": "status", "message": "Starting Training..."})
        
        final_checkpoint = None
        optimizer.zero_grad() # Initialize gradients
        pending_grads = 0
        skipped_steps = 0
        
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

                if step == 0:
                    # Sequence length is what actually drives the activation footprint; print
                    # it so a long clip is visible in the log instead of only showing up as
                    # an XPU OOM a few seconds later.
                    seq_len = int(input_ids.shape[1])
                    logger.info(
                        "[Qwen3TTS][train] epoch %d first batch: seq=%d tokens, ckpt=%s",
                        epoch + 1, seq_len, bool(gradient_checkpointing),
                    )
                    if seq_len > 2000 and not gradient_checkpointing:
                        msg = (f"long training sample ({seq_len} tokens) with gradient "
                               f"checkpointing OFF - this is the usual cause of an XPU OOM; "
                               f"enable gradient checkpointing or split the audio into "
                               f"20-30s segments.")
                        logger.warning(f"[Qwen3TTS][train] {msg}")
                        send_training_update(unique_id, {"type": "status", "message": msg})

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
                
                epoch_loss += loss.item()
                
                if not torch.isfinite(loss):
                    skipped_steps += 1
                    logger.warning(
                        "[Qwen3TTS][train] epoch %d micro-batch %d produced a non-finite loss "
                        "(%s); dropping this batch", epoch + 1, step, loss.item(),
                    )
                    send_training_update(unique_id, {
                        "type": "status",
                        "message": f"Skipped a non-finite batch (loss={loss.item()}); training continues",
                    })
                    self._reset_gradients(model, optimizer)
                    pending_grads = 0
                    pbar.update(1)
                    continue
                
                # Gradient accumulation: the loop used to call zero_grad() right before
                # every backward(), so only the last micro-batch of each cycle ever
                # reached the optimizer and 3/4 of the data was silently dropped.
                # Accumulate across the cycle now, step at its end, and scale the loss
                # so one cycle still equals the mean of its micro-batches.
                (loss / gradient_accumulation_steps).backward()
                pending_grads += 1
                
                if pending_grads >= gradient_accumulation_steps:
                    skipped_steps += self._apply_step(model, optimizer, unique_id, epoch, step)
                    pending_grads = 0
                
                pbar.update(1)
                
                if step % 5 == 0:
                    send_training_update(unique_id, {
                        "type": "progress", 
                        "epoch": epoch+1, 
                        "loss": loss.item()
                    })

            # Flush the tail of the epoch when the dataset size is not a multiple of
            # the accumulation window, otherwise the last few samples never update.
            if pending_grads > 0 and not model_management.processing_interrupted():
                skipped_steps += self._apply_step(model, optimizer, unique_id, epoch, "tail")
                pending_grads = 0

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
                self._run_validation(
                    checkpoint_dir,
                    test_text,
                    speaker_name,
                    unique_id,
                    epoch + 1,
                    train_device,
                    model,
                )

        done_msg = "Done!" if not skipped_steps else f"Done! ({skipped_steps} non-finite update(s) skipped)"
        send_training_update(unique_id, {"type": "status", "message": done_msg})
        if skipped_steps:
            logger.warning(
                "[Qwen3TTS][train] finished with %d skipped update(s) caused by non-finite "
                "values (kept the rest of the run usable)", skipped_steps,
            )
        return (final_checkpoint,)

    DEVICES = ("auto", "cuda", "xpu", "mps", "cpu")

    @staticmethod
    def _available(kind):
        """Is this backend usable in the running PyTorch build?"""
        try:
            if kind == "xpu":
                return bool(getattr(torch, "xpu", None)) and torch.xpu.is_available()
            if kind == "cuda":
                return torch.cuda.is_available()
            if kind == "mps":
                backend = getattr(torch.backends, "mps", None)
                return bool(backend) and backend.is_available()
            if kind == "cpu":
                return True
        except Exception:  # noqa: BLE001 - "unavailable" is the safe answer
            return False
        return False

    @classmethod
    def _resolve_device(cls, requested):
        """Node's device choice; 'auto' keeps the historical priority XPU > CUDA > MPS > CPU.

        An explicit choice that this machine cannot run is reported as such instead of
        silently landing somewhere else, so the node never trains on a device the user
        did not pick.
        """
        req = (requested or "auto").strip().lower()
        if req == "auto":
            for kind in ("xpu", "cuda", "mps"):
                if cls._available(kind):
                    return kind
            return "cpu"
        if req not in cls.DEVICES:
            raise ValueError(
                f"device='{requested}' is not one of {', '.join(cls.DEVICES)}"
            )
        if not cls._available(req):
            raise RuntimeError(
                f"device='{req}' was selected but is not available in this build "
                f"(torch {torch.__version__}; xpu={cls._available('xpu')}, "
                f"cuda={cls._available('cuda')}, mps={cls._available('mps')}). "
                f"Pick 'auto' or a device this machine actually has."
            )
        return req

    @staticmethod
    def _empty_cache(*devices):
        """Release the caching allocator on the given backends (ignores absent ones)."""
        targets = set(devices) or set(Qwen3TTS_Train_Node.DEVICES)
        for kind, flush in (
            ("xpu", lambda: torch.xpu.empty_cache()),
            ("cuda", lambda: torch.cuda.empty_cache()),
            ("mps", lambda: torch.mps.empty_cache()),
        ):
            if kind in targets and Qwen3TTS_Train_Node._available(kind):
                try:
                    flush()
                except Exception as e:  # noqa: BLE001 - cache flushing is best effort
                    logger.debug(f"[Qwen3TTS][train] {kind} empty_cache failed: {e}")

    def _build_optimizer(
        self,
        optimizer_state,
        optimizer_placement,
        params,
        lr,
        device="xpu",
        grad_accum_steps=1,
        unique_id=None,
    ):
        """Two independent choices, so every combination is valid.

        optimizer_state — how the Adam moments (exp_avg / exp_avg_sq) are stored:
            bf16  PyTorch default: the states follow the bf16 parameters (7.2 GiB for 1.9B)
            8bit  torchao block-wise quantized states (~3.6 GiB)
        optimizer_placement — where those states live (a device choice, not a switch):
            vram  on the accelerator, next to the parameters (default)
            ram   in system RAM; updates run on the CPU and are copied back
                  (1.9B: frees ~7.2 GiB of VRAM, 96 GB of RAM is plenty)

        Measured on the A770 (1.9B, L=1100): bf16+vram ~3.0 s/step, bf16+ram ~3.9 s/step,
        8bit+vram ~4.7 s/step. 8bit+vram is the fast way to buy VRAM back; 8bit+ram also
        runs the 8-bit states (on the host copies) and is simply the slowest option.

        Platform note: the kwargs below are only tightened for XPU. CUDA/ROCm/MPS keep
        PyTorch's own defaults, so their fast paths (fused/foreach) are untouched.
        """
        params = list(params)
        optimizer_class = (
            self._eight_bit_optimizer(device) if optimizer_state == "8bit" else AdamW
        )

        if optimizer_placement == "ram":
            if device == "cpu":
                # Nothing to offload: the parameters already live in system RAM, and
                # torchao's CPUOffloadOptimizer needs a CUDA/XPU device for the copies.
                logger.info(
                    "[Qwen3TTS][train] optimizer: %s states on cpu "
                    "(placement=ram is a no-op without an accelerator)",
                    optimizer_state,
                )
                return optimizer_class(params, lr=lr)
            from torchao.optim import CPUOffloadOptimizer

            # torchao frees each device gradient right after copying it to the host, and
            # the copy is an assignment, not an addition. With gradient accumulation the
            # device gradient therefore has to stay alive, otherwise every cycle keeps
            # only its last micro-batch.
            kwargs = {"offload_gradients": grad_accum_steps <= 1}
            if optimizer_class is AdamW and device == "xpu":
                # torchao defaults to fused=True, and the fused Adam kernel asks the
                # XPU device for fp64 (bias-correction scalars). DG2 has no fp64, so
                # the fused path raises "Required aspect fp64 is not supported".
                kwargs.update(fused=False, foreach=False)
            logger.info(
                "[Qwen3TTS][train] optimizer: %s states in RAM on %s "
                "(grad_accum=%d, device gradients %s)",
                optimizer_state,
                device,
                grad_accum_steps,
                "freed after each micro-batch" if kwargs["offload_gradients"] else "kept for accumulation",
            )
            return CPUOffloadOptimizer(
                params,
                optimizer_class=optimizer_class,
                lr=lr,
                # Offload every parameter (minimal_size is "keep anything smaller than
                # this on the GPU"): with the default cut-off a few hundred small
                # tensors stay on the device, so their gradients live somewhere the
                # host-side clipping below cannot see.
                minimal_size=1,
                **kwargs,
            )
        logger.info(
            "[Qwen3TTS][train] optimizer: %s states on the %s (PyTorch defaults)",
            optimizer_state,
            device,
        )
        return optimizer_class(params, lr=lr)

    @staticmethod
    def _eight_bit_optimizer(device):
        """8-bit Adam states: one implementation per device family, no substitution.

        The `device` choice decides which backend runs; a missing dependency raises with
        the install hint instead of quietly training with different states.

        XPU uses bitsandbytes on purpose. torchao's AdamW8bit is the reference
        implementation for CUDA/ROCm/MPS/CPU, but on XPU it silently destroys the
        weights: one real step on the 1.7B model with the segmented dataset turned
        315/480 tensors into NaN while the loss and every gradient were still finite
        (measured 2026-09-20, torch 2.14.0+xpu). The bitsandbytes kernel keeps the same
        step at 0/480.
        """
        if device == "xpu":
            try:
                from bitsandbytes.optim import AdamW8bit
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "optimizer_state=8bit on XPU needs bitsandbytes "
                    f"(pip install bitsandbytes); torchao's 8-bit optimizer is not usable "
                    f"on XPU. Import failed: {e}"
                ) from e
            logger.info("[Qwen3TTS][train] 8-bit states via bitsandbytes on XPU")
            return AdamW8bit
        try:
            from torchao.optim import AdamW8bit
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"optimizer_state=8bit on {device} needs torchao (pip install torchao); "
                f"import failed: {e}"
            ) from e
        logger.info("[Qwen3TTS][train] 8-bit states via torchao on %s", device)
        return AdamW8bit

    @staticmethod
    def _clip_gradients(model, optimizer, max_norm=1.0):
        """Clip on the side the optimizer actually reads gradients from.

        With CPU offload the backward hook copies each gradient to pinned host memory
        and clears the device copy, so clip_grad_norm_(model.parameters(), ...) only
        sees the handful of parameters that stayed on the GPU: measured norm 7.5
        instead of the real 484. The unclipped host gradients then blow the weights up
        (first training run produced 135/404 NaN tensors and generation crashed inside
        the Level Zero driver).
        """
        host_map = getattr(optimizer, "param_d2h_map", None)
        if host_map:
            params = [p for p in host_map.values() if getattr(p, "grad", None) is not None]
            if params:
                return torch.nn.utils.clip_grad_norm_(params, max_norm)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    @classmethod
    def _apply_step(cls, model, optimizer, unique_id, epoch, step):
        """Clip and step, but never let a non-finite gradient reach the weights.

        The XPU backward intermittently returns non-finite gradients for this model
        (measured: 1 of 3 identical runs on the segmented dataset produced an all-NaN
        gradient on the second update). Letting that through turned 400/404 tensors into
        NaN and the next validation then killed the process inside the Level Zero
        driver. Skipping one update keeps the run and the checkpoint usable.

        Returns 1 when the update was skipped, 0 otherwise.
        """
        norm = cls._clip_gradients(model, optimizer, 1.0)
        if not bool(torch.isfinite(norm)):
            logger.warning(
                "[Qwen3TTS][train] non-finite gradient norm (%s) at epoch %s step %s; "
                "skipping this update", norm, epoch + 1, step,
            )
            send_training_update(unique_id, {
                "type": "status",
                "message": "Non-finite gradient detected - skipped one update to keep the weights sane",
            })
            cls._reset_gradients(model, optimizer)
            return 1
        optimizer.step()
        optimizer.zero_grad()
        return 0

    @staticmethod
    def _reset_gradients(model, optimizer):
        """Drop accumulated gradients on both the device and the offloaded host copies."""
        host_map = getattr(optimizer, "param_d2h_map", None)
        if host_map:
            for p in host_map.values():
                if getattr(p, "grad", None) is not None:
                    p.grad.zero_()
        optimizer.zero_grad()

    def _enable_gradient_checkpointing(self, model):
        """Best effort: the talker transformer is an HF model, so reuse its own API."""
        try:
            target = getattr(getattr(model, "talker", None), "model", None)
            if target is None or not hasattr(target, "gradient_checkpointing_enable"):
                return "unsupported"
            target.gradient_checkpointing_enable()
            if hasattr(target, "config"):
                target.config.use_cache = False
            return "enabled"
        except Exception as e:  # noqa: BLE001 - surface the reason instead of failing the run
            return f"failed: {e}"

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

    def _run_validation(
        self, checkpoint_path, text, speaker, unique_id, epoch, device="xpu", train_model=None
    ):
        """Generate a preview with the just-saved checkpoint.

        Validation loads a *second* copy of the 1.7B model, so the training copy has to
        leave the GPU first; otherwise both live on a 16 GB card at once and the driver
        dies with an access violation (observed: 3 epochs trained fine, then the crash
        hit inside generate_custom_voice).
        """
        parked = False
        if train_model is not None and hasattr(train_model, "to"):
            try:
                train_model.to("cpu")
                self._empty_cache(device)
                parked = True
            except Exception as e:  # noqa: BLE001 - validation is best effort
                logger.warning(f"[Qwen3TTS][train] could not park the training model: {e}")
        try:
            device_map_arg = {"": f"{device}:0"} if device in ("xpu", "cuda") else device
            val_model = Qwen3TTSModel.from_pretrained(
                checkpoint_path, 
                dtype=torch.bfloat16, 
                device_map=device_map_arg,  
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
            self._empty_cache(device)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
        finally:
            if parked:
                try:
                    train_model.to(device)
                    self._empty_cache(device)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[Qwen3TTS][train] could not move the training model back: {e}")
