"""Loader regression tests with simulated runtimes; no CUDA/model downloads."""
import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class CudaCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.torch = types.ModuleType("torch")
        self.torch.float32 = "float32"
        self.torch.float16 = "float16"
        self.torch.bfloat16 = "bfloat16"
        self.torch.__version__ = "test-build"
        self.torch.version = types.SimpleNamespace(cuda="test-cuda")
        self.torch.cuda = Mock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.get_device_capability.return_value = (6, 1)
        self.torch.cuda.get_device_name.return_value = "NVIDIA GeForce GTX 1080 Ti"
        self.torch.cuda.get_arch_list.return_value = ["sm_80", "sm_90"]
        self.torch.zeros = Mock()
        self.torch.backends = types.SimpleNamespace(mps=Mock(is_available=Mock(return_value=False)))
        package = types.ModuleType("qwen_test_nodes")
        package.__path__ = [str(ROOT)]
        transformers = types.ModuleType("transformers")
        transformers.pytorch_utils = types.ModuleType("transformers.pytorch_utils")
        folders = types.ModuleType("folder_paths")
        folders.models_dir = self.temp.name
        folders.__file__ = str(Path(self.temp.name) / "folder_paths.py")
        folders.add_model_folder_path = Mock()
        folders.get_folder_paths = Mock(return_value=[])
        comfy = types.ModuleType("comfy")
        comfy.model_management = Mock()
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.ProgressBar = Mock()
        qwen = types.ModuleType("qwen_tts")
        qwen.Qwen3TTSModel = Mock()
        qwen.VoiceClonePromptItem = Mock()
        modules = {
            "qwen_test_nodes": package, "torch": self.torch,
            "numpy": types.ModuleType("numpy"), "folder_paths": folders,
            "comfy": comfy, "comfy.utils": comfy_utils,
            "transformers": transformers,
            "transformers.pytorch_utils": transformers.pytorch_utils,
            "qwen_tts": qwen,
        }
        self.addCleanup(setattr, sys, "path", sys.path[:])
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        spec = importlib.util.spec_from_file_location("qwen_test_nodes.nodes", ROOT / "nodes.py")
        self.nodes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.nodes)
        self.nodes.check_and_download_tokenizer = Mock()
        self.nodes.apply_qwen3_patches = Mock()
        self.nodes.check_attention_implementation = Mock(return_value=["sage_attn", "flash_attn", "sdpa", "eager"])
        self.loader = qwen.Qwen3TTSModel.from_pretrained

    def load(self, device="cuda", precision="bf16", attention="auto"):
        return self.nodes.load_qwen_model("Base", "0.6B", device, precision,
                                          attention, custom_model_path=self.temp.name)

    def test_import_failure_retains_cause_and_stops_before_runtime_work(self):
        import builtins

        real_import = builtins.__import__
        for error in (ModuleNotFoundError("No module named 'librosa'", name="librosa"),
                      ImportError("cannot import name 'Example' from 'dependency'")):
            with self.subTest(error=str(error)):
                attempts = []

                def failing_import(name, *args, **kwargs):
                    if name == "qwen_tts":
                        attempts.append(name)
                        raise error
                    return real_import(name, *args, **kwargs)

                spec = importlib.util.spec_from_file_location("qwen_test_nodes.nodes", ROOT / "nodes.py")
                nodes = importlib.util.module_from_spec(spec)
                output = io.StringIO()
                with patch("builtins.__import__", side_effect=failing_import), \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    spec.loader.exec_module(nodes)
                self.assertEqual(attempts, ["qwen_tts"])
                self.assertIsNone(nodes.Qwen3TTSModel)
                self.assertIsNone(nodes.VoiceClonePromptItem)
                self.assertIn(str(error), output.getvalue())
                self.assertIn(sys.executable, output.getvalue())
                self.assertIn(str(ROOT / "requirements.txt"), output.getvalue())
                with patch.object(nodes, "check_and_download_tokenizer") as download, \
                        patch.object(nodes, "get_attention_implementation") as attention:
                    with self.assertRaises(RuntimeError) as raised:
                        nodes.load_qwen_model("Base", "0.6B", "auto", "bf16")
                    self.assertIs(raised.exception.__cause__, error)
                    self.assertIn(str(error), str(raised.exception))
                    self.assertIn("Restart ComfyUI", str(raised.exception))
                    download.assert_not_called()
                    attention.assert_not_called()
                self.assertEqual(nodes._MODEL_CACHE, {})

    def test_windows_install_hint_quotes_runtime_and_requirements_paths(self):
        with patch.object(self.nodes.os, "name", "nt"), \
                patch.object(sys, "executable", r"C:\Comfy Desktop\env\python.exe"), \
                patch.object(self.nodes, "current_dir", "C:/Custom Nodes/Qwen TTS"):
            message = self.nodes._qwen_import_diagnostic(ImportError("missing dependency"))
        self.assertIn('"C:\\Comfy Desktop\\env\\python.exe" -m pip install -r "C:/Custom Nodes/Qwen TTS/requirements.txt"', message)
        self.assertIn("cmd.exe", message)
        self.assertIn("uv pip install --python", message)

    def test_missing_bundled_package_has_restore_guidance(self):
        with patch.object(self.nodes.os.path, "isdir", return_value=False):
            message = self.nodes._qwen_import_diagnostic(ImportError("missing qwen_tts"))
        self.assertIn("Restore the complete custom node repository", message)
        self.assertNotIn("pip install", message)

    def test_pascal_uses_fp32_eager_for_all_attention_choices(self):
        for attention in self.nodes.ATTENTION_OPTIONS:
            with self.subTest(attention=attention):
                self.nodes._MODEL_CACHE.clear()
                self.load(attention=attention)
                self.assertEqual(self.loader.call_args.kwargs, {
                    "device_map": "cuda", "dtype": "float32", "attn_implementation": "eager"})
        self.nodes.check_attention_implementation.assert_not_called()

    def test_auto_device_resolves_before_compatibility_check(self):
        self.load(device="auto")
        self.assertEqual(self.loader.call_args.kwargs["attn_implementation"], "eager")
        self.assertEqual(self.loader.call_args.kwargs["dtype"], "float32")

    def test_effective_configuration_shares_cache(self):
        first = self.load()
        second = self.load(precision="fp32", attention="eager")
        self.assertIs(first, second)
        self.loader.assert_called_once()
        self.torch.zeros.assert_called_once()

    def test_missing_kernel_fails_before_download_or_load(self):
        error = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        self.torch.zeros.return_value.add_.side_effect = error
        with self.assertRaisesRegex(RuntimeError, "GTX 1080 Ti.*sm_61.*device='cpu'") as raised:
            self.load()
        self.assertIs(raised.exception.__cause__, error)
        self.loader.assert_not_called()
        self.nodes.check_and_download_tokenizer.assert_not_called()
        self.assertEqual(self.nodes._MODEL_CACHE, {})

    def test_async_kernel_failure_is_caught(self):
        self.torch.cuda.synchronize.side_effect = RuntimeError("CUDA error: invalid device function")
        with self.assertRaisesRegex(RuntimeError, "compiled architectures"):
            self.load()
        self.loader.assert_not_called()

    def test_other_probe_errors_are_preserved(self):
        error = RuntimeError("CUDA out of memory")
        self.torch.zeros.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.load()
        self.assertIs(raised.exception, error)
        self.loader.assert_not_called()

    def test_arch_list_is_diagnostic_not_an_exact_match_requirement(self):
        self.load()
        self.loader.assert_called_once()  # sm_61 absent, but the execution probe succeeded.
        self.torch.cuda.synchronize.assert_called_once()

    def test_modern_cuda_keeps_bf16_and_attention_priority(self):
        self.torch.cuda.get_device_capability.return_value = (8, 6)
        self.assertEqual(self.nodes.get_attention_implementation("auto", "cuda"), "sage_attn")
        self.load(attention="sdpa")
        self.assertEqual(self.loader.call_args.kwargs["dtype"], "bfloat16")
        self.assertEqual(self.loader.call_args.kwargs["attn_implementation"], "sdpa")

    def test_non_cuda_devices_do_not_probe_cuda(self):
        for device in ("cpu", "mps", "xpu"):
            with self.subTest(device=device):
                self.load(device=device, precision="fp32", attention="eager")
        self.torch.cuda.get_device_capability.assert_not_called()
        self.torch.zeros.assert_not_called()

    def test_sdpa_failure_retries_explicit_eager(self):
        self.torch.cuda.get_device_capability.return_value = (8, 6)
        self.loader.side_effect = [ValueError("SDPA unavailable"), Mock()]
        self.load(attention="sdpa")
        self.assertEqual([c.kwargs["attn_implementation"] for c in self.loader.call_args_list], ["sdpa", "eager"])

    def test_load_kernel_failure_does_not_retry(self):
        self.torch.cuda.get_device_capability.return_value = (8, 6)
        self.loader.side_effect = RuntimeError("CUDA error: no kernel image is available for execution on the device")
        with self.assertRaisesRegex(RuntimeError, "Changing attention alone"):
            self.load(attention="sdpa")
        self.loader.assert_called_once()
        self.assertEqual(self.nodes._MODEL_CACHE, {})

    def test_sage_kernel_failure_does_not_retry(self):
        self.torch.cuda.get_device_capability.return_value = (8, 6)
        sage = types.ModuleType("sageattention")
        sage.sageattn = Mock()
        self.loader.side_effect = RuntimeError("CUDA error: invalid device function")
        with patch.dict(sys.modules, {"sageattention": sage}):
            with self.assertRaisesRegex(RuntimeError, "Changing attention alone"):
                self.load(attention="sage_attn")
        self.loader.assert_called_once()

    def test_sage_import_failure_retries_explicit_eager(self):
        self.torch.cuda.get_device_capability.return_value = (8, 6)
        with patch.dict(sys.modules, {"sageattention": None}):
            self.load(attention="sage_attn")
        self.assertEqual(self.loader.call_args.kwargs["attn_implementation"], "eager")

    def test_eager_failure_is_not_retried(self):
        error = ValueError("Invalid model config")
        self.loader.side_effect = error
        with self.assertRaises(ValueError) as raised:
            self.load()
        self.assertIs(raised.exception, error)
        self.loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
