"""Offline checks for documented dispatch, prompts and asset/retry boundaries."""
import ast
import contextlib
import importlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import settings
from release_runtime import selected_query_engine, retrieval_module
from src.utils.llm_judge import LLMJudge

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def test_profile_modules_and_source_caps_match_run_notes(self):
        for profile in settings.PROFILES:
            with self.subTest(profile=profile):
                config = settings.make_config(profile)
                runner_path = ROOT / "experiments" / (config.dataset + "_runner.py")
                parsed = ast.parse(runner_path.read_text(encoding="utf-8"))
                imports = {node.module for node in parsed.body if isinstance(node, ast.ImportFrom)}
                self.assertIn(f"src.processing.{config.dataset}_memory_pipeline", imports)
                self.assertIn(f"src.processing.{config.dataset}_memory_evolver", imports)
                self.assertTrue(any(isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                                    and node.test.id == "ENABLE_CONSOLIDATION"
                                    and any(isinstance(child, ast.Attribute) and child.attr == "execute_evolution_async"
                                            for child in ast.walk(node)) for node in ast.walk(parsed)))
                base = type("BaseQuery", (), {"FINAL_BASE_LIMIT": 25})
                selected = types.SimpleNamespace(QueryEngine=base)
                with patch.object(settings, "_ACTIVE_CONFIG", config), patch("release_runtime.importlib.import_module", return_value=selected) as imported:
                    engine = selected_query_engine()
                    imported.assert_called_once_with(retrieval_module(config))
                    if not config.consolidation:
                        self.assertEqual(engine.FINAL_BASE_LIMIT, 10 if config.dataset == "longmemeval" else 25)
                    else:
                        self.assertIs(engine, base)

    def test_evaluator_duration_protocol_is_explicit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            judge = LLMJudge(types.SimpleNamespace(get_completion=lambda prompt: "yes"))
        prompt = judge._build_temporal_prompt("How many days?", "2", "3", query_date="2024-01-03")
        self.assertIn("+/- 1 unit difference for durations", prompt)
        self.assertIn("Reference Time: 2024-01-03", prompt)
        self.assertIn("[Ground Truth]: 2", prompt)
        self.assertIn("[Model Answer]: 3", prompt)

    def test_missing_checkpoint_fails_before_model_initialization(self):
        tree = ast.parse((ROOT / "src/utils/vqvae_inference_v10.py").read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        namespace = {"Path": Path, "torch": None}
        exec(compile(tree, "vqvae_inference_v10.py", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                namespace["ModelInference"](weights_path=Path(temporary) / "absent.pt")

    def test_provider_rejection_retains_original_evidence(self):
        prompts = []

        def reject(**kwargs):
            prompts.append(kwargs["messages"][-1]["content"])
            raise RuntimeError("inappropriate content")

        fake_sdk = types.ModuleType("openai")
        fake_sdk.OpenAI = lambda **kwargs: types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=reject)))
        module_name = "src.utils.llm_interface"
        saved = sys.modules.pop(module_name, None)
        try:
            with patch.dict(sys.modules, {"openai": fake_sdk}), contextlib.redirect_stdout(io.StringIO()):
                module = importlib.import_module(module_name)
                llm = module.OpenAIInterface(api_key="offline-value")
                original = "[Background] Synthetic context.\n[1] Synthetic target."
                with patch.object(module.time, "sleep") as wait, self.assertRaises(RuntimeError):
                    llm.get_completion(original)
                self.assertEqual(prompts, [original])
                wait.assert_not_called()
        finally:
            sys.modules.pop(module_name, None)
            if saved is not None:
                sys.modules[module_name] = saved


if __name__ == "__main__":
    unittest.main()
