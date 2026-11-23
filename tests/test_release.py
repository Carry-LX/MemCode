"""Offline regression tests using real storage/retrieval code and deterministic doubles."""
import ast
import importlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import settings
from release_runtime import preflight, retrieval_module, safe_remove_run_directory, write_run_manifest
from src.core.memory_structures import AtomicMemoryUnit, BaseMemoryUnit
from src.core.metadata_db_handler import MetadataDBHandler

ROOT = Path(__file__).resolve().parents[1]


def load_retriever_without_model_dependencies():
    tree = ast.parse((ROOT / "src/retrieval/codebook_retriever.py").read_text(encoding="utf-8"))
    # Model libraries are not needed for retrieval over injected fake encoders.
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("src.")]
    namespace = {"ModelInference": object, "EmbeddingModel": object, "MetadataDBHandler": MetadataDBHandler,
                 "VectorDBHandler": object, "BaseMemoryUnit": BaseMemoryUnit, "AtomicMemoryUnit": AtomicMemoryUnit}
    exec(compile(tree, "codebook_retriever.py", "exec"), namespace)
    return namespace["QueryEngine"]


class FakeEmbedder:
    def embed(self, texts, is_query=False):
        assert is_query
        return [[1.0, 0.0] for _ in texts]


class FakeVQ:
    def get_indices(self, texts):
        return [[7] for _ in texts]


class FakeVectors:
    def search(self, index_name, vector, k):
        return [(f"fact-{i}", 0.1) for i in range(20)]


class FakeMetadata:
    def __init__(self):
        self.facts = {f"fact-{i}": AtomicMemoryUnit(id=f"fact-{i}", fact_text=f"Fact {i}",
                                                    source_base_ids=[f"base-{i}"], code_indices=[7]) for i in range(50)}

    def get_stats(self):
        return {"total_docs": 1000, "code_freq": {7: 1}, "co_occurrence": {}}

    def get_atomic_memory_by_id(self, fact_id):
        return self.facts.get(fact_id)

    def get_atomic_ids_by_code(self, code):
        return list(self.facts) if code == 7 else []

    def get_base_memory_by_id(self, base_id):
        return BaseMemoryUnit(id=base_id, text=base_id, timestamp=datetime(2024, 1, 1), metadata={"speaker": "user"})


class ReleaseTests(unittest.TestCase):
    def test_all_profiles_have_distinct_paths_and_correct_dispatch(self):
        configs = [settings.make_config(p) for p in settings.PROFILES]
        self.assertEqual(len({c.output_dir for c in configs}), 8)
        for c in configs:
            self.assertEqual(c.memory_model, c.qa_model)
            self.assertEqual(c.judge_model, settings.JUDGE_MODEL)
            self.assertEqual(preflight(c, require_assets=False), [])
            self.assertTrue((ROOT / f"run_{c.profile}.py").is_file())
            module = retrieval_module(c)
            if c.consolidation and c.dataset == "longmemeval":
                self.assertEqual(module.endswith("_qwen"), c.family == "qwen")

    def test_background_and_model_specific_thresholds(self):
        def constants(name):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            return {n.target.id: ast.literal_eval(n.value) for n in ast.walk(tree)
                    if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and isinstance(n.value, ast.Constant)}
        gpt = constants("src/retrieval/longmemeval_evolution_retriever.py")
        qwen = constants("src/retrieval/longmemeval_evolution_retriever_qwen.py")
        self.assertEqual((gpt["FINAL_ATOMIC_LIMIT"], gpt["FINAL_BASE_LIMIT"]), (40, 10))
        self.assertEqual((qwen["FINAL_ATOMIC_LIMIT"], qwen["FINAL_BASE_LIMIT"]), (24, 6))
        tree = ast.parse((ROOT / "src/processing/longmemeval_memory_pipeline.py").read_text(encoding="utf-8"))
        windows = [n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                   and any(isinstance(t, ast.Name) and t.id == "MAX_CONTEXT_WINDOW" for t in n.targets)]
        self.assertEqual(windows, [5, 5])

    def test_twenty_plus_twenty_retrieval_and_source_context_limit(self):
        engine_type = load_retriever_without_model_dependencies()
        engine = engine_type(FakeEmbedder(), FakeMetadata(), FakeVectors(), object(), FakeVQ())
        candidates = engine._retrieve_atomic_candidates(["test question"], 20)
        self.assertEqual(len(candidates), 40)
        self.assertEqual(len({f.id for f in candidates}), 40)
        engine.FINAL_BASE_LIMIT = 10
        result = engine.query("test question", k=20, total_atomic_count=50)
        prompt = result["final_prompt"]
        source_context = prompt.split("[SOURCE CONVERSATION (Original Context)]:")[1].split("[INSTRUCTIONS")[0]
        self.assertEqual(sum(line.startswith("- [") for line in source_context.splitlines()), 10)

    def test_sqlite_roundtrip_and_codebook_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "metadata.db")
            db = MetadataDBHandler(db_path)
            base = BaseMemoryUnit(text="Maya visited Paris", id="base-1", timestamp=datetime(2024, 1, 1))
            fact = AtomicMemoryUnit(fact_text="Maya visited Paris", source_base_ids=["base-1"], id="fact-1", code_indices=[7, 9])
            db.add_base_memory(base)
            db.add_atomic_memory(fact)
            db.update_stats([fact])
            db.close()
            restored = MetadataDBHandler(db_path)
            self.assertEqual(restored.get_atomic_memory_by_id("fact-1").source_base_ids, ["base-1"])
            self.assertEqual(restored.get_atomic_ids_by_code(7), ["fact-1"])
            self.assertEqual(restored.get_stats()["total_docs"], 1)
            restored.close()

    def test_resume_manifest_rejects_changed_model_and_omits_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, checkpoint = root / "data.json", root / "model.pt"
            data.write_text("[]", encoding="utf-8")
            checkpoint.write_bytes(b"test-checkpoint")
            config = replace(settings.make_config("longmemeval_gpt_baseline"), output_dir=root, data_path=data, api_key="private-value")
            with patch.object(settings, "VQ_WEIGHTS_PATH", checkpoint):
                write_run_manifest(config)
                self.assertNotIn("private-value", (root / "run_config.json").read_text())
                with self.assertRaises(ValueError):
                    write_run_manifest(replace(config, qa_model="different-model"))

    def test_deletion_guard_and_source_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            child = run / "row_0_data"
            child.mkdir()
            safe_remove_run_directory(child, run)
            self.assertFalse(child.exists())
            with self.assertRaises(ValueError):
                safe_remove_run_directory(root, run)
            config = replace(settings.make_config("longmemeval_gpt_consolidated"), output_dir=run, source_dir=root)
            with self.assertRaises(ValueError):
                preflight(config, require_assets=False)

    def test_python_syntax_utf8_lf_and_local_imports(self):
        for file in ROOT.rglob("*.py"):
            if "__pycache__" in file.parts or file.name == "local_config.py":
                continue
            raw = file.read_bytes()
            self.assertNotIn(b"\r", raw, file)
            tree = ast.parse(raw.decode("utf-8"), filename=str(file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in ("src", "experiments", "vqvae_training"):
                    path = ROOT.joinpath(*node.module.split("."))
                    self.assertTrue(path.with_suffix(".py").exists() or (path / "__init__.py").exists(), (file, node.module))


if __name__ == "__main__":
    unittest.main()
