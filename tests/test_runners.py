"""Exercise all eight real evaluation loops with local data and model/pipeline doubles."""
import asyncio
import contextlib
import importlib
import io
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import settings
from src.core.memory_structures import AtomicMemoryUnit, BaseMemoryUnit

EVENTS = []


class FakeLLM:
    def __init__(self, api_key=None, model=None, base_url=None):
        self.model = model
        self.call_count = 0
        self.tokens = 0

    def get_completion(self, prompt):
        self.tokens += 1
        EVENTS.append(("answer", self.model))
        return "Paris"

    def get_total_usage(self):
        return {"total_tokens": self.tokens * 2, "total_prompt_tokens": self.tokens,
                "total_completion_tokens": self.tokens}


class FakeJudge:
    STANDARD_TASK = "standard"
    TEMPORAL_REASONING = "temporal"
    KNOWLEDGE_UPDATE = "update"
    PREFERENCE_TASK = "preference"
    ABSTENTION_TASK = "abstention"

    def __init__(self, llm_interface):
        self.llm = llm_interface

    def evaluate(self, *args, **kwargs):
        self.llm.get_completion("judge")
        return True


class FakeEmbedder:
    def __init__(self, **kwargs):
        pass


class FakeVectors:
    def __init__(self, storage_path):
        self.indexes = {}

    def create_or_load_index(self, name, dimension):
        self.indexes[name] = {"index": types.SimpleNamespace(ntotal=1), "id_map": {0: "fact"}}


class FakePipeline:
    def __init__(self, splitter, embedder, llm, meta_db, vec_db, vq_model):
        self.db, self.llm = meta_db, llm

    async def build(self, **kwargs):
        base = BaseMemoryUnit(text="Maya visited Paris", timestamp=datetime(2024, 1, 1), id="base")
        fact = AtomicMemoryUnit(id="fact", fact_text="Maya visited Paris", source_base_ids=["base"], code_indices=[7])
        self.db.add_base_memory(base)
        self.db.add_atomic_memory(fact)
        self.db.update_stats([fact])
        EVENTS.append(("build", self.llm.model))
        self.llm.get_completion("write")

    async def process_locomo_data_async(self, *args, **kwargs):
        await self.build()

    async def process_longmemeval_data_async(self, *args, **kwargs):
        await self.build()

    def shutdown_databases(self):
        self.db.close()


class FakeConsolidator:
    def __init__(self, **kwargs):
        self.llm = kwargs["llm"]

    async def execute_evolution_async(self):
        EVENTS.append(("consolidate", self.llm.model))
        self.llm.get_completion("consolidate")


class FakeQuery:
    def __init__(self, *args, **kwargs):
        pass

    def query(self, *args, **kwargs):
        return {"final_prompt": "Where did Maya go?", "trace_info": {"offline_double": True}}


def module(name, **attributes):
    value = types.ModuleType(name)
    value.__dict__.update(attributes)
    return value


class RunnerTests(unittest.TestCase):
    def test_eight_real_runner_loops_and_consolidation_flags(self):
        doubles = {
            "src.utils.vqvae_inference_v10": module("vq", ModelInference=lambda **kwargs: object()),
            "src.utils.embedding_bge": module("embedding", EmbeddingModel=FakeEmbedder),
            "src.utils.llm_interface": module("llm", OpenAIInterface=FakeLLM),
            "src.utils.llm_judge": module("judge", LLMJudge=FakeJudge),
            "src.core.vector_db_handler": module("vectors", VectorDBHandler=FakeVectors),
            "src.processing.text_splitter": module("splitter", ConversationSplitter=lambda: object()),
            "nltk": module("nltk", data=types.SimpleNamespace(find=lambda resource: resource)),
        }
        for dataset in ("locomo", "longmemeval"):
            doubles[f"src.processing.{dataset}_memory_pipeline"] = module("pipeline", AsyncProcessingPipeline=FakePipeline)
            doubles[f"src.processing.{dataset}_memory_evolver"] = module("evolver", MemoryConsolidator=FakeConsolidator)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, doubles), patch("release_runtime.selected_query_engine", return_value=FakeQuery):
            root = Path(temporary)
            checkpoint = root / "weights.pt"
            checkpoint.write_bytes(b"offline-double")
            locomo = root / "locomo.json"
            locomo.write_text(json.dumps([{"conversation": {}, "qa": [{"question": "Where did Maya go?", "answer": "Paris", "category": 1}]}]), encoding="utf-8")
            longmem = root / "longmem.json"
            longmem.write_text(json.dumps([{"question_id": "example", "question": "Where did Maya go?", "answer": "Paris",
                                           "question_type": "single-session-user", "question_date": "2024/01/02 (Tue) 00:00",
                                           "haystack_dates": ["2024/01/01 (Mon) 00:00"], "haystack_sessions": [[]]}]), encoding="utf-8")
            for name in settings.PROFILES:
                with self.subTest(profile=name), patch.object(settings, "VQ_WEIGHTS_PATH", checkpoint):
                    EVENTS.clear()
                    config = replace(settings.make_config(name), data_path=locomo if name.startswith("locomo_") else longmem,
                                     output_dir=root / name, max_rows=1, api_key="offline-double")
                    settings._ACTIVE_CONFIG = config
                    module_name = "experiments." + config.dataset + "_runner"
                    sys.modules.pop(module_name, None)
                    runner = importlib.import_module(module_name)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            asyncio.run(runner.main_async())
                        self.assertTrue((config.output_dir / "run_config.json").is_file())
                        self.assertTrue((config.output_dir / f"eval_results_{config.dataset}_persist.json").is_file())
                        self.assertTrue((config.output_dir / f"eval_performance_{config.dataset}.json").is_file())
                        self.assertEqual(len(list(config.output_dir.rglob("qa_trace_*.json"))), 1)
                        self.assertEqual(sum(event[0] == "consolidate" for event in EVENTS), int(config.consolidation))
                        self.assertIn(("build", config.memory_model), EVENTS)
                        self.assertIn(("answer", config.qa_model), EVENTS)
                        if config.dataset == "locomo":
                            with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                                asyncio.run(runner.main_async())
                    finally:
                        sys.modules.pop(module_name, None)
            settings._ACTIVE_CONFIG = None

            for family in ("gpt", "qwen"):
                with self.subTest(existing_memory=family), patch.object(settings, "VQ_WEIGHTS_PATH", checkpoint):
                    source = root / f"longmemeval_{family}_baseline"
                    vector_dir = source / "row_0_data/vector_stores"
                    vector_dir.mkdir(exist_ok=True)
                    (vector_dir / "atomic_memory.faiss").write_bytes(b"offline-vector-double")
                    (vector_dir / "atomic_memory_map.json").write_text('{"0": "fact"}', encoding="utf-8")
                    source_bytes = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
                    name = f"longmemeval_{family}_consolidated"
                    config = replace(settings.make_config(name, existing=True), data_path=longmem, source_dir=source,
                                     output_dir=root / (name + "_existing"), max_rows=1, api_key="offline-double")
                    settings._ACTIVE_CONFIG = config
                    EVENTS.clear()
                    module_name = "experiments.existing_memory_runner"
                    sys.modules.pop(module_name, None)
                    runner = importlib.import_module(module_name)
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            asyncio.run(runner.main_async())
                        self.assertTrue((config.output_dir / "eval_results_longmemeval_skipped.json").is_file())
                        self.assertTrue((config.output_dir / "eval_performance_longmemeval_skipped.json").is_file())
                        self.assertEqual(len(list(config.output_dir.rglob("qa_trace_*.json"))), 1)
                        self.assertNotIn("build", [event[0] for event in EVENTS])
                        self.assertIn(("consolidate", config.memory_model), EVENTS)
                        self.assertEqual(source_bytes, {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()})
                        (vector_dir / "atomic_memory.faiss").unlink()
                        with self.assertRaises(FileNotFoundError), contextlib.redirect_stdout(io.StringIO()):
                            asyncio.run(runner.main_async())
                    finally:
                        sys.modules.pop(module_name, None)
                        settings._ACTIVE_CONFIG = None


if __name__ == "__main__":
    unittest.main()
