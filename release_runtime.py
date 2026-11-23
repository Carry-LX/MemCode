"""Profile dispatch, run configuration records and preflight checks."""
import asyncio
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import settings

SOURCE_MEMORY_FILES = ("metadata.db", "metadata.db.stats.json", "vector_stores/atomic_memory.faiss",
                       "vector_stores/atomic_memory_map.json")


def validate_source_memory(config, row_count):
    """Fail before paid evaluation if any expected source row is incomplete."""
    if config.source_dir is None:
        raise ValueError("Existing-memory source is required")
    missing = [str(config.source_dir / f"row_{index}_data" / relative)
               for index in range(row_count) for relative in SOURCE_MEMORY_FILES
               if not (config.source_dir / f"row_{index}_data" / relative).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete source memory; evaluation will not skip rows:\n" + "\n".join(missing[:10]))


def retrieval_module(config):
    if config.retriever_override:
        allowed = {"src.retrieval.bm25_baseline_retriever", "src.retrieval.ablation_vector_top40_retriever",
                   "src.retrieval.ablation_codebook_expansion_retriever",
                   "src.retrieval.ablation_codebook_expansion_retriever_v2",
                   "src.retrieval.ablation_codebook_expansion_retriever_v3",
                   "src.retrieval.idf_retriever", "src.retrieval.codebook_expansion_retriever"}
        if config.retriever_override not in allowed:
            raise ValueError("Unsupported retriever override; see README.md")
        if config.consolidation:
            raise ValueError("Ablation overrides are supported only for baseline profiles")
        return config.retriever_override
    if not config.consolidation:
        return "src.retrieval.codebook_retriever"
    if config.dataset == "locomo":
        return "src.retrieval.locomo_evolution_retriever"
    return "src.retrieval.longmemeval_evolution_retriever" + ("_qwen" if config.family == "qwen" else "")


def selected_query_engine():
    config = settings.active_config()
    base = importlib.import_module(retrieval_module(config)).QueryEngine
    if not config.consolidation and not config.retriever_override:
        class DatasetQueryEngine(base):
            FINAL_BASE_LIMIT = 10 if config.dataset == "longmemeval" else 25
        return DatasetQueryEngine
    return base


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def public_run_record(config):
    record = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items() if key != "api_key"}
    record.update(retrieval_module=retrieval_module(config),
                  dataset_sha256=digest(config.data_path),
                  checkpoint_sha256=digest(settings.root_path(settings.VQ_WEIGHTS_PATH)),
                  t5_model=str(settings.T5_MODEL_PATH), checkpoint=str(settings.root_path(settings.VQ_WEIGHTS_PATH)))
    return record


def write_run_manifest(config):
    record = public_run_record(config)
    path = config.output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
        raise ValueError("Existing results use a different configuration; choose a new RUN_LABEL")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def safe_remove_run_directory(path, run_root):
    target, root = Path(path).resolve(), Path(run_root).resolve()
    if target == root or root not in target.parents:
        raise ValueError("Refusing to remove a directory outside the selected run")
    if Path(path).is_symlink():
        raise ValueError("Refusing to recursively remove a symlink")
    shutil.rmtree(target)


def preflight(config, require_assets=True):
    retrieval_module(config)
    if config.source_dir and (config.source_dir == config.output_dir or config.source_dir in config.output_dir.parents or config.output_dir in config.source_dir.parents):
        raise ValueError("Source and output memory directories must not overlap")
    if os.name == "nt" and not str(config.output_dir).isascii():
        raise ValueError("Choose an ASCII-only OUTPUT_DIR on Windows for the FAISS files")
    if not require_assets:
        return []
    issues = []
    for module in ("torch", "transformers", "sentence_transformers", "faiss", "nltk", "openai", "numpy", "scipy", "sklearn", "sentencepiece"):
        if importlib.util.find_spec(module) is None:
            issues.append(f"Missing Python dependency: {module}")
    if not config.api_key:
        issues.append("API_KEY is empty; edit local_config.py or set OPENAI_API_KEY")
    try:
        if int(importlib.metadata.version("transformers").split(".")[0]) >= 5:
            issues.append("Requires transformers<5; install requirements.txt in an isolated environment")
    except importlib.metadata.PackageNotFoundError:
        pass
    if importlib.util.find_spec("nltk") is not None:
        import nltk
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab/english/"):
            try:
                nltk.data.find(resource)
            except LookupError:
                issues.append(f"NLTK resource missing: {resource}; run python -m nltk.downloader punkt punkt_tab")
    if not config.data_path.is_file():
        issues.append(f"Dataset missing: {config.data_path}")
    else:
        data = json.loads(config.data_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            issues.append("Dataset must be a nonempty JSON array")
        elif config.dataset == "locomo" and not all("conversation" in r and "qa" in r for r in data):
            issues.append("LoCoMo rows require conversation and qa")
        elif config.dataset == "longmemeval" and not all("haystack_sessions" in r and "haystack_dates" in r and "question" in r for r in data):
            issues.append("LongMemEval rows require haystack_sessions, haystack_dates and question")
    if not settings.root_path(settings.VQ_WEIGHTS_PATH).is_file():
        issues.append(f"Trained checkpoint missing: {settings.VQ_WEIGHTS_PATH}")
    if config.source_dir is not None:
        if not config.source_dir.is_dir():
            issues.append(f"Existing-memory source missing: {config.source_dir}")
        elif config.data_path.is_file() and isinstance(data, list):
            try:
                validate_source_memory(config, min(len(data), config.max_rows) if config.max_rows else len(data))
            except FileNotFoundError as exc:
                issues.append(str(exc))
    return issues


def launch(profile, existing=False):
    config = settings.make_config(profile, existing=existing)
    if existing and not config.source_dir:
        raise ValueError("Set EXISTING_MEMORY_SOURCE_DIR in local_config.py")
    issues = preflight(config)
    if issues:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(issues))
    if config.output_dir.exists() and (config.output_dir / "run_config.json").exists():
        if json.loads((config.output_dir / "run_config.json").read_text(encoding="utf-8")) != public_run_record(config):
            raise ValueError("Configuration changed; choose a new RUN_LABEL")
    elif config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError("Unverified existing run directory; choose a new RUN_LABEL")
    settings._ACTIVE_CONFIG = config
    os.chdir(settings.ROOT)
    name = "existing_memory_runner" if existing else config.dataset + "_runner"
    runner = importlib.import_module("experiments." + name)
    asyncio.run(runner.main_async())
