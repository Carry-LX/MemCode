"""Edit explicit variables here; no command-line experiment parameters are needed."""
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GPT_MODEL = "gpt-4o-mini"
QWEN_MODEL = "qwen3-30b-a3b-instruct-2507"
JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
T5_MODEL_PATH = "t5-base"  # Hugging Face ID or a local directory.
VQ_WEIGHTS_PATH = ROOT / "models/T5/vqvae_t5_v10_epoch10.pt"
DEVICE = "cuda"  # Explicitly choose "cpu" if required.
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "runs"
RUN_LABEL = "run_001"  # Use a new label for each independent repetition.
MAX_ROWS = None  # Set to 1 for a paid end-to-end pilot; None evaluates all rows.
EXISTING_MEMORY_SOURCE_DIR = None  # Directory containing row_0_data/, etc.
EXISTING_MEMORY_PROFILE = "longmemeval_gpt_consolidated"
RETRIEVER_OVERRIDE = None  # Optional: "src.retrieval.bm25_baseline_retriever".

# Private overrides are never included in the release archive or Git.
private_path = ROOT / "local_config.py"
if private_path.is_file():
    spec = importlib.util.spec_from_file_location("memcode_local_config", private_path)
    private = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(private)
    for name in ("API_KEY", "API_BASE_URL", "GPT_MODEL", "QWEN_MODEL", "JUDGE_MODEL",
                 "EMBEDDING_MODEL", "T5_MODEL_PATH", "VQ_WEIGHTS_PATH", "DEVICE", "DATA_DIR",
                 "OUTPUT_DIR", "RUN_LABEL", "MAX_ROWS", "EXISTING_MEMORY_SOURCE_DIR",
                 "EXISTING_MEMORY_PROFILE", "RETRIEVER_OVERRIDE"):
        if hasattr(private, name):
            globals()[name] = getattr(private, name)

PROFILES = {
    "locomo_gpt_baseline": ("locomo", "gpt", False),
    "locomo_qwen_baseline": ("locomo", "qwen", False),
    "locomo_gpt_consolidated": ("locomo", "gpt", True),
    "locomo_qwen_consolidated": ("locomo", "qwen", True),
    "longmemeval_gpt_baseline": ("longmemeval", "gpt", False),
    "longmemeval_qwen_baseline": ("longmemeval", "qwen", False),
    "longmemeval_gpt_consolidated": ("longmemeval", "gpt", True),
    "longmemeval_qwen_consolidated": ("longmemeval", "qwen", True),
}


def root_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


@dataclass(frozen=True)
class RunConfig:
    profile: str
    dataset: str
    family: str
    consolidation: bool
    data_path: Path
    output_dir: Path
    memory_model: str
    qa_model: str
    judge_model: str
    api_key: str = field(repr=False)
    base_url: str
    embedding_model: str
    device: str
    max_rows: object
    source_dir: object = None
    retriever_override: object = None


def make_config(profile, existing=False):
    dataset, family, consolidation = PROFILES[profile]
    if existing and (dataset != "longmemeval" or not consolidation):
        raise ValueError("Existing-memory workflow requires a consolidated LongMemEval profile")
    if not RUN_LABEL or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in RUN_LABEL):
        raise ValueError("RUN_LABEL must contain ASCII letters, digits, underscores or hyphens")
    if MAX_ROWS is not None and (isinstance(MAX_ROWS, bool) or not isinstance(MAX_ROWS, int) or MAX_ROWS < 1):
        raise ValueError("MAX_ROWS must be None or a positive integer")
    if DEVICE not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be cpu or cuda")
    model = GPT_MODEL if family == "gpt" else QWEN_MODEL
    suffix = "_existing" if existing else ""
    return RunConfig(profile, dataset, family, consolidation,
                     root_path(DATA_DIR) / ("locomo_10.json" if dataset == "locomo" else "longmemeval_s.json"),
                     root_path(OUTPUT_DIR) / f"{profile}{suffix}" / RUN_LABEL,
                     model, model, JUDGE_MODEL, API_KEY, API_BASE_URL,
                     EMBEDDING_MODEL, DEVICE, MAX_ROWS,
                     root_path(EXISTING_MEMORY_SOURCE_DIR) if existing and EXISTING_MEMORY_SOURCE_DIR else None,
                     RETRIEVER_OVERRIDE)


_ACTIVE_CONFIG = None


def active_config():
    if _ACTIVE_CONFIG is None:
        raise RuntimeError("Start an experiment through a run_*.py entry script")
    return _ACTIVE_CONFIG
