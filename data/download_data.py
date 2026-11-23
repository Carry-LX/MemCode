"""Optional official-source download. Edit DATASETS here; no CLI options."""
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

DATASETS = {
    "locomo_10.json": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "longmemeval_s.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json",
}
DATA_DIR = Path(__file__).resolve().parent


def download():
    reference = {r["filename"]: r for r in json.loads((DATA_DIR / "reference_manifest.json").read_text(encoding="utf-8"))}
    for name, url in DATASETS.items():
        target = DATA_DIR / name
        if target.exists():
            print(f"Keeping existing dataset: {target}")
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=DATA_DIR, prefix="download_", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            rows = json.loads(temporary.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError("Expected a nonempty benchmark JSON array")
            sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
            temporary.rename(target)
            print(f"Downloaded {name}: {len(rows)} rows, SHA256={sha256}")
            if name in reference and sha256 != reference[name]["sha256"]:
                print("Dataset SHA256 differs from reference_manifest.json.")
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    download()
