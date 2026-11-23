"""Package source, configuration, run instructions and tests."""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT.parent / "outputs/releases/memcode_github_code_only_20261010_checked.zip"
ROOT_FILES = {"README.md", "README_zh.md", "settings.py", "local_config.example.py", "release_runtime.py",
              "check_environment.py", "requirements.txt", "requirements-analysis.txt", ".gitignore",
              ".gitattributes", "LICENSE"}
SOURCE_DIRS = {"src", "experiments", "vqvae_training", "tests", "tools", ".github"}
EXTENSIONS = {".py", ".md", ".json", ".csv", ".txt", ".yml"}


def included(path):
    relative = path.relative_to(ROOT)
    if any(part in ("__pycache__", ".git", ".venv", "runs") for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES or (relative.name.startswith("run_") and relative.suffix == ".py")
    if relative.parts[0] in SOURCE_DIRS:
        return path.suffix in EXTENSIONS
    return relative.as_posix() in {"data/README.md", "data/reference_manifest.json", "data/download_data.py", "models/README.md"}


def package():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_release import validate
    if validate():
        raise RuntimeError("Source validation failed")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and included(path))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Archive already exists: {OUTPUT_PATH}")
    with zipfile.ZipFile(OUTPUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, "memcode/" + file.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(OUTPUT_PATH) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        assert len(archive.namelist()) == len(files)
        assert not any("local_config.py" in name or "__pycache__" in name or name.endswith((".pt", ".db", ".faiss")) for name in archive.namelist())
    print(f"Packaged {len(files)} files: {OUTPUT_PATH}")
    print(f"SHA256={hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    package()
