"""Check source syntax, text encoding and accidental credential exposure."""
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
PRIVATE_PATH = re.compile(r"[A-Z]:[\\/]Users[\\/]", re.I)


def validate():
    issues = []
    count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in (".py", ".md", ".json", ".csv", ".txt", ".yml"):
            continue
        if any(part in path.parts for part in ("__pycache__", "runs", ".git", ".venv")) or path.name == "local_config.py":
            continue
        if "data" in path.relative_to(ROOT).parts and path.suffix == ".json" and path.name != "reference_manifest.json":
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"Non-UTF-8 text: {path.relative_to(ROOT)}")
            continue
        if b"\r" in raw:
            issues.append(f"Non-LF text: {path.relative_to(ROOT)}")
        if SECRET.search(text):
            issues.append(f"Credential-like token: {path.relative_to(ROOT)}")
        if PRIVATE_PATH.search(text):
            issues.append(f"Personal workspace path: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            count += 1
            try:
                ast.parse(text, filename=str(path.relative_to(ROOT)))
                list(tokenize.generate_tokens(io.StringIO(text).readline))
            except (SyntaxError, tokenize.TokenError) as exc:
                issues.append(f"Syntax: {path.relative_to(ROOT)}: {exc}")
    print(json.dumps({"python_files": count, "issues": issues}, indent=2))
    return issues


if __name__ == "__main__":
    sys.exit(bool(validate()))
