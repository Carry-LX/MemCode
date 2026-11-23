"""Read-only checks: no API calls, model downloads or benchmark evaluation."""
from release_runtime import preflight
from settings import PROFILES, make_config

if __name__ == "__main__":
    for name in PROFILES:
        issues = preflight(make_config(name))
        print(f"{name}: {'ready' if not issues else 'not ready'}")
        for issue in issues:
            print(f"  - {issue}")
