"""Consolidate a copy of an existing LongMemEval memory store."""
from release_runtime import launch
from settings import EXISTING_MEMORY_PROFILE

if __name__ == "__main__":
    launch(EXISTING_MEMORY_PROFILE, existing=True)
