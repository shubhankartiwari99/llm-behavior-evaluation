#!/usr/bin/env python3
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ALLOWED_EMO_FILES = {
    ROOT / "app" / "inference.py",
    ROOT / "app" / "scaffolds.py",
    ROOT / "app" / "policies.py",
    ROOT / "app" / "intent.py",
}

RUNTIME_SCAN_ROOTS = {
    ROOT / "app",
}

RANDOMNESS_PATTERNS = [
    re.compile(r"^\s*import\s+random\b", re.MULTILINE),
    re.compile(r"^\s*from\s+random\s+import\b", re.MULTILINE),
    re.compile(r"numpy\.random"),
    re.compile(r"\btemperature\s*=\s*(?!0\.0\b|0\b|0\.7\b|temperature\b|kwargs\.get\(\"temperature\",\s*0\.7\))[^,\)]+"),
    re.compile(r"\bdo_sample\s*=\s*(?!False\b|True\b|do_sample\b|kwargs\.get\(\"do_sample\",\s*True\))[^,\)]+"),
    re.compile(r"\bsampling\b"),
]

EMO_PHRASES = [
    "I hear you",
    "That sounds",
    "I am here with you",
    "I'm here with you",
    "We can just stay",
    "feels heavy",
    "sounds heavy",
    "overwhelming",
    "exhausting",
    "heavy",
]


def iter_py_files():
    for root in RUNTIME_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if ".git" in path.parts or "venv" in path.parts or "artifacts" in path.parts:
                continue
            yield path


def check_randomness(contents: str, path: Path, failures: list):
    for pattern in RANDOMNESS_PATTERNS:
        if pattern.search(contents):
            failures.append(f"randomness_ban:{path}")
            return


def check_emotional_strings(contents: str, path: Path, failures: list):
    if path in ALLOWED_EMO_FILES:
        return
    for phrase in EMO_PHRASES:
        if phrase in contents:
            failures.append(f"ad_hoc_emotional_string:{path}:{phrase}")
            return


def main():
    failures = []
    for path in iter_py_files():
        contents = path.read_text(encoding="utf-8")
        check_randomness(contents, path, failures)
        check_emotional_strings(contents, path, failures)

    if failures:
        print("PHASE0_GUARDS_FAILED")
        for item in failures:
            print(item)
        sys.exit(2)

    print("PHASE0_GUARDS_OK")


if __name__ == "__main__":
    main()
