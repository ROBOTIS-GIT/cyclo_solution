#!/usr/bin/env bash
set -euo pipefail

task_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_project_dir"

task_mode="${1:---check}"
if (( $# > 1 )) || [[ "$task_mode" != --check && "$task_mode" != --fix ]]; then
    printf 'Usage: %s [--check|--fix]\n' "$0" >&2
    exit 2
fi

for task_tool in clang-format ruff cmake-format; do
    if ! command -v "$task_tool" >/dev/null 2>&1; then
        printf 'Missing tool: %s. Install requirements-lint.txt in a Python virtual environment.\n' \
            "$task_tool" >&2
        exit 1
    fi
done

mapfile -d '' -t task_cpp_files < <(python3 - <<'PY'
from pathlib import Path
import sys

for directory in ("src", "include", "test"):
    for path in sorted(Path(directory).rglob("*")):
        if path.suffix in {".cpp", ".hpp"}:
            sys.stdout.buffer.write(str(path).encode() + b"\0")
PY
)

if [[ "$task_mode" == --fix ]]; then
    clang-format -i "${task_cpp_files[@]}"
    ruff format .
    ruff check --fix .
    ruff format .
    cmake-format -i CMakeLists.txt
fi

clang-format --dry-run --Werror "${task_cpp_files[@]}"
ruff check .
ruff format --check .
cmake-format --check CMakeLists.txt
bash -n docker/container.sh docker/entrypoint.sh scripts/lint.sh scripts/test_perception.sh scripts/build_ai_engine.sh test/run_isolated.sh

python3 - <<'PY'
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

for directory in ("config", "docker"):
    for path in Path(directory).glob("*.yaml"):
        yaml.safe_load(path.read_text())
for path in Path("config").glob("*.json"):
    json.loads(path.read_text())
for path in (Path("package.xml"), Path("planning_request_adapters.xml"), Path("constraint_samplers.xml")):
    ET.parse(path)
print("PASS: C++, Python and CMake style; shell and configuration syntax.")
PY
