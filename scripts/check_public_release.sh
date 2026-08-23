#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f README.md || ! -f LICENSE ]]; then
  echo "[error] run this script from the repository root" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[check] Python syntax"
"${PYTHON_BIN}" -m py_compile $(find scripts -name '*.py' | sort)

echo "[check] required public-release files"
test -s README.md
test -s LICENSE
test -s requirements.txt
test -s requirements-r.txt

echo "[check] obvious secrets and local absolute paths"
if command -v rg >/dev/null 2>&1; then
  if rg -n "(password|passwd|secret|token|api[_-]?key|AWS_|PRIVATE KEY|BEGIN RSA|BEGIN OPENSSH|sk-[A-Za-z0-9])" . \
      --glob '!LICENSE' \
      --glob '!scripts/check_public_release.sh'; then
    echo "[error] possible secret-like strings found" >&2
    exit 1
  fi
  if rg -n "(/Users/|/home/|/work7/|/scratch/|/gpfs/|/lustre/|k_yamada)" README.md ENVIRONMENT.md scripts \
      --glob '!scripts/check_public_release.sh'; then
    echo "[error] possible local absolute paths found" >&2
    exit 1
  fi
else
  echo "[warn] rg not found; skipped text scans"
fi

echo "[ok] public-release checks completed"
