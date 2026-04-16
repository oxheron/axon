#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Axon on AMD (ROCm): install official vLLM ROCm prebuilt wheel + Axon Python deps.
#
# Same flow as requirements-amd.txt (vLLM wheel bundles PyTorch — do not layer a
# separate PyTorch from PyPI on this path). Upstream:
#   https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
#
# Usage (from repo root, or any cwd — paths are resolved from this script):
#   bash scripts/install_axon_amd.sh
#   bash scripts/install_axon_amd.sh --nightly    # ROCm 7.2.1 / rocm721 nightlies
#
# Environment:
#   VENV_DIR            Virtualenv path (default: <repo>/.venv)
#   UV                  uv binary (default: uv)
#   SKIP_VLLM           If set to 1, only install requirements-amd.txt (vLLM already present)
#   VLLM_ROCM_VARIANT   For --nightly: override e.g. rocm721 (default: curl from nightly index)
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REQ_AMD="$PROJECT_DIR/requirements-amd.txt"

UV="${UV:-uv}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
USE_NIGHTLY=0
SKIP_VLLM="${SKIP_VLLM:-0}"

usage() {
    sed -n '1,20p' "$0" | tail -n +2
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        --nightly) USE_NIGHTLY=1; shift ;;
        --venv)
            [[ -n "${2:-}" ]] || { echo "error: --venv needs a path" >&2; exit 1; }
            VENV_DIR="$2"
            shift 2
            ;;
        *)
            echo "error: unknown option: $1 (try --help)" >&2
            exit 1
            ;;
    esac
done

[[ "$(uname -s)" == "Linux" ]] || {
    echo "error: ROCm vLLM wheels are Linux-only." >&2
    exit 1
}

[[ -f "$REQ_AMD" ]] || {
    echo "error: missing $REQ_AMD" >&2
    exit 1
}

command -v "$UV" &>/dev/null || {
    echo "error: '$UV' not found. Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
}

PY="$VENV_DIR/bin/python"
[[ -x "$PY" ]] || {
    echo "[info] creating venv at $VENV_DIR (Python 3.12, managed-python if needed)..."
    "$UV" venv --python 3.12 --seed --managed-python "$VENV_DIR"
}

install_vllm_stable() {
    echo "[info] installing vLLM (stable ROCm index)..."
    "$UV" pip install --python "$PY" vllm \
        --extra-index-url 'https://wheels.vllm.ai/rocm/' \
        --upgrade
}

install_vllm_nightly() {
    local variant="${VLLM_ROCM_VARIANT:-}"
    if [[ -z "$variant" ]]; then
        echo "[info] detecting VLLM_ROCM_VARIANT from https://wheels.vllm.ai/rocm/nightly ..."
        variant=$(curl -fsSL 'https://wheels.vllm.ai/rocm/nightly' | grep -oP 'rocm\d+' | head -1 | sed 's/%2B/+/g' || true)
    fi
    [[ -n "$variant" ]] || {
        echo "error: could not determine ROCm variant (set VLLM_ROCM_VARIANT manually)" >&2
        exit 1
    }
    echo "[info] installing vLLM nightly (variant=$variant)..."
    "$UV" pip install --python "$PY" --pre vllm \
        --extra-index-url "https://wheels.vllm.ai/rocm/nightly/${variant}" \
        --index-strategy unsafe-best-match
}

if [[ "$SKIP_VLLM" != "1" ]]; then
    if [[ "$USE_NIGHTLY" -eq 1 ]]; then
        install_vllm_nightly
    else
        install_vllm_stable
    fi
else
    echo "[info] SKIP_VLLM=1 — skipping vLLM install"
fi

echo "[info] installing Axon AMD requirements (amdsmi + base)..."
"$UV" pip install --python "$PY" --no-cache-dir -r "$REQ_AMD"

echo
echo "Done. Activate:"
echo "  source ${VENV_DIR}/bin/activate"
