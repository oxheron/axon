#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Build vLLM from source on ROCm, following upstream’s ROCm-from-source flow.
#
# Default pins (TRITON / FA / AITER / MORI) match vLLM’s docker/Dockerfile.rocm_base
# on the main branch unless you override env vars. Re-check that file when upgrading:
#   https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base
#
# Intended environment:
#   - Ubuntu 22.04–style system with ROCm in /opt/rocm, or
#   - AMD ROCm PyTorch Docker image (e.g. rocm/pytorch:…), then run this inside it.
#
# Not aimed at arbitrary Arch hosts unless you already have a working ROCm stack.
#
# Usage (from repo root):
#   bash scripts/build_vllm_rocm_source.sh
#   If PYTORCH_ROCM_ARCH is unset, gfx target(s) are taken from rocminfo (unique, ';'-joined).
#   Override manually:  export PYTORCH_ROCM_ARCH="gfx942"
#
# Options:
#   --install-pytorch-nightly URL   pip install torch/vision from this index (optional)
#   --with-flash                    build ROCm flash-attention (FA_BRANCH)
#   --with-aiter                    build AITER (AITER_BRANCH)
#   --with-mori                     build MORI (MORI_BRANCH)
#   --work-dir PATH                 clone/build under PATH (default: ./build/vllm-rocm-src)
#   --vllm-ref REF                  vLLM git ref to build (default: main)
#   -h, --help
#
# After success, activate your venv and use the same Python; vLLM is installed editable
# from the clone (setup.py develop).
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- pins from vLLM docker/Dockerfile.rocm_base (update when bumping vLLM) ---
TRITON_REPO="${TRITON_REPO:-https://github.com/ROCm/triton.git}"
TRITON_BRANCH="${TRITON_BRANCH:-ba5c1517}"
# Upstream Dockerfile cherry-picks onto ROCm/triton (may fail if branch moves):
TRITON_CHERRY1="${TRITON_CHERRY1:-555d04f}"
TRITON_CHERRY2="${TRITON_CHERRY2:-dd998b6}"

FA_REPO="${FA_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
FA_BRANCH="${FA_BRANCH:-0e60e394}"

AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_BRANCH="${AITER_BRANCH:-v0.1.10.post3}"

MORI_REPO="${MORI_REPO:-https://github.com/ROCm/mori.git}"
MORI_BRANCH="${MORI_BRANCH:-2d02c6a9}"

VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-main}"

WORK_DIR="${WORK_DIR:-$PROJECT_DIR/build/vllm-rocm-src}"
INSTALL_PYTORCH_NIGHTLY=""
WITH_FLASH=0
WITH_AITER=0
WITH_MORI=0

usage() {
  cat <<'EOF'
Build vLLM from source on ROCm (see header in this script for defaults and Dockerfile pins).

  bash scripts/build_vllm_rocm_source.sh [options]
  # optional: export PYTORCH_ROCM_ARCH="gfx942"  if rocminfo detection is wrong

Options:
  --install-pytorch-nightly URL   pip install torch torchvision from this --index-url
  --with-flash                    build flash-attention (FA_BRANCH)
  --with-aiter                    build AITER (develop)
  --with-mori                     build MORI
  --work-dir PATH                 clones/builds here (default: ./build/vllm-rocm-src)
  --vllm-ref REF                  vLLM git ref (default: main)
  -h, --help

Env (optional): PYTORCH_ROCM_ARCH TRITON_BRANCH FA_BRANCH AITER_BRANCH MORI_BRANCH ROCM_PATH
  SKIP_TRITON_CHERRY_PICK=1      skip cherry-picks if re-running on an already-patched tree
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --install-pytorch-nightly)
      INSTALL_PYTORCH_NIGHTLY="${2:?URL required after --install-pytorch-nightly}"
      shift 2
      ;;
    --with-flash) WITH_FLASH=1; shift ;;
    --with-aiter) WITH_AITER=1; shift ;;
    --with-mori) WITH_MORI=1; shift ;;
    --work-dir)
      WORK_DIR="${2:?path required}"
      shift 2
      ;;
    --vllm-ref)
      VLLM_REF="${2:?ref required}"
      shift 2
      ;;
    *)
      echo "error: unknown option: $1 (try --help)" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ROCM_PATH:-}" && -d /opt/rocm ]]; then
  export ROCM_PATH=/opt/rocm
fi

rocminfo_bin() {
  if command -v rocminfo >/dev/null 2>&1; then
    echo "rocminfo"
  elif [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/bin/rocminfo" ]]; then
    echo "${ROCM_PATH}/bin/rocminfo"
  elif [[ -x /opt/rocm/bin/rocminfo ]]; then
    echo /opt/rocm/bin/rocminfo
  else
    return 1
  fi
}

detect_pytorch_rocm_arch() {
  local bin out
  bin="$(rocminfo_bin)" || return 1
  out="$("$bin" 2>/dev/null | grep -oE 'gfx[0-9][0-9a-z]*' | sort -u | paste -sd';' -)"
  [[ -n "$out" ]] || return 1
  printf '%s' "$out"
}

if [[ -z "${PYTORCH_ROCM_ARCH:-}" ]]; then
  if detected="$(detect_pytorch_rocm_arch)"; then
    PYTORCH_ROCM_ARCH="$detected"
    echo "==> PYTORCH_ROCM_ARCH from rocminfo: $PYTORCH_ROCM_ARCH"
  else
    echo "error: PYTORCH_ROCM_ARCH unset and rocminfo did not yield gfx targets." >&2
    echo "  Install ROCm / rocminfo, or set e.g.  export PYTORCH_ROCM_ARCH=\"gfx942\"" >&2
    exit 1
  fi
else
  echo "==> PYTORCH_ROCM_ARCH from environment: $PYTORCH_ROCM_ARCH"
fi

if [[ ! -d "$WORK_DIR" ]]; then
  mkdir -p "$WORK_DIR"
fi
WORK_DIR="$(cd "$WORK_DIR" && pwd)"

echo "==> work dir: $WORK_DIR"

if ! command -v python3 >/dev/null; then
  echo "error: python3 not found" >&2
  exit 1
fi
if ! command -v git >/dev/null; then
  echo "error: git not found" >&2
  exit 1
fi

if [[ -n "${ROCM_PATH:-}" ]]; then
  echo "==> ROCM_PATH=$ROCM_PATH"
else
  echo "warning: ROCM_PATH unset and /opt/rocm not used — HIP builds may fail" >&2
fi

python3 -m pip install -q --upgrade pip
python3 -m pip install -q ninja "cmake<4" wheel pybind11 packaging

if [[ -n "$INSTALL_PYTORCH_NIGHTLY" ]]; then
  echo "==> installing PyTorch from index: $INSTALL_PYTORCH_NIGHTLY"
  python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  python3 -m pip install --no-cache-dir torch torchvision --index-url "$INSTALL_PYTORCH_NIGHTLY"
fi

echo "==> Triton (ROCm): clone $TRITON_REPO @ $TRITON_BRANCH"
cd "$WORK_DIR"
if [[ -d triton ]]; then
  echo "    (reusing existing triton dir)"
else
  git clone "$TRITON_REPO" triton
fi
cd triton
git fetch --all --tags
git checkout "$TRITON_BRANCH"

if [[ "${SKIP_TRITON_CHERRY_PICK:-0}" != "1" ]]; then
  git config user.email "${GIT_USER_EMAIL:-build@vllm-rocm.local}" || true
  git config user.name "${GIT_USER_NAME:-vllm-rocm-build}" || true
  echo "==> Triton: cherry-pick $TRITON_CHERRY1 $TRITON_CHERRY2 (set SKIP_TRITON_CHERRY_PICK=1 to skip)"
  if git cherry-pick "$TRITON_CHERRY1" && git cherry-pick "$TRITON_CHERRY2"; then
    :
  else
    echo "error: cherry-pick failed — branch may have diverged; try SKIP_TRITON_CHERRY_PICK=1 or update TRITON_BRANCH in script" >&2
    exit 1
  fi
fi

python3 -m pip uninstall -y triton 2>/dev/null || true
if [[ -f setup.py ]]; then
  python3 setup.py install
else
  cd python
  python3 setup.py install
  cd ..
fi
cd "$WORK_DIR"

if [[ "$WITH_FLASH" -eq 1 ]]; then
  echo "==> flash-attention: $FA_REPO @ $FA_BRANCH (GPU_ARCHS=$PYTORCH_ROCM_ARCH)"
  if [[ -d flash-attention ]]; then
    echo "    (reusing existing flash-attention dir)"
  else
    git clone "$FA_REPO" flash-attention
  fi
  cd flash-attention
  git fetch --all --tags
  git checkout "$FA_BRANCH"
  git submodule update --init
  GPU_ARCHS="$PYTORCH_ROCM_ARCH" python3 setup.py install
  cd "$WORK_DIR"
fi

if [[ "$WITH_AITER" -eq 1 ]]; then
  echo "==> AITER: $AITER_REPO @ $AITER_BRANCH"
  python3 -m pip uninstall -y aiter 2>/dev/null || true
  if [[ -d aiter ]]; then
    echo "    (reusing existing aiter dir)"
  else
    git clone --recursive "$AITER_REPO" aiter
  fi
  cd aiter
  git fetch --all --tags
  git checkout "$AITER_BRANCH"
  git submodule sync
  git submodule update --init --recursive
  AITER_ROCM_ARCH="${AITER_ROCM_ARCH:-$PYTORCH_ROCM_ARCH}"
  export AITER_ROCM_ARCH
  python3 setup.py develop
  cd "$WORK_DIR"
fi

if [[ "$WITH_MORI" -eq 1 ]]; then
  echo "==> MORI: $MORI_REPO @ $MORI_BRANCH"
  if [[ -d mori ]]; then
    echo "    (reusing existing mori dir)"
  else
    git clone "$MORI_REPO" mori
  fi
  cd mori
  git fetch --all --tags
  git checkout "$MORI_BRANCH"
  git submodule sync
  git submodule update --init --recursive
  MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx942;gfx950}"
  export MORI_GPU_ARCHS
  MORI_GPU_ARCHS="$MORI_GPU_ARCHS" python3 setup.py install
  cd "$WORK_DIR"
fi

echo "==> vLLM: clone $VLLM_REPO @ $VLLM_REF"
if [[ -d vllm ]]; then
  cd vllm
  git fetch --all --tags
  git checkout "$VLLM_REF"
  git pull --ff-only || true
else
  git clone "$VLLM_REPO" vllm
  cd vllm
  git checkout "$VLLM_REF"
fi

echo "==> pip: AMD SMI (wheel from ROCm if present)"
if [[ -d /opt/rocm/share/amd_smi ]]; then
  python3 -m pip install /opt/rocm/share/amd_smi
else
  echo "    (no /opt/rocm/share/amd_smi — installing PyPI amdsmi if needed)"
  python3 -m pip install -q amdsmi || true
fi

echo "==> pip: vLLM ROCm requirements + helpers"
python3 -m pip install -q --upgrade \
  numba scipy "huggingface-hub[cli,hf_transfer]" setuptools_scm
python3 -m pip install -q -r requirements/rocm.txt

export PYTORCH_ROCM_ARCH
echo "==> vLLM: python setup.py develop (ROCm arch=$PYTORCH_ROCM_ARCH)"
echo "    note: upstream says 'pip install .' may not work for ROCm; using setup.py develop"
python3 setup.py develop

echo ""
echo "Done. vLLM editable install is from: $WORK_DIR/vllm"
echo "Verify:  python3 -c 'import vllm; import torch; print(vllm.__version__, torch.__version__, getattr(torch.version,\"hip\",None))'"
