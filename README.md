# Axon: Barebones Distributed vLLM Inference

This repo provides a minimal three-service setup for static, pipeline-parallel inference across multiple machines:

- `src/axon/coordinator.py`: node registration + startup broadcast
- `src/axon/node_agent.py`: node VRAM detection + Ray join + optional local vLLM worker launch
- `src/axon/server.py`: OpenAI-compatible `/v1/chat/completions` API on coordinator

Docker assets:

- `docker/coordinator.Dockerfile`
- `docker/node-agent.Dockerfile`
- `docker/server.Dockerfile`
- `scripts/test_12gb_gpu.sh` (single-GPU smoke test)

## Assumptions

- One GPU per machine
- All machines have the same model available locally
- Static cluster membership (no dynamic resize)
- Ray + vLLM installed on each machine (on AMD, vLLM must be a **ROCm** build, not the default PyPI CUDA wheel)

### NVIDIA vs AMD (ROCm)

- **NVIDIA:** install the CUDA build of PyTorch and vLLM as usual (`requirements.txt` pulls `vllm` from PyPI). VRAM reporting uses NVML (`pynvml`) or PyTorch.
- **AMD:** do **not** rely on `pip install vllm` from PyPI for inference — that wheel is typically **CUDA-linked** and will fail on a ROCm-only machine (for example missing `libcudart.so.*`, `UnspecifiedPlatform`, or “failed to infer device type”). Install **vLLM built for ROCm** using the upstream process below, then install Axon’s Python dependencies with `requirements-base.txt` or `requirements-amd.txt` as documented. The node agent also reads VRAM via `rocm-smi` when NVML is unavailable.

Docker images in this repo default to `vllm/vllm-openai:latest` (CUDA). For AMD hosts you can either use a published ROCm image as a base (for example `vllm/vllm-openai-rocm:latest`) or, for parity with vLLM’s documented ROCm path, **build vLLM from `Dockerfile.rocm`** and run Axon inside that environment (see [vLLM Docker deployment](https://docs.vllm.ai/en/stable/deployment/docker.html)).

## Install

**NVIDIA (or default CUDA-capable PyTorch on PyPI):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If pip fails with `Memoryview is too large` while downloading PyTorch or vLLM, reinstall with `pip install --no-cache-dir -r requirements.txt`.

### AMD: vLLM on ROCm (install vLLM first)

Upstream vLLM documents ROCm support (hardware, ROCm version, and build steps) here: [vLLM installation — GPU / ROCm](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html). The summary below matches the **recommended** upstream layout; **exact package pins evolve**, so if anything disagrees with the link, follow the documentation.

**Requirements (typical upstream ROCm 6.2 flow):**

- OS: Linux  
- Python: 3.9–3.12  
- GPUs such as MI200 (gfx90a), MI300 (gfx942), Radeon RX 7900 (gfx1100), with a **ROCm** stack that matches what you build against (for example ROCm **6.2** in the upstream guide)

**Option 1 — Build from source with Docker (recommended upstream path)**

Use [Docker BuildKit](https://docs.docker.com/build/buildkit/). Either set `DOCKER_BUILDKIT=1` for `docker build`, or enable BuildKit in the Docker daemon configuration.

From a checkout of the [vLLM repository](https://github.com/vllm-project/vllm), build the ROCm image (defaults are tuned for MI200/MI300-class GPUs):

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
```

For **Radeon RX 7900 (gfx1100)**, upstream recommends disabling CK flash-attention during the image build:

```bash
DOCKER_BUILDKIT=1 docker build --build-arg BUILD_FA=0 -f Dockerfile.rocm -t vllm-rocm .
```

`Dockerfile.rocm` supports other `BASE_IMAGE`, `FX_GFX_ARCHS`, `BUILD_TRITON`, `FA_BRANCH`, and related `--build-arg` values; see the vLLM doc above.

Run the image with GPU access (example from upstream):

```bash
docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v <path/to/model>:/app/model \
  vllm-rocm \
  bash
```

**Option 2 — Host build script (PyTorch ROCm wheels + vLLM `setup.py develop`)**

From the Axon repo root, with your virtualenv activated and ROCm drivers installed:

```bash
pip install --no-cache-dir -U pip
bash scripts/build_pytorch_vllm_rocm.sh
```

The script installs **ROCm PyTorch** from a configurable index (default: PyTorch **nightly** `rocm6.2`), clones vLLM, installs vLLM’s ROCm requirements file (`requirements/rocm.txt` on current main, or legacy `requirements-rocm.txt` on older tags), sets **`PYTORCH_ROCM_ARCH`** from `rocminfo` when possible (uses **`/opt/rocm/bin/rocminfo`** if `rocminfo` is not on `PATH`), then runs **`python setup.py develop`** in the vLLM tree. See the header in `scripts/build_pytorch_vllm_rocm.sh` for environment variables (`PYTORCH_INDEX_URL`, `VLLM_GIT_REF`, `BUILD_TRITON_FROM_SOURCE`, `VLLM_ROCM_REQUIREMENTS`, `ROCMINFO`, etc.).

For **Triton from source** or **CK flash-attention** (for example RX 7900 / gfx1100 nuances), follow the [vLLM ROCm installation page](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html); you can enable the optional Triton build with `BUILD_TRITON_FROM_SOURCE=1 bash scripts/build_pytorch_vllm_rocm.sh` (slow).

**Installing Axon after vLLM + PyTorch exist**

- **If your environment already has ROCm PyTorch and your ROCm vLLM build** (for example inside the `vllm-rocm` image): install only Axon’s shared libraries so pip does not pull the CUDA vLLM wheel:

  ```bash
  pip install --no-cache-dir -U pip
  pip install --no-cache-dir -r requirements-base.txt
  ```

- **Prebuilt ROCm vLLM (upstream wheel bundles PyTorch):** follow the `uv` flow in `requirements-amd.txt`, then install Axon:

  ```bash
  uv venv --python 3.12 --seed --managed-python
  source .venv/bin/activate
  uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade
  uv pip install --no-cache-dir -r requirements-amd.txt
  ```

- **vLLM from source with your own ROCm PyTorch** (for example **Option 2** below): after the vLLM install, use `pip install --no-cache-dir -r requirements-base.txt` only so pip does not pull the CUDA vLLM wheel.

Use `--no-cache-dir` for large wheels so pip does not try to cache multi‑gigabyte downloads (older pip can raise `ValueError: Memoryview is too large`).

## 1) Start Coordinator (on head node)

```bash
PYTHONPATH=src python -m axon.coordinator \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --min-nodes 2 \
  --autostart-ray-head
```

Useful endpoints:

- `GET /status`
- `GET /config` (available after startup)
- `POST /register` (used by node agents)

## 2) Start Node Agent (on each worker node)

```bash
PYTHONPATH=src python -m axon.node_agent \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --host 0.0.0.0 \
  --port 9000 \
  --advertise-host <THIS_NODE_REACHABLE_IP> \
  --vllm-gpu-memory-utilization 0.72 \
  --vllm-max-model-len 1024 \
  --vllm-dtype float16
```

When enough nodes register, the coordinator broadcasts `/startup`. The node then:

1. Joins Ray cluster at coordinator-provided head address
2. Starts a local vLLM process using the coordinator's `pipeline_parallel_size`

If you only want Ray join (and no local vLLM API on workers), pass `--no-vllm-worker`.

## 3) Start Inference Server (on coordinator node)

```bash
PYTHONPATH=src python -m axon.server \
  --coordinator-url http://127.0.0.1:8000 \
  --vllm-worker-url http://127.0.0.1:8100 \
  --host 0.0.0.0 \
  --port 8080
```

The server waits until the coordinator reports the pipeline is ready, then proxies OpenAI-compatible traffic to the vLLM worker URL above.

## Test Inference

```bash
curl http://<COORDINATOR_IP>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 64
  }'
```

Streaming mode:

```bash
curl http://<COORDINATOR_IP>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Count to five."}],
    "max_tokens": 64,
    "stream": true
  }'
```

## Docker (per-service images)

Build images:

```bash
docker build -f docker/coordinator.Dockerfile -t axon-coordinator:test .
docker build -f docker/node-agent.Dockerfile -t axon-node-agent:test .
docker build -f docker/server.Dockerfile -t axon-server:test .
```

AMD (ROCm) example — same Axon Dockerfiles, **prebuilt** ROCm vLLM base image (quick path; for the upstream-recommended ROCm build, use **`Dockerfile.rocm`** in the vLLM repo and install Axon with `requirements-base.txt` inside that image):

```bash
export ROCM_BASE=vllm/vllm-openai-rocm:latest
docker build -f docker/coordinator.Dockerfile --build-arg "VLLM_BASE_IMAGE=${ROCM_BASE}" -t axon-coordinator:test .
docker build -f docker/node-agent.Dockerfile --build-arg "VLLM_BASE_IMAGE=${ROCM_BASE}" -t axon-node-agent:test .
docker build -f docker/server.Dockerfile --build-arg "VLLM_BASE_IMAGE=${ROCM_BASE}" -t axon-server:test .
```

Run containers on AMD with GPU access per [vLLM ROCm instructions](https://docs.vllm.ai/en/stable/deployment/docker.html) (for example `--device /dev/kfd`, `--device /dev/dri`, `--group-add video`, `--ipc host`).

## 10-12GB GPU smoke test script

This script is designed for a single machine with a max ~16GB GPU and runs a 10-12GB friendly test profile by default:

```bash
./scripts/test_12gb_gpu.sh
```

Optional environment variables:

- `MODEL` (default: `Qwen/Qwen2.5-3B-Instruct`)
- `COORD_PORT` (default: `8000`)
- `SERVER_PORT` (default: `8080`)
- `TIMEOUT_SEC` (default: `420`)
- `GPU_VENDOR` (default: `auto`) — `nvidia`, `amd`, or `auto` (detect via `nvidia-smi` / `rocm-smi`)
- `VLLM_BASE_IMAGE` — Docker `FROM` for all three images (defaults by vendor in the script: CUDA vs `vllm-openai-rocm`)
- `AXON_DOCKER_SERVER_GPU_FLAGS` — override the `docker run` GPU-related flags for the inference container (space-separated)
- `KEEP_CONTAINERS=1` to keep stack running after the test
