# Axon: Barebones Distributed vLLM Inference

This repo provides a minimal three-service setup for static, pipeline-parallel inference across multiple machines:

- `coordinator/src/`: Go `main` entrypoint for the coordinator binary
- `coordinator/internal/server/`: coordinator HTTP API and orchestration logic
- `coordinator/test/`: Go unit tests (`go test ./...` from `coordinator/`)
- `node/src/`: Python node service for VRAM detection, Ray join, and local vLLM worker launch
- `user/src/`: Python OpenAI-compatible `/v1/chat/completions` API that talks to the coordinator

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
cd coordinator
go run ./src \
  --host 0.0.0.0 \
  --port 8000 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --min-nodes 2 \
  --autostart-ray-head
```

Optional Phase Two controls:

- `--execution-mode dry_run` validates the load plan and topology without
  launching executable backend workers
- `--execution-mode slice_loaded_pipeline` preserves coordinator-assigned slice
  semantics while passing backend launch details through the same startup
  contract
- Repeat `--backend-launch-arg ...` or `--backend-env KEY=VALUE` to inject
  backend-specific worker flags without changing the public protocol

Useful endpoints:

- `GET /status`
- `GET /config` (available after startup)
- `POST /register` (used by node agents)

## 2) Start Node Service (on each worker node)

```bash
python node/src/agent.py \
  --coordinator-url http://<COORDINATOR_IP>:8000 \
  --host 0.0.0.0 \
  --port 9000 \
  --advertise-host <THIS_NODE_REACHABLE_IP> \
  --vllm-gpu-memory-utilization 0.72 \
  --vllm-max-model-len 1024 \
  --vllm-dtype float16
```

When enough nodes register, the coordinator broadcasts a slice-aware `/startup`
load plan. The node then:

1. Receives `cluster_id`, `execution_mode`, the ordered topology, and a
   per-node `assignment`
2. Joins Ray cluster at the coordinator-provided head address when the plan
   requires a multi-node backend
3. Starts a local vLLM process using the planned stage count plus any backend
   env/CLI overrides
4. Reports lifecycle updates such as `assigned`, `load_started`,
   `backend_joined`, `slice_loaded`, and `pipeline_ready` back to the
   coordinator

If you only want Ray join (and no local vLLM API on workers), pass `--no-vllm-worker`.

## 3) Start User Service (on coordinator node)

```bash
python user/src/server.py \
  --coordinator-url http://127.0.0.1:8000 \
  --host 0.0.0.0 \
  --port 8080
```

The user service waits until the coordinator reports the pipeline is ready, then
proxies OpenAI-compatible traffic to the coordinator. The coordinator forwards
the same request to the configured entry node's vLLM worker instead of always
assuming "first registered node wins".

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

## Local Two-Node Harness

For a reproducible Phase Two control-plane run, use the compose-based two-node
harness:

```bash
bash scripts/two_node_compose.sh
```

You need either the **Docker Compose V2 plugin** (`docker compose version` works)
or standalone **`docker-compose`** on `PATH`. Plain `docker` without Compose
cannot run this stack.

The **Docker daemon** must be running and reachable (for example
`docker info` succeeds). Errors about `/var/run/docker.sock` mean the client
cannot talk to the daemon; start Docker (`systemctl start docker` on many
Linux setups) or fix `DOCKER_HOST` for rootless or remote engines.

Defaults:

- brings up `coordinator`, `node-a`, `node-b`, and `user`
- starts the coordinator with `--min-nodes 2`
- keeps node registration order deterministic for local entry-node selection
- defaults to `AXON_EXECUTION_MODE=dry_run` so the full two-node topology can be
  validated without requiring a working distributed backend on day one

Useful overrides:

- `AXON_EXECUTION_MODE=slice_loaded_pipeline` to exercise the real executable path
- `AXON_EXECUTION_MODE=vllm_ray_pipeline` to test the current Ray-backed path
- `KEEP_RUNNING=1` to leave the compose stack up for manual inspection
- `VLLM_BASE_IMAGE=...` to swap CUDA vs ROCm-compatible images

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

for rocm make sure to export HSA Override (dev note)