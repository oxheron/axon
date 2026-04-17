# Axon Service Protocol

This document defines the language-neutral HTTP/JSON boundary between the
three runtime services:

- `user`: OpenAI-compatible API surface
- `coordinator`: control plane and routing service
- `node`: model-worker controller

## Node Registration

`node` registers itself with `coordinator`.

- Method: `POST /register`
- Request body:

```json
{
  "node_id": "node-a",
  "host": "127.0.0.1",
  "port": 9000,
  "vram_gb": 24.0,
  "callback_url": "http://127.0.0.1:9000",
  "worker_url": "http://127.0.0.1:8100"
}
```

- Response body:

```json
{
  "accepted": true,
  "registered_nodes": 1,
  "startup_triggered": true
}
```

Coordinator returns `409` once startup has already been triggered.

## Cluster Startup

`coordinator` tells each registered `node` to start its local worker setup.

- Method: `POST {callback_url}/startup`
- Request body:

```json
{
  "cluster_id": "cluster-1744825012345678000",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "execution_mode": "vllm_ray_pipeline",
  "pipeline_parallel_size": 2,
  "stage_count": 2,
  "entry_node_id": "node-a",
  "ray_head_address": "127.0.0.1:6379",
  "backend_config": {
    "ray_head_address": "127.0.0.1:6379",
    "env_overrides": {},
    "launch_args": []
  },
  "nodes": [
    {
      "node_id": "node-a",
      "host": "127.0.0.1",
      "port": 9000,
      "vram_gb": 24.0,
      "callback_url": "http://127.0.0.1:9000",
      "worker_url": "http://127.0.0.1:8100",
      "stage_index": 0,
      "stage_role": "entry"
    },
    {
      "node_id": "node-b",
      "host": "127.0.0.1",
      "port": 9001,
      "vram_gb": 24.0,
      "callback_url": "http://127.0.0.1:9001",
      "worker_url": "http://127.0.0.1:8101",
      "stage_index": 1,
      "stage_role": "final"
    }
  ],
  "assignment": {
    "node_id": "node-a",
    "stage_index": 0,
    "stage_count": 2,
    "stage_role": "entry",
    "load_strategy": "vllm_ray_stage",
    "slice_spec": {
      "kind": "stage_index",
      "stage_index": 0,
      "stage_count": 2,
      "partition_label": "0/2",
      "executable": true
    },
    "peer_nodes": [
      {
        "node_id": "node-b",
        "stage_index": 1,
        "stage_role": "final",
        "worker_url": "http://127.0.0.1:8101"
      }
    ],
    "worker_endpoint": "http://127.0.0.1:8100"
  }
}
```

- Response body:

```json
{
  "accepted": true,
  "duplicate": false
}
```

The coordinator keeps `pipeline_parallel_size` and `ray_head_address` for
backward compatibility, but `cluster_id`, `execution_mode`, `nodes`, and
`assignment` are the authoritative slice-aware load-plan fields.

## Node Lifecycle Reporting

`node` reports coordinator-visible lifecycle progress after registration and
through startup.

- Method: `POST /node-status`
- Request body:

```json
{
  "node_id": "node-a",
  "cluster_id": "cluster-1744825012345678000",
  "lifecycle_state": "pipeline_ready",
  "lifecycle_detail": "Worker health endpoint is serving traffic.",
  "worker_url": "http://127.0.0.1:8100",
  "assignment": {
    "node_id": "node-a",
    "stage_index": 0,
    "stage_count": 2,
    "stage_role": "entry",
    "load_strategy": "vllm_ray_stage",
    "slice_spec": {
      "kind": "stage_index",
      "stage_index": 0,
      "stage_count": 2,
      "partition_label": "0/2",
      "executable": true
    },
    "peer_nodes": [
      {
        "node_id": "node-b",
        "stage_index": 1,
        "stage_role": "final",
        "worker_url": "http://127.0.0.1:8101"
      }
    ],
    "worker_endpoint": "http://127.0.0.1:8100"
  }
}
```

Expected lifecycle states include:

- `registered`
- `assigned`
- `load_started`
- `slice_loaded`
- `backend_joined`
- `pipeline_ready`
- `dry_run_ready`
- `failed`

## Coordinator Status

`user` polls `coordinator` for control-plane readiness.

- Method: `GET /status`
- Response body:

```json
{
  "min_nodes": 2,
  "registered_nodes": 2,
  "cluster_ready": true,
  "entry_node_ready": true,
  "all_nodes_ready": true,
  "backend_ready": true,
  "pipeline_ready": true,
  "inference_ready": true,
  "cluster_id": "cluster-1744825012345678000",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "execution_mode": "vllm_ray_pipeline",
  "stage_count": 2,
  "ray_head_address": "127.0.0.1:6379",
  "entry_node_id": "node-a",
  "selected_node_id": "node-a",
  "nodes": [
    {
      "node_id": "node-a",
      "host": "127.0.0.1",
      "port": 9000,
      "vram_gb": 24.0,
      "callback_url": "http://127.0.0.1:9000",
      "worker_url": "http://127.0.0.1:8100",
      "stage_index": 0,
      "stage_role": "entry",
      "cluster_id": "cluster-1744825012345678000",
      "lifecycle_state": "pipeline_ready",
      "lifecycle_detail": "Worker health endpoint is serving traffic.",
      "assignment": {
        "node_id": "node-a",
        "stage_index": 0,
        "stage_count": 2,
        "stage_role": "entry",
        "load_strategy": "vllm_ray_stage",
        "slice_spec": {
          "kind": "stage_index",
          "stage_index": 0,
          "stage_count": 2,
          "partition_label": "0/2",
          "executable": true
        },
        "peer_nodes": [
          {
            "node_id": "node-b",
            "stage_index": 1,
            "stage_role": "final",
            "worker_url": "http://127.0.0.1:8101"
          }
        ],
        "worker_endpoint": "http://127.0.0.1:8100"
      },
      "updated_at": "2026-04-16T12:00:00Z"
    },
    {
      "node_id": "node-b",
      "host": "127.0.0.1",
      "port": 9001,
      "vram_gb": 24.0,
      "callback_url": "http://127.0.0.1:9001",
      "worker_url": "http://127.0.0.1:8101",
      "stage_index": 1,
      "stage_role": "final",
      "cluster_id": "cluster-1744825012345678000",
      "lifecycle_state": "backend_joined",
      "lifecycle_detail": "Joined Ray cluster at 127.0.0.1:6379.",
      "assignment": {
        "node_id": "node-b",
        "stage_index": 1,
        "stage_count": 2,
        "stage_role": "final",
        "load_strategy": "vllm_ray_stage",
        "slice_spec": {
          "kind": "stage_index",
          "stage_index": 1,
          "stage_count": 2,
          "partition_label": "1/2",
          "executable": true
        },
        "peer_nodes": [
          {
            "node_id": "node-a",
            "stage_index": 0,
            "stage_role": "entry",
            "worker_url": "http://127.0.0.1:8100"
          }
        ],
        "worker_endpoint": "http://127.0.0.1:8101"
      },
      "updated_at": "2026-04-16T12:00:02Z"
    }
  ]
}
```

`selected_node_id` follows `entry_node_id` when the coordinator has an explicit
topology; otherwise it falls back to the first registered node.

Readiness fields are interpreted as:

- `cluster_ready`: enough nodes have registered and the coordinator has an active topology
- `entry_node_ready`: the designated entry node has reached its terminal ready state
- `all_nodes_ready`: every assigned node has reached the expected ready state for the active mode
- `backend_ready`: backend join/load prerequisites are satisfied for executable modes
- `pipeline_ready`: the control plane has a complete, ready topology
- `inference_ready`: the pipeline is executable and the entry worker is healthy enough to accept `/v1` traffic

## Inference Routing

The `user` service proxies OpenAI-style requests to `coordinator`.
The `coordinator` selects a node and proxies the same request to the node's
vLLM worker.

- `user -> coordinator`: `/{v1 path}`
- `coordinator -> node worker`: `/{v1 path}`

The body, headers, query parameters, status code, and streaming response
semantics are preserved end-to-end.

## Health Endpoints

- `GET /healthz` exists on all three services.
- `user /healthz` reports control-plane readiness from `coordinator`.
- `coordinator /healthz` reports its own liveness.
- `node /healthz` reports registration/startup/worker state.
