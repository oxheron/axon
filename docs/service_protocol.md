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
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "pipeline_parallel_size": 1,
  "ray_head_address": "127.0.0.1:6379"
}
```

- Response body:

```json
{
  "accepted": true,
  "duplicate": false
}
```

## Coordinator Status

`user` polls `coordinator` for control-plane readiness.

- Method: `GET /status`
- Response body:

```json
{
  "min_nodes": 1,
  "registered_nodes": 1,
  "pipeline_ready": true,
  "inference_ready": true,
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "ray_head_address": "127.0.0.1:6379",
  "selected_node_id": "node-a",
  "nodes": [
    {
      "node_id": "node-a",
      "host": "127.0.0.1",
      "port": 9000,
      "vram_gb": 24.0,
      "callback_url": "http://127.0.0.1:9000",
      "worker_url": "http://127.0.0.1:8100"
    }
  ]
}
```

For the first pass, `selected_node_id` is the first registered node.

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
