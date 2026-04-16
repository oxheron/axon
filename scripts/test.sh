#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Axon test script — sends a chat completion request and prints the response.
#
# Usage:
#   bash scripts/test.sh
#
# Environment variables:
#   SERVER_URL   Inference server URL (default: http://localhost:8080)
#   MODEL        Model name           (default: Qwen/Qwen2.5-3B-Instruct)
#   PROMPT       User message         (default: "Say hello in one sentence.")
#   STREAM       Set to 1 for streaming output (default: 0)
# ---------------------------------------------------------------------------

SERVER_URL="${SERVER_URL:-http://localhost:8080}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PROMPT="${PROMPT:-Say hello in one sentence.}"
STREAM="${STREAM:-0}"

python3 - <<PYEOF
import json
import os
import sys
import urllib.error
import urllib.request

server = os.environ.get("SERVER_URL", "http://localhost:8080")
model  = os.environ.get("MODEL",      "Qwen/Qwen2.5-3B-Instruct")
prompt = os.environ.get("PROMPT",     "Say hello in one sentence.")
stream = os.environ.get("STREAM",     "0") == "1"

payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 128,
    "stream": stream,
}).encode()

req = urllib.request.Request(
    f"{server}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

print(f"Sending request to {server}")
print(f"  model:  {model}")
print(f"  prompt: {prompt}")
print(f"  stream: {stream}")
print()

try:
    with urllib.request.urlopen(req) as resp:
        if stream:
            print("=== Streaming response ===")
            for raw_line in resp:
                line = raw_line.decode().rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    print()
                    print("[stream complete]")
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    print(delta, end="", flush=True)
                except Exception:
                    pass
        else:
            body = json.loads(resp.read())
            print("=== Full response ===")
            print(json.dumps(body, indent=2))
            print()
            print("=== Assistant message ===")
            print(body["choices"][0]["message"]["content"])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.reason}", file=sys.stderr)
    print(e.read().decode(), file=sys.stderr)
    sys.exit(1)
except urllib.error.URLError as e:
    print(f"Connection error: {e.reason}", file=sys.stderr)
    print("Is the server running? Try: bash scripts/start.sh", file=sys.stderr)
    sys.exit(1)
PYEOF
