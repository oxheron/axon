from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from api.app import create_app
from runtime.state import UserServiceState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [user] %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Axon user-facing API service")
    parser.add_argument(
        "--coordinator-url",
        default=os.environ.get("COORDINATOR_URL"),
        required=os.environ.get("COORDINATOR_URL") is None,
        help="Coordinator base URL, e.g. http://10.0.0.10:8000",
    )
    parser.add_argument("--host", default="0.0.0.0", help="User service bind host")
    parser.add_argument("--port", type=int, default=8080, help="User service bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = UserServiceState(coordinator_url=args.coordinator_url.rstrip("/"))
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
