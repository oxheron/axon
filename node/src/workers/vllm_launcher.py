from __future__ import annotations

import importlib
from typing import Any

from vllm.engine.arg_utils import EngineArgs

_KNOWN_BACKENDS = frozenset({"ray", "mp", "uni", "external_launcher"})
_orig_check = EngineArgs._check_feature_supported


def _resolve_class_by_qualname(qualname: str) -> Any:
    """Resolve ``package.module.Class`` for custom executor backends (stdlib only)."""
    parts = qualname.split(".")
    if len(parts) < 2:
        raise ValueError(f"invalid executor qualname: {qualname!r}")
    for i in range(len(parts) - 1, 0, -1):
        modname = ".".join(parts[:i])
        attr_path = parts[i:]
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        obj: Any = mod
        try:
            for attr in attr_path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise ImportError(f"cannot resolve executor class: {qualname!r}")


def _patched_check(self: EngineArgs) -> None:
    backend = self.distributed_executor_backend
    if isinstance(backend, str) and backend not in _KNOWN_BACKENDS:
        try:
            self.distributed_executor_backend = _resolve_class_by_qualname(backend)
        except Exception:
            pass
    _orig_check(self)


EngineArgs._check_feature_supported = _patched_check  # type: ignore[method-assign]

if __name__ == "__main__":
    import uvloop
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
