from __future__ import annotations

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.import_utils import resolve_obj_by_qualname

_KNOWN_BACKENDS = frozenset({"ray", "mp", "uni", "external_launcher"})
_orig_check = EngineArgs._check_feature_supported


def _patched_check(self: EngineArgs) -> None:
    backend = self.distributed_executor_backend
    if isinstance(backend, str) and backend not in _KNOWN_BACKENDS:
        try:
            self.distributed_executor_backend = resolve_obj_by_qualname(backend)
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
