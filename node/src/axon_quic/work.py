"""AxonWork — a torch.distributed Work handle that wraps an already-completed op.

B-3 sends/receives are synchronous (the UDS call blocks until the sidecar ACKs
or returns data). The Work object returned from isend/irecv therefore reports
is_completed()=True immediately and wait() is a no-op.

If the operation raised an exception (timeout, lost connection), it is stored
on the Work and re-raised by wait() so vLLM sees a ConnectionError rather than
a deadlock.
"""
from __future__ import annotations

from typing import Optional


class AxonWork:
    """Completed work handle returned from AxonQuicProcessGroup isend/irecv."""

    def __init__(self, exc: Optional[BaseException] = None) -> None:
        self._exc = exc

    # ── torch.distributed Work protocol ─────────────────────────────────

    def is_completed(self) -> bool:
        return True

    def is_success(self) -> bool:
        return self._exc is None

    def exception(self) -> Optional[BaseException]:
        return self._exc

    def wait(self, timeout: int = 0) -> bool:
        if self._exc is not None:
            raise self._exc
        return True

    def get_future(self):  # type: ignore[override]
        """Return a resolved concurrent.futures.Future for callers that need one."""
        import concurrent.futures

        fut: concurrent.futures.Future = concurrent.futures.Future()
        if self._exc is not None:
            fut.set_exception(self._exc)
        else:
            fut.set_result(None)
        return fut

    def synchronize(self) -> None:
        self.wait()
