"""HTTP cancellation bridge; model execution owns the safe stop boundary."""
import logging
import threading

import anyio

log = logging.getLogger(__name__)
# Generation workers can occupy the default limiter while waiting on the engine.
# Cancellation must not queue behind the very requests it needs to stop.
_cancel_limiter = anyio.CapacityLimiter(8)


class CancelRelay:
    """One handle, possibly bound after disconnect; never hold a lock over RPC."""

    def __init__(self):
        self._lock = threading.Lock()
        self._handle = None
        self._cancelled = False
        self._finished = False

    @staticmethod
    def _deliver(handle):
        if handle is None or getattr(handle, "state", None) in ("done", "cancelled"):
            return
        try:
            handle.cancel()
        except Exception:
            # Transport cleanup must still finish; do not claim backend stopped.
            log.exception("Backend cancellation failed; request may still be running")

    def bind(self, handle):
        """Called by the generation worker, not the ASGI event loop."""
        with self._lock:
            if self._finished:
                return
            cancelled = self._cancelled
            if not cancelled:
                self._handle = handle
        if cancelled:
            self._deliver(handle)

    def finish(self):
        with self._lock:
            self._finished = True
            self._handle = None

    def cancel(self):
        """Synchronous worker entry; idempotent including failed delivery."""
        with self._lock:
            if self._cancelled or self._finished:
                return
            self._cancelled = True
            handle, self._handle = self._handle, None
        self._deliver(handle)

    async def acancel(self):
        # Starlette may already have cancelled the response task on disconnect.
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(self.cancel, limiter=_cancel_limiter)


async def generate_until_disconnect(request, layer, body):
    """Return the result, or None after disconnect; abandon only the local waiter.

    A late query submission still binds to the cancelled relay and sends cancel.
    Exceptions are re-raised outside the task group to preserve API error mapping.
    """
    relay = CancelRelay()
    done = anyio.Event()
    result = error = None

    def generate():
        try:
            value = layer.generate(body, on_submit=relay.bind)
        except BaseException:
            relay.cancel()
            raise
        else:
            relay.finish()
            return value

    async def run():
        nonlocal result, error
        try:
            result = await anyio.to_thread.run_sync(generate, abandon_on_cancel=True)
        except Exception as exc:
            error = exc
        finally:
            done.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(run)
            while not done.is_set():
                if await request.is_disconnected():
                    await relay.acancel()
                    group.cancel_scope.cancel()
                    return None
                with anyio.move_on_after(0.05):
                    await done.wait()
        if error is not None:
            raise error
        return result
    finally:
        await relay.acancel()
