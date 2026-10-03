"""CPU/ASGI tests: cancellation delivery, late binding and API wiring."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from server.cancellation import CancelRelay, generate_until_disconnect


class Handle:
    state = 'running'

    def __init__(self):
        self.calls = 0
        self.stopped = threading.Event()

    def cancel(self):
        self.calls += 1
        self.state = 'cancelled'
        self.stopped.set()


@pytest.mark.parametrize('late', [False, True])
def test_relay_idempotent(late):
    relay, handle = CancelRelay(), Handle()
    if late:
        relay.cancel()
        relay.bind(handle)
    else:
        relay.bind(handle)
        relay.cancel()
    relay.cancel()
    assert handle.calls == 1


def test_relay_finished():
    relay, handle = CancelRelay(), Handle()
    relay.bind(handle)
    relay.finish()
    relay.cancel()
    assert handle.calls == 0


def test_relay_race():
    for _ in range(100):
        relay, handle = CancelRelay(), Handle()
        barrier = threading.Barrier(3)
        def bind():
            barrier.wait()
            relay.bind(handle)
        def cancel():
            barrier.wait()
            relay.cancel()
        a, b = threading.Thread(target=bind), threading.Thread(target=cancel)
        a.start(); b.start(); barrier.wait(); a.join(); b.join()
        assert handle.calls == 1


def test_cancel_failure_does_not_block_event_loop(caplog):
    class FailingHandle(Handle):
        def cancel(self):
            time.sleep(0.1)
            raise RuntimeError('injected RPC failure')
    async def scenario():
        relay = CancelRelay()
        relay.bind(FailingHandle())
        task = asyncio.create_task(relay.acancel())
        ticks = 0
        while not task.done():
            await asyncio.sleep(0.005)
            ticks += 1
        await task
        assert ticks >= 5
    asyncio.run(scenario())
    assert 'Backend cancellation failed' in caplog.text


@pytest.mark.parametrize('mode', ['normal', 'disconnect', 'late', 'error', 'task_cancel'])
def test_generate_lifecycle(mode):
    handle = Handle()
    worker_done = threading.Event()
    bound = threading.Event()
    class Layer:
        def generate(self, body, *, on_submit):
            try:
                if mode == 'late':
                    time.sleep(0.15)
                on_submit(handle)
                bound.set()
                if mode == 'normal':
                    return 'result'
                if mode == 'error':
                    raise ValueError('original error')
                assert handle.stopped.wait(2), 'worker not cancelled'
                return 'cancelled result'
            finally:
                worker_done.set()
    class Request:
        async def is_disconnected(self):
            return mode == 'late' or (mode == 'disconnect' and bound.is_set())
    async def scenario():
        task = asyncio.create_task(generate_until_disconnect(Request(), Layer(), {}))
        if mode == 'task_cancel':
            while not bound.is_set():
                await asyncio.sleep(0.005)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == 'error':
            with pytest.raises(ValueError, match='original error'):
                await task
        else:
            result = await task
            assert result == ('result' if mode == 'normal' else None)
        # Do not block the event loop while the abandoned worker binds late.
        for _ in range(400):
            if worker_done.is_set():
                break
            await asyncio.sleep(0.005)
        assert worker_done.is_set()
    asyncio.run(scenario())
    assert handle.calls == (0 if mode == 'normal' else 1)


@pytest.mark.parametrize('endpoint', ['/v1/messages', '/v1/chat/completions'])
@pytest.mark.parametrize('disconnect', [False, True])
def test_nonstream_asgi(endpoint, disconnect, monkeypatch):
    from server import server as http
    from server.service import GenerationResult
    handle = Handle()
    bound = threading.Event()
    class Layer:
        _strategy = SimpleNamespace(check_ready=lambda: None)
        def build(self, body):
            return None
        def generate(self, body, *, on_submit):
            on_submit(handle)
            bound.set()
            if disconnect:
                assert handle.stopped.wait(2)
            return GenerationResult(message_id='msg_test', model='glm-5.3',
                content=[{'type': 'text', 'text': '17'}], stop_reason='end_turn',
                input_tokens=8, output_tokens=1, stats={})
    monkeypatch.setitem(http._state, 'service', Layer())
    async def scenario():
        payload = {'model':'glm-5.3', 'messages':[{'role':'user','content':'9+8'}],
                   'max_tokens':32, 'stream':False}
        incoming = asyncio.Queue()
        incoming.put_nowait({'type':'http.request', 'body':json.dumps(payload).encode(), 'more_body':False})
        output = []
        async def send(event):
            output.append(event)
        async def drop():
            while not bound.is_set():
                await asyncio.sleep(0.005)
            await incoming.put({'type':'http.disconnect'})
        scope = {'type':'http','asgi':{'version':'3.0'},'http_version':'1.1',
                 'method':'POST','scheme':'http','path':endpoint,'raw_path':endpoint.encode(),
                 'query_string':b'', 'headers':[(b'authorization',b'Bearer devkey'),
                 (b'content-type',b'application/json')], 'client':('127.0.0.1',1),
                 'server':('127.0.0.1',8000)}
        dropper = asyncio.create_task(drop()) if disconnect else None
        await asyncio.wait_for(http.app(scope, incoming.get, send), 3)
        if dropper:
            await dropper
        start = next(x for x in output if x['type']=='http.response.start')
        assert start['status'] == (499 if disconnect else 200)
    asyncio.run(scenario())
    assert handle.calls == int(disconnect)


def test_cancel_with_generation_limiter_exhausted():
    import anyio
    async def scenario():
        limiter = anyio.to_thread.current_default_thread_limiter()
        old = limiter.total_tokens
        limiter.total_tokens = 1
        handle = Handle()
        relay = CancelRelay()
        relay.bind(handle)
        async def occupy():
            await anyio.to_thread.run_sync(lambda: handle.stopped.wait(2))
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(occupy)
                while limiter.borrowed_tokens != 1:
                    await anyio.sleep(0.005)
                with anyio.fail_after(0.5):
                    await relay.acancel()
                assert handle.calls == 1
        finally:
            limiter.total_tokens = old
    asyncio.run(scenario())
