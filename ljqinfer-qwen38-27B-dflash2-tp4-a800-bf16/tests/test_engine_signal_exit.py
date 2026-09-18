"""Exercise production main with real Uvicorn and signals, no model/GPU load."""
import os
from pathlib import Path
import subprocess
import sys
import pytest

@pytest.mark.parametrize('failure,expected', [('none', 0), ('startup', 1), ('shutdown', 1)])
def test_engine_signal_exit_status(failure, expected):
    source_path = Path(__file__).resolve().parents[1] / 'server' / 'engine_server.py'
    script = r'''
import ast, asyncio, os, signal, sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
failure = sys.argv[2]
@asynccontextmanager
async def lifespan(app):
    if failure == 'startup':
        raise RuntimeError('injected startup failure')
    asyncio.get_running_loop().call_later(.3, os.kill, os.getpid(), signal.SIGTERM)
    yield
    if failure == 'shutdown':
        raise RuntimeError('injected shutdown failure')
    print('TEST_RELEASE_COMPLETE', flush=True)
app = FastAPI(lifespan=lifespan)
# Test-only ephemeral port; production main and signal handling are unchanged.
original_config = uvicorn.Config
def ephemeral_config(*args, **kwargs):
    kwargs['port'] = 0
    return original_config(*args, **kwargs)
uvicorn.Config = ephemeral_config
module = ast.parse(Path(sys.argv[1]).read_text())
main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
exec(compile(ast.Module(body=[main], type_ignores=[]), sys.argv[1], 'exec'), globals())
main()
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, '-c', script, str(source_path), failure],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == expected, result.stdout + result.stderr
    if failure == 'none':
        assert 'TEST_RELEASE_COMPLETE' in result.stdout
        assert 'Application shutdown complete.' in result.stderr
    else:
        assert 'injected ' + failure + ' failure' in result.stderr
