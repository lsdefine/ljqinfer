"""Save long generations outside the source tree; never log authentication headers."""
import json
import logging
import os
import time
import uuid
from pathlib import Path

DEFAULT_DIR = Path('/mnt/data/kw/glm53_long_generations')
THRESHOLD = 8192


class GenerationTrace:
    def __init__(self, input_ids, plan, request_id):
        self.input_ids = list(input_ids)
        self.plan = dict(plan)
        self.request_id = request_id
        self.output_ids = []
        self.terminal = None
        self.started = time.time()

    def event(self, event):
        kind = event.get('type')
        if kind == 'token':
            self.output_ids.extend(event.get('token_ids') or [])
        elif kind in ('end', 'error'):
            self.terminal = dict(event)

    def finish(self, tokenizer, queue):
        if len(self.output_ids) < THRESHOLD:
            return
        try:
            directory = DEFAULT_DIR
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            best = run = 0
            last = token = None
            for value in self.output_ids:
                run = run + 1 if value == last else 1
                if run > best: best, token = run, value
                last = value
            payload = dict(schema=1, request_id=self.request_id,
                started_at=self.started, saved_at=time.time(), plan=self.plan,
                sampler='batched-nucleus-inverse-cdf-v1',
                input_ids=self.input_ids, output_ids=self.output_ids,
                output_text=tokenizer.decode(self.output_ids, skip_special_tokens=False),
                terminal=self.terminal, decode_stats=getattr(queue, 'decode_stats', None),
                longest_same_token_run=dict(token_id=token, count=best))
            name = f"{time.time_ns()}_{uuid.uuid4().hex}.json"
            temporary = directory / (name + '.tmp')
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
            path = directory / name
            os.replace(temporary, path)
            logging.warning('long generation saved: id=%s tokens=%d path=%s',
                            self.request_id, len(self.output_ids), path)
        except Exception:
            # Disk-full/permission failure must not fail a completed inference.
            logging.exception('long generation save failed: id=%s', self.request_id)
