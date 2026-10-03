"""Execution/strategy finish ownership, independent of model numerical parity."""
import threading
import pytest
from model.model_api import ModelExecution, PrefillCancelled
from model.past import SlotPool
from strategy.strategy import Strategy


class Compute:
    def forward(self, tokens, *, past, slot, start, history_tokens=()):
        return None

    def finish_prefill(self, *, past, slot):
        return ('finished', past.pos[slot])


@pytest.mark.parametrize('failure', ['before_cancel', 'after_cancel', 'commit', 'compute'])
def test_finish_failure_closes_session(failure):
    cancel = threading.Event()
    past = SlotPool(1, 8)
    compute = Compute()
    strategy = Strategy(ModelExecution(compute, past), namespace='finish-contract')
    with strategy.open((7, 8), cancel=cancel) as session:
        session.run()
        calls = []
        def finish(*, past, slot):
            calls.append(slot)
            if failure == 'after_cancel': cancel.set()
            if failure == 'commit': past.set_pos(slot, 3)
            if failure == 'compute': raise RuntimeError('injected compute failure')
        compute.finish_prefill = finish
        if failure == 'before_cancel': cancel.set()
        expected = PrefillCancelled if 'cancel' in failure else RuntimeError
        with pytest.raises(expected): session.finish_prefill()
        assert len(calls) == (failure != 'before_cancel')
        assert session.closed and session.slot in past.free_slots
        assert past.pos[session.slot] == 0


def test_finish_empty_or_released_model_slot_rejected():
    past = SlotPool(1, 8)
    execution = ModelExecution(Compute(), past)
    with pytest.raises(ValueError): execution.finish_prefill(0)
    with pytest.raises(ValueError): execution.finish_prefill(-1)
    with pytest.raises(ValueError): execution.finish_prefill(1)
