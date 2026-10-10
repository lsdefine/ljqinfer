"""CPU-only soft thinking budget; preview never commits speculative tokens."""
from dataclasses import dataclass, replace

THINK_START = 128821
THINK_END = 128822
ASSISTANT = 128804
MAX_BIAS = 32.0


def validate_budget(value):
    if value is None:
        return 0
    if type(value) is not int or not 0 <= value <= 2**31 - 1:
        raise ValueError('thinking_budget_tokens must be a non-negative int32')
    return value


def end_bias(count, budget):
    if budget <= 0:
        return 0.0
    # No bias through 90% of budget; quadratically ramp to +32 at 100%, then hold.
    # This encourages </think> without forcing it; integer arithmetic keeps the 90% boundary exact.
    x = min(1.0, max(0.0, (10 * count - 9 * budget) / budget))
    return MAX_BIAS * x * x


@dataclass
class ThinkingBudget:
    budget: int = 0
    active: bool = False
    count: int = 0
    closed: bool = False

    @classmethod
    def from_prompt(cls, tokens, budget=None):
        state = cls(budget=validate_budget(budget))
        if not state.budget:
            return state
        # Only the current assistant suffix, never a previous turn's thinking.
        start = next((i for i in range(len(tokens)-1, -1, -1)
                      if tokens[i] == ASSISTANT), -1)
        if start >= 0:
            for token in tokens[start+1:]:
                state.observe(token)
            state.count = 0  # Budget counts generated tokens, not prompt text.
        return state

    def observe(self, token):
        if not self.budget or self.closed:
            return
        if token == THINK_END:
            self.active, self.closed = False, True
        elif token == THINK_START:
            self.active = True
        elif self.active:
            self.count += 1

    def accept(self, tokens):
        for token in tokens:
            self.observe(token)

    def preview(self, proposals):
        """Q6 input[0] is the already-observed anchor; row j predicts j+1."""
        state, result = replace(self), []
        for j, token in enumerate(proposals):
            if j:
                state.observe(token)
            result.append(end_bias(state.count, state.budget) if state.active else 0.0)
        return result
