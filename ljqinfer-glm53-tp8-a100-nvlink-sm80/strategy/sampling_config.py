"""CPU-only validation shared by HTTP, scheduling and sampling runtime."""
import math

# Single-turn constrained sampling; explicit temperature=0 keeps greedy.
DEFAULT_TEMPERATURE = 0.5


def sampling_config(*, temperature=0., top_p=.95, seed=None):
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise ValueError('temperature must be finite and nonnegative')
    if isinstance(top_p, bool) or top_p != .95:
        raise ValueError('only top_p=0.95 is supported')
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('seed must be an integer in [0, 2**63)')
    return dict(temperature=float(temperature), top_p=.95, seed=seed)
