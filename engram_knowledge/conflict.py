"""A conflict is worked out from the stored observations each time. It is never stored, so it cannot go stale."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Conflict:
    kind: str  # value: different values at once; date: one value given different start dates; ended: closes a value not held
    slot: str
    values: tuple
    fact_ids: tuple
    sources: tuple
