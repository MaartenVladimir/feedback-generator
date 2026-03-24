"""
ErrorChecker — the unit of reusable error detection.

An ErrorChecker is a named, callable check that inspects two consecutive
student states (prev, new) and either returns a specific feedback message
or None if the error pattern is not present.

The `id` field is a stable string identifier used for logging / analytics,
so the DLE can record *which* error type a student made, not just the message.
"""

from dataclasses import dataclass
from typing import Callable, Optional
from sympy import Eq, Expr

# A check function takes the previous and new state and returns a message or None.
# Both arguments are the same type (Eq for equation goals, Expr for expression goals).
CheckFn = Callable[[object, object], Optional[str]]


@dataclass(frozen=True)
class ErrorChecker:
    """
    A named, reusable error pattern check.

    Attributes
    ----------
    id : str
        Stable identifier for logging, e.g. "partial_multiplication".
        Use snake_case. This is what gets recorded in system logs.
    description : str
        One-line human description of the error pattern.
        Useful for documentation and future problem-generation tooling.
    check : CheckFn
        Function (prev, new) -> str | None.
        Return a feedback message if the error is detected, else None.
    """
    id: str
    description: str
    check: CheckFn
