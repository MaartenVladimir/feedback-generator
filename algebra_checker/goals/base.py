"""
Abstract base class for all mathematical goals.

A Goal encapsulates the full check for a student step:
  1. Parse the raw strings.
  2. Check correctness (algebraic equivalence / solution preservation).
  3. Run registered error checks to diagnose *what* went wrong.
  4. Assess strategy (does this step move toward the goal well?).

Error checkers are declared as `default_error_checks` on the subclass and
can be extended per-instance via `extra_checks`. This means:
  - Common errors (sign flip, wrong division) live in errors/ and are
    imported by any goal that needs them.
  - Goal-specific errors are added to `default_error_checks`.
  - The DLE can instantiate a goal with additional checks for a specific
    problem: SolveLinearFractionGoal(extra_checks=[custom_check]).

The `id` on each ErrorChecker is logged by the DLE, giving you a stable
string (e.g. "partial_multiplication") per error type in the system logs.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..validator import StepResult
from ..errors.base import ErrorChecker

_FALLBACK_MESSAGE = (
    "This step does not preserve the equation's solutions. "
    "Check each operation carefully."
)


class Goal(ABC):
    """
    Base class for a mathematical goal.

    Subclasses declare which error checks they use by default:

        class SolveLinearFractionGoal(Goal):
            default_error_checks = [partial_multiplication, wrong_multiplier]

    Callers can extend this at instantiation time:

        goal = SolveLinearFractionGoal(extra_checks=[sign_error_on_move])

    Usage
    -----
    goal = SolveLinearFractionGoal()
    result = goal.check_step("x/2 + 1 = 3", "x + 1 = 3")

    result.is_correct          # False
    result.error_id            # "partial_multiplication"
    result.error_diagnosis     # "It looks like you multiplied by 2 but not every term..."
    """

    default_error_checks: list[ErrorChecker] = []

    def __init__(self, extra_checks: list[ErrorChecker] | None = None):
        self.error_checks: list[ErrorChecker] = (
            self.default_error_checks + (extra_checks or [])
        )

    def diagnose_error(self, prev, new) -> tuple[Optional[str], str]:
        """
        Run all registered error checks in order.

        Returns
        -------
        (error_id, message)
            error_id is the ErrorChecker.id of the first matching check,
            or None if no check matched (fallback message is returned).
        """
        for checker in self.error_checks:
            msg = checker.check(prev, new)
            if msg:
                return checker.id, msg
        return None, _FALLBACK_MESSAGE

    @abstractmethod
    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        """
        Full check of one student step.

        Parameters
        ----------
        prev_raw : str
            The previous state as the student entered it (or as stored).
        new_raw : str
            The student's new step.

        Returns
        -------
        StepResult
            Contains correctness, completion status, strategy assessment,
            and error_id for logging.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this goal (shown in UI)."""
