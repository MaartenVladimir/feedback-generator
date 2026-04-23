"""
Goal: Solve a quadratic (or factored) equation for x.

Handles multi-branch solution paths where intermediate steps and the final
answer may be disjunctions

Both prev_raw and new_raw may be a single equation or a disjunction.
Correctness is checked by comparing solution sets via SymPy's solve().
"""

from sympy import Eq, solve
from typing import Union, List, Set

from ..parser import (
    ParseError,
    parse_equation,
    parse_disjunction,
    is_disjunction,
    x,
)
from ..validator import StepResult, StepStatus, StrategyRating
from .base import Goal
from ..errors import error_checkers, strategy_checks

Step = Union[Eq, List[Eq]]

def _parse_step(raw: str) -> Step:
    """Parse a raw string into a single Eq or a list of Eq (disjunction)."""
    if is_disjunction(raw):
        return parse_disjunction(raw)
    return parse_equation(raw)


def _solution_set(step: Step) -> Set:
    """Extract the full solution set from a step (single Eq or list of Eq)."""
    eqs = step if isinstance(step, list) else [step]
    sols: Set = set()
    for eq in eqs:
        try:
            sols.update(solve(eq, x))
        except Exception:
            pass
    return sols


def _is_complete(step: Step) -> bool:
    """True if every branch is in the form x=<number> or <number>=x."""
    eqs = step if isinstance(step, list) else [step]
    return all(
        (eq.lhs == x and eq.rhs.is_number) or
        (eq.rhs == x and eq.lhs.is_number)
        for eq in eqs
    )


class SolveQuadraticEquationGoal(Goal):
    """
    Goal: reduce a factored / quadratic equation to  x=a  v  x=b  (or x=a).

    Correctness is checked by solution-set equivalence, so it handles:
      single → single          normal algebraic step within one branch
      single → disjunction     zero-product split
      disjunction → disjunction  solving each branch
    """

    default_error_checks = error_checkers

    @property
    def description(self) -> str:
        return "Solve the quadratic equation for x"

    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        # 1. Parse
        try:
            prev = _parse_step(prev_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Could not parse previous step: {e.reason}",
            )
        try:
            new = _parse_step(new_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Could not parse your step: {e.reason}",
            )

        # 2. Compare solution sets
        try:
            prev_sols = _solution_set(prev)
            new_sols = _solution_set(new)
        except Exception:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message="Could not evaluate the solution set of your step.",
            )

        if prev_sols != new_sols:
            # Only run error templates when both steps are single equations
            # (templates cannot match disjunction steps)
            error_id, diagnosis = (None, None)
            if isinstance(prev, Eq) and isinstance(new, Eq):
                error_id, diagnosis = self.diagnose_error(prev, new)
            return StepResult(
                status=StepStatus.INCORRECT,
                is_correct=False,
                message=diagnosis or "Deze stap is incorrect. Probeer het nog eens.",
                error_id=error_id,
                error_diagnosis=diagnosis,
            )

        # 3. Completion check
        if _is_complete(new):
            return StepResult(
                status=StepStatus.COMPLETE,
                is_correct=True,
                message="Correct! You've solved the equation.",
            )

        # 4. Strategy assessment — only for single-equation steps
        if isinstance(prev, Eq) and isinstance(new, Eq):
            for check_fn in strategy_checks:
                outcome = check_fn(prev, new)
                if outcome is not None:
                    msg, rating = outcome
                    status = StepStatus.CORRECT_SUBOPTIMAL if rating == StrategyRating.COUNTERPRODUCTIVE else StepStatus.CORRECT
                    return StepResult(
                        status=status,
                        is_correct=True,
                        strategy_rating=rating,
                        strategy_message=msg,
                        message=msg,
                    )

        return StepResult(
            status=StepStatus.CORRECT,
            is_correct=True,
            message="Correct step.",
        )
