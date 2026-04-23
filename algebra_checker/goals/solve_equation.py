"""
Goal: Solve an equation for x.

All specific error and strategy patterns are defined as JSON templates in
algebra_checker/errors/templates/. Adding a new pattern requires only a new
.json file
"""

from sympy import Eq, expand
from sympy.core.add import Add

from ..parser import ParseError, parse_equation
from ..validator import (
    StepResult, StepStatus, StrategyRating,
    validate_equation_step,
)
from .base import Goal
from ..errors import error_checkers, strategy_checks


def _term_count(eq: Eq) -> int:
    """Number of terms in the expanded LHS - RHS expression."""
    combined = expand(eq.lhs - eq.rhs)
    if isinstance(combined, Add):
        return len(combined.args)
    return 1


def _student_expanded(prev_eq: Eq, new_eq: Eq) -> bool:
    """True if the new equation has strictly more terms than the previous."""
    return _term_count(new_eq) > _term_count(prev_eq)


def _assess_strategy(prev_eq: Eq, new_eq: Eq, result: StepResult) -> StepResult:
    """
    Augment a correct StepResult with strategy_rating and strategy_message.

    First runs all strategy templates (loaded from JSON). If none match,
    falls back to a general complexity-increase check.
    """
    for check_fn in strategy_checks:
        outcome = check_fn(prev_eq, new_eq)
        if outcome is not None:
            msg, rating = outcome
            if rating == StrategyRating.COUNTERPRODUCTIVE:
                result.status = StepStatus.CORRECT_SUBOPTIMAL
            result.strategy_rating  = rating
            result.strategy_message = msg
            return result

    # General fallback: any step that increases term count is mildly suboptimal
    if _student_expanded(prev_eq, new_eq):
        result.strategy_rating  = StrategyRating.SUBOPTIMAL
        result.strategy_message = (
            "Deze stap klopt, maar er is misschien een snellere manier."
        )
        return result

    result.strategy_rating = StrategyRating.OPTIMAL
    return result


class SolveEquationGoal(Goal):
    """
    Goal: reduce an equation to  x = <value>  (or a set of values).

    check_step() returns a StepResult with:
      - correctness (equivalence of solution set)
      - completion detection (x = number)
      - strategic feedback driven by JSON strategy templates
      - error diagnosis driven by JSON error templates
    """

    default_error_checks = error_checkers

    @property
    def description(self) -> str:
        return "Een vergelijking oplossen."

    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        # 1. Parse
        try:
            prev_eq = parse_equation(prev_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Er is een probleem met je antwoord: {e.reason}",
            )
        try:
            new_eq = parse_equation(new_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Er is een probleem met je antwoord: {e.reason}",
            )

        # 2. Correctness check
        result = validate_equation_step(prev_eq, new_eq)

        if not result.is_correct:
            error_id, diagnosis = self.diagnose_error(prev_eq, new_eq)
            result.error_id       = error_id
            result.error_diagnosis = diagnosis
            result.message        = diagnosis
            return result

        if result.status == StepStatus.COMPLETE:
            return result

        # 3. Strategy analysis
        return _assess_strategy(prev_eq, new_eq, result)
