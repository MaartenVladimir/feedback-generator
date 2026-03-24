"""
Goal: Solve an equation for x.

Strategic feedback targets the two main forms of procedural over-reliance
that appear in the thesis context:

1. Zero-product property missed
   e.g.  (x - 1)(x + 3) = 0  →  student expands instead of reading off roots.
   Also covers:  (x + 1)^2 = 0  →  student should recognise x = -1 directly.

2. Perfect-square / square-root opportunity missed
   e.g.  (x + 1)^2 = 4  →  take sqrt both sides (x + 1 = ±2),
   instead of expanding to a general quadratic.

3. General complexity increase
   Any step that increases the number of terms without recognisably moving
   toward x = <value> gets a SUBOPTIMAL rating.
"""

from sympy import (
    Eq, Mul, Pow, S, expand, factor, simplify, solve
)
from sympy.core.add import Add

from ..parser import ParseError, parse_equation
from ..validator import (
    StepResult, StepStatus, StrategyRating,
    validate_equation_step,
)
from ..parser import x
from .base import Goal


# ---------------------------------------------------------------------------
# Structural pattern detectors  (operate on the *previous* equation)
# ---------------------------------------------------------------------------

def _rhs_is_zero(eq: Eq) -> bool:
    return eq.rhs == S.Zero or simplify(eq.rhs) == S.Zero

def _lhs_is_zero(eq: Eq) -> bool:
    return eq.lhs == S.Zero or simplify(eq.rhs) == S.zero

def _has_zero_product_structure(eq: Eq) -> bool:
    """
    True when LHS or RHS is a product whose factors (involving x) each have degree ≤ 2
    and RHS = 0.  In this case the zero-product property applies directly.
    """
    if _rhs_is_zero(eq):
        lhs_factored = factor(eq.lhs)
    elif _lhs_is_zero:
        lhs_factored = factor(eq.rhs)  # zero side is treated as "rhs"
    else:
        return False

    # Pow with base containing x and exponent ≥ 2: (x+a)^n = 0
    if isinstance(lhs_factored, Pow):
        if x in lhs_factored.base.free_symbols and lhs_factored.exp.is_integer:
            return True

    # Mul: look for at least two factors involving x
    if isinstance(lhs_factored, Mul):
        x_factors = [
            f for f in lhs_factored.args
            if x in f.free_symbols
        ]
        if len(x_factors) >= 2:
            return True

    return False


def _has_perfect_square_opportunity(eq: Eq) -> bool:
    """
    True when equation has the form  a*(x + b)^2 = c  (with c a number ≥ 0).
    Taking the square root is the efficient path; expanding is not.
    """
    if eq.rhs.is_number:
        lhs, rhs = eq.lhs, eq.rhs
    elif eq.lhs.is_number:
        lhs, rhs = eq.rhs, eq.lhs
    else:
        return False
    
    # Direct form: (x + b)^2 = c
    if (
        isinstance(lhs, Pow)
        and lhs.exp == 2
        and x in lhs.base.free_symbols
        and lhs.base.as_poly(x).degree() == 1
    ):
        return True

    # Scaled form: a*(x + b)^2 = c
    if isinstance(lhs, Mul):
        pow_factors = [
            f for f in lhs.args
            if isinstance(f, Pow)
            and f.exp == 2
            and x in f.base.free_symbols
        ]
        numeric_factors = [f for f in lhs.args if f.is_number]
        if pow_factors and numeric_factors:
            return True

    return False


# ---------------------------------------------------------------------------
# Complexity metric  (lower = simpler = closer to solved)
# ---------------------------------------------------------------------------

def _term_count(eq: Eq) -> int:
    """Number of terms in the expanded LHS - RHS expression."""
    combined = expand(eq.lhs - eq.rhs)
    if isinstance(combined, Add):
        return len(combined.args)
    return 1


def _student_expanded(prev_eq: Eq, new_eq: Eq) -> bool:
    """True if the new equation has strictly more terms than the previous."""
    return _term_count(new_eq) > _term_count(prev_eq)


# ---------------------------------------------------------------------------
# Strategy analysis
# ---------------------------------------------------------------------------

def _assess_strategy(prev_eq: Eq, new_eq: Eq, result: StepResult) -> StepResult:
    """
    Augment a (correct) StepResult with strategy_rating and strategy_message.
    Operates on the *previous* equation to detect missed opportunities.
    """
    # --- Zero-product property missed ---
    if _has_zero_product_structure(prev_eq) and _student_expanded(prev_eq, new_eq):
        result.status = StepStatus.CORRECT_SUBOPTIMAL
        result.strategy_rating = StrategyRating.COUNTERPRODUCTIVE
        result.strategy_message = (
            "This step is correct, but expanding here makes the problem harder. "
            "Notice that the equation has the structure A \u00b7 B = 0. "
            "This means at least one factor must be zero — you can read off the "
            "solutions directly without expanding."
        )
        return result

    # --- Perfect-square opportunity missed ---
    if _has_perfect_square_opportunity(prev_eq) and _student_expanded(prev_eq, new_eq):
        result.status = StepStatus.CORRECT_SUBOPTIMAL
        result.strategy_rating = StrategyRating.COUNTERPRODUCTIVE
        result.strategy_message = (
            "This step is correct, but expanding loses a useful structure. "
            "The equation has the form (x + a)\u00b2 = c. "
            "Taking the square root of both sides gives x + a = \u00b1\u221ac, "
            "which is faster than expanding and applying the quadratic formula."
        )
        return result

    # --- General: did the step increase complexity without clear benefit? ---
    if _student_expanded(prev_eq, new_eq):
        result.strategy_rating = StrategyRating.SUBOPTIMAL
        result.strategy_message = (
            "This step is correct, but expanding has increased the number of terms. "
            "Consider whether there is a structural shortcut available."
        )
        return result

    # Step looks strategically sound
    result.strategy_rating = StrategyRating.OPTIMAL
    return result


# ---------------------------------------------------------------------------
# Goal class
# ---------------------------------------------------------------------------

from ..errors import sign_error_on_move, wrong_division

class SolveEquationGoal(Goal):
    """
    Goal: reduce an equation to  x = <value>  (or a set of values).

    check_step() returns a StepResult with:
      - correctness (equivalence of solution set)
      - completion detection (x = number)
      - strategic feedback (zero-product missed, perfect-square missed, etc.)
    """

    default_error_checks = [sign_error_on_move, wrong_division]

    @property
    def description(self) -> str:
        return "Solve the equation for x"

    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        # 1. Parse
        try:
            prev_eq = parse_equation(prev_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Could not parse previous step: {e.reason}",
            )
        try:
            new_eq = parse_equation(new_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Could not parse your step: {e.reason}",
            )

        # 2. Correctness check
        result = validate_equation_step(prev_eq, new_eq)

        if not result.is_correct:
            error_id, diagnosis = self.diagnose_error(prev_eq, new_eq)
            result.error_id = error_id
            result.error_diagnosis = diagnosis
            result.message = diagnosis
            return result

        if result.status == StepStatus.COMPLETE:
            return result

        # 3. Strategy analysis
        return _assess_strategy(prev_eq, new_eq, result)
