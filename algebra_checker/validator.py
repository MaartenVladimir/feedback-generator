"""
Step validator using symbolic equivalence checking.

The core insight: instead of enumerating all correct patterns, we check
whether the student's new expression/equation is algebraically equivalent
to the previous one. SymPy handles this via simplification.

For equations (balance method):
  - Both the original and new equation must have the same solution set.
  - We verify this by solving both and comparing solutions.
  - We also check structural equivalence as a fast path.

For expressions (expand/factor/simplify):
  - The new expression must be algebraically identical to the original.
  - Checked via: simplify(original - new) == 0
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Set

import sympy
from sympy import (
    Eq, Expr, Symbol, simplify, expand,
    solve, S, oo, zoo, nan, nsimplify
)

from .parser import x


class StepStatus(Enum):
    CORRECT = auto()
    INCORRECT = auto()
    PARSE_ERROR = auto()
    EQUIVALENT_BUT_NO_PROGRESS = auto()  # valid but didn't move forward
    COMPLETE = auto()  # problem is solved
    CORRECT_SUBOPTIMAL = auto()  # correct but strategically poor


class StrategyRating(Enum):
    """How well the student's step serves the overall goal."""
    OPTIMAL = auto()          # Good move toward the goal
    SUBOPTIMAL = auto()       # Correct but not the best path
    COUNTERPRODUCTIVE = auto()  # Correct but actively moves away from goal


@dataclass
class StepResult:
    """Result of validating a single student step."""
    status: StepStatus
    is_correct: bool
    message: str = ""
    # What the student's step looks like canonically
    canonical_form: str = ""
    # If we detected what transformation they applied
    transformation: Optional[str] = None
    # If incorrect: stable id of the matched ErrorChecker (for system logs)
    error_id: Optional[str] = None
    # If incorrect: human-readable diagnosis (filled in by errors module)
    error_diagnosis: Optional[str] = None
    # Additional hints
    hints: List[str] = field(default_factory=list)
    # Strategic assessment (set by goal-level checkers)
    strategy_rating: Optional[StrategyRating] = None
    strategy_message: str = ""


# ---------------------------------------------------------------------------
# Equivalence checking
# ---------------------------------------------------------------------------

def _expressions_equivalent(a: Expr, b: Expr) -> bool:
    """Check if two expressions are algebraically equivalent."""
    try:
        diff = simplify(expand(a) - expand(b))
        return diff == 0 or simplify(diff) == 0
    except Exception:
        return False


def _equations_equivalent(eq1: Eq, eq2: Eq) -> bool:
    """
    Check if two equations have the same solution set.

    Strategy:
    1. Fast path: check if LHS-RHS of both simplify to the same thing.
    2. Slow path: solve both and compare solution sets.
    """
    # Fast path: rewrite as LHS - RHS = 0 and compare
    expr1 = eq1.lhs - eq1.rhs
    expr2 = eq2.lhs - eq2.rhs
    if _expressions_equivalent(expr1, expr2):
        return True

    # Check if one is a constant multiple of the other
    try:
        ratio = simplify(expr1 / expr2)
        if ratio.is_number and ratio != 0:
            return True
    except Exception:
        pass

    # Slow path: compare solution sets (variable-agnostic)
    try:
        var_syms = eq1.free_symbols | eq2.free_symbols
        var = next(iter(var_syms)) if len(var_syms) == 1 else x
        sols1 = set(solve(eq1, var))
        sols2 = set(solve(eq2, var))
        if sols1 == sols2 and len(sols1) > 0:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Completion checking
# ---------------------------------------------------------------------------

def _is_solved_equation(eq: Eq) -> bool:
    """Check if equation is in the form var = <number> or <number> = var."""
    lhs, rhs = eq.lhs, eq.rhs

    if lhs.is_Symbol and rhs.is_number:
        return True
    if rhs.is_Symbol and lhs.is_number:
        return True
    return False


# ---------------------------------------------------------------------------
# Main validation entry points
# ---------------------------------------------------------------------------

def validate_equation_step(
    prev_eq: Eq,
    new_eq: Eq,
) -> StepResult:
    """
    Validate a step in equation solving.

    Checks:
    1. Are the equations equivalent (same solution set)?
    2. Is the new equation "solved" (x = number)?
    """
    equivalent = _equations_equivalent(prev_eq, new_eq)

    if not equivalent:
        return StepResult(
            status=StepStatus.INCORRECT,
            is_correct=False,
            message="Deze stap is incorrect.",
            canonical_form=str(new_eq),
        )

    if _is_solved_equation(new_eq):
        return StepResult(
            status=StepStatus.COMPLETE,
            is_correct=True,
            message="Juist! De vergelijking is opgelost!",
            canonical_form=str(new_eq),
        )

    return StepResult(
        status=StepStatus.CORRECT,
        is_correct=True,
        message="Juist!",
        canonical_form=str(new_eq),
    )


def validate_expression_step(
    prev_expr: Expr,
    new_expr: Expr,
) -> StepResult:
    """
    Validate a step in expression manipulation.

    Only checks algebraic equivalence — completion is goal-specific and
    handled by the calling Goal subclass after this returns.
    """
    equivalent = _expressions_equivalent(prev_expr, new_expr)

    if not equivalent:
        return StepResult(
            status=StepStatus.INCORRECT,
            is_correct=False,
            message="Er zit in een fout in je antwoord.",
            canonical_form=str(new_expr),
        )

    if str(simplify(prev_expr)) == str(simplify(new_expr)):
        return StepResult(
            status=StepStatus.EQUIVALENT_BUT_NO_PROGRESS,
            is_correct=True,
            message="Je stap klopt. Maar brengt je nog niet dichterbij de oplossing.",
            canonical_form=str(new_expr),
        )

    return StepResult(
        status=StepStatus.CORRECT,
        is_correct=True,
        message="Juist!",
        canonical_form=str(new_expr),
    )


def validate_step(prev, new) -> StepResult:
    """
    Unified validation: auto-detects equations vs expressions.

    Only checks correctness (equivalence). Completion and strategy
    are handled by the calling Goal subclass.

    Parameters
    ----------
    prev : Eq or Expr
        The previous state.
    new : Eq or Expr
        The student's new step.
    """
    if isinstance(prev, Eq) and isinstance(new, Eq):
        return validate_equation_step(prev, new)
    elif isinstance(prev, Expr) and isinstance(new, Expr):
        return validate_expression_step(prev, new)
    else:
        return StepResult(
            status=StepStatus.PARSE_ERROR,
            is_correct=False,
            message="Er is een probleem met je antwoord.",
        )