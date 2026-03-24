"""
Error checkers reusable across any equation-solving goal.

Each check follows a two-phase structure:
  Phase 1 — Pattern match: does this transition look like the error we detect?
             Return None immediately if not — let the next checker try.
  Phase 2 — Diagnosis: describe specifically what went wrong.

These are goal-agnostic: they detect algebraic mistakes that can occur
regardless of whether the equation has fractions, brackets, etc.
"""

from sympy import Eq, expand, simplify
from ..parser import x
from .base import ErrorChecker


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def _check_sign_error_on_move(prev_eq: Eq, new_eq: Eq) -> str | None:
    """
    Detect: student moved a constant term across the equals sign but kept its sign.
    e.g.  x + 2 = 6  →  x = 8  (added 2 to RHS instead of subtracting)

    Phase 1 — Pattern match: a term-move is recognised when a constant
    disappears from the LHS (lhs_removed is a nonzero number) and the RHS
    changes by exactly that same amount (rhs_added == lhs_removed). If the
    signs had been flipped correctly the step would have been marked correct
    and never reached error diagnosis.

    Phase 2 — Diagnosis: report the correct RHS value.
    """
    # Phase 1: did the student remove a constant from the LHS?
    try:
        lhs_removed = expand(prev_eq.lhs - new_eq.lhs)
        rhs_added = expand(new_eq.rhs - prev_eq.rhs)

        if not (lhs_removed.is_number and lhs_removed != 0):
            return None  # no constant was removed from LHS — not a term-move step

        # Phase 2: did the RHS change by the same amount (sign not flipped)?
        if simplify(lhs_removed - rhs_added) == 0:
            correct_rhs = prev_eq.rhs - lhs_removed
            return (
                f"When moving a term to the other side its sign must flip. "
                f"You moved {lhs_removed} to the right but added it — "
                f"it should be subtracted. "
                f"The right side should become {correct_rhs}, not {new_eq.rhs}."
            )
    except Exception:
        pass
    return None


def _check_wrong_division(prev_eq: Eq, new_eq: Eq) -> str | None:
    """
    Detect: student divided by the wrong number when isolating x.
    e.g.  5x = 30  →  x = 15  (divided by 2 instead of 5)

    Phase 1 — Pattern match: only applies when the new step is of the form
    x = <number>, which is the final isolation step.

    Phase 2 — Diagnosis: compute the correct answer and report the discrepancy.
    """
    # Phase 1: is the new step of the form  x = <number>?
    try:
        if new_eq.lhs != x or not new_eq.rhs.is_number:
            return None  # not an isolation step — not applicable

        coeff = prev_eq.lhs.coeff(x)
        if not coeff or not coeff.is_number or coeff == 1:
            return None  # no numeric coefficient to divide by

        # Phase 2: did the student divide by the correct coefficient?
        expected = simplify(prev_eq.rhs / coeff)
        if simplify(new_eq.rhs - expected) != 0:
            return (
                f"You divided by the wrong number. "
                f"The coefficient of x is {coeff}, so both sides must be "
                f"divided by {coeff}. The answer should be x = {expected}, "
                f"not x = {new_eq.rhs}."
            )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Exported ErrorChecker instances
# ---------------------------------------------------------------------------

sign_error_on_move = ErrorChecker(
    id="sign_error_on_move",
    description="Student moved a term across the equals sign without flipping its sign.",
    check=_check_sign_error_on_move,
)

wrong_division = ErrorChecker(
    id="wrong_division",
    description="Student divided by the wrong coefficient when isolating x.",
    check=_check_wrong_division,
)
