"""
Partial-factoring feedback checkers for FactorExpressionGoal.

These checkers receive (prev: Expr, student: Expr) where the student's step
is algebraically CORRECT but not yet in "best" factored form. (Defined as biggest (negative) factor taken out)

These checks also fire on correct steps (unlike other errorcheckers available)

New ones can be added: 

1. Write a function  _check_<name>(prev, student) -> str | None
2. Wrap it in an ErrorChecker at the bottom of this file.
3. Import it in factor_expression.py and add to FactorExpressionGoal.partial_checks.
"""

from sympy import Mul, factor, expand, latex as _latex
from .base import ErrorChecker


def _check_positive_factor_for_negative_leading(prev, student) -> str | None:
    """
    Detects: student factored out a positive number when the canonical
    factored form has a negative leading factor.

    Example:  -5x - 5  →  student enters 5(-x - 1)
              canonical is -5(x + 1), so the student should write -5(x + 1).

    Phase 1: canonical form of the original has a negative numeric factor.
    Phase 2: student's form has a positive numeric factor.
    """
    # Check if factored form has a negative coefficient
    canonical = factor(expand(prev))
    if not isinstance(canonical, Mul):
        return None
    canonical_num = next((f for f in canonical.args if f.is_number), None)
    if canonical_num is None or canonical_num >= 0:
        return None

    # check if the students answer have a positive coefficient
    if not isinstance(student, Mul):
        return None
    student_num = next((f for f in student.args if f.is_number), None)
    if student_num is None or student_num <= 0:
        return None

    return (
        "Als de eerste term negatief is, haal je altijd de grootste negatieve factor buiten haakjes: "
        f"het wordt dus \\({_latex(canonical)}\\) in plaats van \\({_latex(student)}\\)."
    )


# ---------------------------------------------------------------------------
# Exported ErrorChecker instances
# ---------------------------------------------------------------------------

positive_factor_for_negative_leading = ErrorChecker(
    id="positive_factor_for_negative_leading",
    description=(
        "Student factored out a positive number for a negative-leading expression "
        "(e.g. 5(-x-1) instead of -5(x+1))."
    ),
    check=_check_positive_factor_for_negative_leading,
)
