"""
Goal: simplify (herleiden) an algebraic expression step-by-step.

The item stores the starting expression as an equation  y = f(x)  in
sympy_str, for example:

    y = 5*x + 4*x - 6

The student rewrites the RHS step by step (combining like terms) until
the expression is fully simplified, e.g.  y = 9*x - 6.

Correctness
-----------
Each intermediate step must be algebraically equivalent to the previous
one (same expression, just rewritten / partially simplified).

Completion
----------
The expression is fully simplified when the student's RHS has no more
like terms to combine.  We detect this by comparing the number of terms
in the student's input (parsed with evaluate=False, so like terms are
NOT automatically combined) against the number of terms in its fully
expanded form (where SymPy *does* combine like terms).

  - 5x + 4x - 6  has 3 terms, expand gives 9x - 6 (2 terms) → not done
  - 9x - 6       has 2 terms, expand gives 9x - 6 (2 terms) → done ✓

Error detection
---------------
Wrong sign when combining x-terms or constant terms (see expression_errors.py).
"""

from sympy import Add, expand, simplify as sym_simplify, Symbol

from ..parser import ParseError, parse_equation
from ..validator import StepResult, StepStatus
from .base import Goal
from ..errors.expression_errors import wrong_sign_combining_x, wrong_sign_combining_const
from ..validator import _expressions_equivalent   # shared equivalence helper


_FALLBACK_INCORRECT = (
    "Deze stap is niet correct. "
    "Controleer de tekens en de berekening bij het samenvoegen."
)
_MSG_CORRECT   = "Correct!"
_MSG_COMPLETE  = "Correct! De uitdrukking is volledig herleid."
_MSG_NO_PROGRESS = (
    "Dit is hetzelfde als de vorige stap. "
    "Probeer de gelijksoortige termen samen te voegen."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _term_count(expr) -> int:
    """Number of top-level additive terms in an expression."""
    return len(Add.make_args(expr))


def _is_fully_simplified(expr) -> bool:
    """
    True when the expression contains no combinable like terms.

    Strategy: if SymPy's expand() (which combines like terms) produces a
    result with the same number of terms as the student's input, there is
    nothing left to combine.
    """
    return _term_count(expr) == _term_count(expand(expr))


def _extract_rhs(eq, raw: str):
    """
    Return the expression side of a  y = <expr>  equation.

    Accepts both orientations (y on left or right).
    Raises ParseError if neither side is y.
    """
    if eq.lhs.is_Symbol:
        return eq.rhs
    if eq.rhs.is_Symbol:
        return eq.lhs
    raise ParseError(raw, "Verwacht een vergelijking van de vorm  <variabele> = <uitdrukking>")


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

class SimplifyExpressionGoal(Goal):
    """
    Goal: simplify an algebraic expression (herleiden).

    Items must provide a starting expression in  sympy_str  as
        y = <expression in x>
    Students enter each simplification step in the same format.
    """

    default_error_checks = [wrong_sign_combining_x, wrong_sign_combining_const]

    @property
    def description(self) -> str:
        return "Herleid de uitdrukking"

    # Override diagnose_error to use an expression-specific fallback message
    # instead of the equation-oriented one from the base class.
    def diagnose_error(self, prev_rhs, new_rhs):
        for checker in self.error_checks:
            msg = checker.check(prev_rhs, new_rhs)
            if msg:
                return checker.id, msg
        return None, _FALLBACK_INCORRECT

    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        # 1. Parse both steps as  y = <expr>  equations
        try:
            prev_eq = parse_equation(prev_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Kan de vorige stap niet lezen: {e.reason}",
            )
        try:
            new_eq = parse_equation(new_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Kan jouw stap niet lezen: {e.reason}",
            )

        # 2. Extract the expression side (RHS of  y = <expr>)
        try:
            prev_rhs = _extract_rhs(prev_eq, prev_raw)
            new_rhs  = _extract_rhs(new_eq,  new_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=e.reason,
            )

        # 3. Correctness: is the new expression equivalent to the previous one?
        if not _expressions_equivalent(prev_rhs, new_rhs):
            error_id, diagnosis = self.diagnose_error(prev_rhs, new_rhs)
            return StepResult(
                status=StepStatus.INCORRECT,
                is_correct=False,
                error_id=error_id,
                error_diagnosis=diagnosis,
                message=diagnosis,
            )

        # 4. No-progress guard: student re-entered the same thing
        if str(prev_raw).strip() == str(new_raw).strip():
            return StepResult(
                status=StepStatus.EQUIVALENT_BUT_NO_PROGRESS,
                is_correct=True,
                message=_MSG_NO_PROGRESS,
            )

        # 5. Completion: is the new RHS fully simplified?
        if _is_fully_simplified(new_rhs):
            return StepResult(
                status=StepStatus.COMPLETE,
                is_correct=True,
                message=_MSG_COMPLETE,
            )

        # 6. Correct intermediate step
        return StepResult(
            status=StepStatus.CORRECT,
            is_correct=True,
            message=_MSG_CORRECT,
        )
