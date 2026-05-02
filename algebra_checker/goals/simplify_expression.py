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

from sympy import Add, expand, simplify as sym_simplify, Symbol, Pow, latex as sym_latex

from ..parser import ParseError, parse_equation, parse_expr_safe
from ..validator import StepResult, StepStatus
from .base import Goal
from ..errors.expression_errors import wrong_variable, wrong_sign_combining_x, wrong_sign_combining_const, wrong_unlike_term_addition
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
    True when the expression contains no combinable like terms and no
    unevaluated numeric operations.

    For symbolic expressions: done when expand() produces the same number
    of terms (no like terms left to combine).
    For purely numeric expressions: done only when the result is an atomic
    number — e.g. 9 is done, but (7-4)**2 or 3**2 or 36-3 are not.
    """
    if _term_count(expr) != _term_count(expand(expr)):
        return False
    if expr.is_number:
        if _term_count(expr) > 1:  # unevaluated sum like 3 + 12
            return False
        return not any(n.exp.is_positive for n in expr.atoms(Pow))
    return True


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


def _parse_step(raw: str):
    """Parse a step as either 'y = expr' or a bare expression."""
    if '=' in raw.replace('==', ''):
        eq = parse_equation(raw)
        return _extract_rhs(eq, raw)
    return parse_expr_safe(raw)


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

    default_error_checks = [wrong_variable, wrong_sign_combining_x, wrong_sign_combining_const, wrong_unlike_term_addition]
    input_hint = (
        r"\begin{array}{l}"
        r"\text{Voer elke stap in.}\\[6pt]"
        r"y = 5x + 4x - 6 \\"
        r"y = 9x - 6"
        r"\end{array}"
    )
    selftest_input_hint = (
        r"\begin{array}{l}"
        r"\text{Voer alleen je antwoord in. Bijvoorbeeld:}\\[6pt]"
        r"y = 2x + 1 \\ \\"
        r"\text{of}\\ \\"
        r"3q + 4 + 2f"
        r"\end{array}"
    )
    @classmethod
    def get_expected_answer(cls, item: dict) -> str | None:
        try:
            sympy_str = item['sympy_str']
            if '=' in sympy_str.replace('==', ''):
                # Form: y = expr
                eq = parse_equation(sympy_str)
                lhs = eq.lhs if eq.lhs.is_Symbol else eq.rhs
                rhs = eq.rhs if eq.lhs.is_Symbol else eq.lhs
                return f'{sym_latex(lhs)} = {cls.latex_expr(sym_simplify(expand(rhs)))}'
            else:
                # Form: expr
                expr = parse_expr_safe(sympy_str)
                return cls.latex_expr(sym_simplify(expand(expr)))
        except Exception:
            return None

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
        # 1. Parse both steps (accepts  y = <expr>  or a bare expression)
        try:
            prev_rhs = _parse_step(prev_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Er is iets mis met je invoer.",
            )
        try:
            new_rhs = _parse_step(new_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Er is iets mis met je invoer.",
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
