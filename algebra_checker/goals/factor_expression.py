"""
Goal: factor an algebraic expression (ontbinden in factoren).

The item stores the starting expression in sympy_str.
Students enter the factored form, optionally in multiple steps.

A final answer is defined as an expression of irreducible factors.
For example: (x+1)(4x + 4) is not final, because 4x + 4 is reducible.


Partial-factoring feedback (partial_checks)
-------------------------------------------
When a step is algebraically correct but not yet complete, the goal runs
the checkers in `partial_checks` to produce a specific yellow warning
instead of a silent green tick.  These are ErrorChecker objects from
errors/factor_expression_errors.py
"""

from sympy import Add, Mul, Pow, factor, expand

from ..parser import ParseError, parse_expr_safe
from ..validator import StepResult, StepStatus, _expressions_equivalent
from ..errors.base import ErrorChecker
from ..errors.factor_expression_errors import positive_factor_for_negative_leading
from .base import Goal


# ---------------------------------------------------------------------------
# Primitivity helpers
# ---------------------------------------------------------------------------

def _subexpr_is_irreducible(f) -> bool:
    """True if f cannot be factored into a non-trivial product."""
    if f.is_number or f.is_Symbol:
        return True
    if isinstance(f, Pow):
        return _subexpr_is_irreducible(f.base)
    if isinstance(f, Add):
        # An Add is irreducible iff factor() gives back an Add (not a Mul).
        return isinstance(factor(f), Add)
    # Mul or other: expand and re-factor; irreducible if result is an Add.
    return isinstance(factor(expand(f)), Add)


def _is_primitive_factored(expr) -> bool:
    """
    True if expr is in primitive factored form (no factor can be further factored).

    An expression is primitive factored when:
      - it is NOT an Add (i.e. it has been factored into a product), AND
      - every non-trivial factor is irreducible.
    """
    if isinstance(expr, Add):
        return False
    if isinstance(expr, Mul):
        return all(_subexpr_is_irreducible(f) for f in expr.args)
    if isinstance(expr, Pow):
        return _subexpr_is_irreducible(expr.base)
    # Single atom (Symbol, Number) is trivially factored.
    return True


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

class FactorExpressionGoal(Goal):
    """
    Goal: factor an algebraic expression (ontbinden in factoren).

    Works for both 2-term (extract GCF) and 3-term (quadratic) factoring.
    Accepts intermediate partially-factored steps; completes when the
    expression is in primitive factored form.

    partial_checks
        ErrorChecker objects that run when a step is correct but not yet
        complete.  They return a targeted yellow warning message rather than
        silently accepting the step.  Add new checks in
        errors/factor_expression_errors.py and register them here.
    """

    default_error_checks: list[ErrorChecker] = []

    partial_checks: list[ErrorChecker] = [
        positive_factor_for_negative_leading,
    ]

    input_hint = (
        r"\begin{array}{l}"
        r"\text{Ontbind de uitdrukking stap voor stap in factoren.}\\"
        r"4x + 8 \\"
        r"4(x + 2)"
        r"\end{array}"
    )
    selftest_input_hint = (
        r"\begin{array}{l}"
        r"\text{Voer alleen je eindantwoord in. Bijvoorbeeld:}\\"
        r"4(x + 2)"
        r"\end{array}"
    )

    @classmethod
    def get_expected_answer(cls, item: dict) -> str | None:
        try:
            expr = parse_expr_safe(item['sympy_str'])
            return cls.latex_expr(factor(expr))
        except Exception:
            return None

    @property
    def description(self) -> str:
        return "Ontbind de uitdrukking in factoren."

    def _run_partial_checks(
        self, prev_expr, student_expr
    ) -> StepResult | None:
        """
        Run partial_checks and return a CORRECT_SUBOPTIMAL StepResult if any
        check fires, otherwise None.
        """
        for checker in self.partial_checks:
            msg = checker.check(prev_expr, student_expr)
            if msg:
                return StepResult(
                    status=StepStatus.CORRECT_SUBOPTIMAL,
                    is_correct=True,
                    error_id=checker.id,
                    message=msg,
                )
        return None

    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        try:
            prev_expr = parse_expr_safe(prev_raw)
        except ParseError:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message="Er is iets mis met je invoer.",
            )
        try:
            student_expr = parse_expr_safe(new_raw)
        except ParseError:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message="Er is iets mis met je invoer.",
            )

        # check algebraic equivalence with the previous step
        if not _expressions_equivalent(prev_expr, student_expr):
            error_id, diagnosis = self.diagnose_error(prev_expr, student_expr)
            return StepResult(
                status=StepStatus.INCORRECT,
                is_correct=False,
                error_id=error_id,
                error_diagnosis=diagnosis,
                message=diagnosis,
            )

        # catch repeat submission
        if str(prev_raw).strip() == str(new_raw).strip():
            return StepResult(
                status=StepStatus.EQUIVALENT_BUT_NO_PROGRESS,
                is_correct=True,
                message=(
                    "Dit is gelijk aan de vorige stap. "
                    "Probeer de grootste gemeenschappelijke factor buiten haakjes te halen."
                ),
            )

        # check if expr is an Add
        if isinstance(student_expr, Add):
            return StepResult(
                status=StepStatus.EQUIVALENT_BUT_NO_PROGRESS,
                is_correct=True,
                message=(
                    "Dit klopt, maar de uitdrukking is nog niet ontbonden. "
                    "Haal de grootste gemeenschappelijke factoren buiten haakjes."
                ),
            )

        # check if expr is fully factored
        if _is_primitive_factored(student_expr):
            partial = self._run_partial_checks(prev_expr, student_expr)
            if partial:
                return partial
            return StepResult(
                status=StepStatus.COMPLETE,
                is_correct=True,
                message="Juist! De uitdrukking is volledig ontbonden in factoren.",
            )

        # expression is algebraically correct and progress has been made, but not yet fully factored
        partial = self._run_partial_checks(prev_expr, student_expr)
        if partial:
            return partial

        return StepResult(
            status=StepStatus.CORRECT_SUBOPTIMAL,
            is_correct=True,
            message=(
                "Correct, maar nog niet het eindantwoord. "
                "Kun je nog een grotere gemeenschappelijke factor vinden?"
            ),
        )
