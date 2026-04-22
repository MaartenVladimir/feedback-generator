"""
Error checkers for expression-simplification goals (herleiden).

These checkers receive (prev_rhs: Expr, new_rhs: Expr) — the right-hand
side of the student's y = <expression> at two consecutive steps.

Because the parser uses evaluate=False, like terms in prev_rhs are NOT
already combined.  We can inspect the individual terms the student wrote
to detect whether a minus sign was dropped when combining them.

Two-phase structure (same as equation_errors.py):
  Phase 1 — Pattern match: does this transition look like the target error?
             Return None immediately if not, so the next checker gets a try.
  Phase 2 — Diagnosis: produce a specific Dutch-language feedback message.
"""

from sympy import Add, expand, simplify
from .base import ErrorChecker
from sympy import S

# Helper functions 

def _var_coefficients(expr, var) -> list:
    coeffs = []
    for term in Add.make_args(expr):
        c = term.coeff(var, 1)
        if c != 0:
            coeffs.append(c)
    return coeffs


def _const_terms(expr) -> list:
    """
    Return each purely numerical term in an unevaluated expression.

    For example:
        5x - 4 + 3  →  [-4, 3]
    """
    consts = []
    for term in Add.make_args(expr):
        if term.is_number:
            consts.append(term)
    return consts


# Check functions 

def _check_wrong_sign_x(prev_expr, new_expr) -> str | None:
    for var in prev_expr.free_symbols:
        coeffs = _var_coefficients(prev_expr, var)
        if len(coeffs) <= 1:
            continue
        numeric_coeffs = [c for c in coeffs if c.is_number]
        if not any(c < 0 for c in numeric_coeffs):
            continue
        correct_coeff = sum(numeric_coeffs)
        wrong_coeff   = sum(abs(c) for c in numeric_coeffs)
        if correct_coeff == wrong_coeff:
            continue
        try:
            new_coeff = expand(new_expr).coeff(var, 1)
            if simplify(new_coeff - wrong_coeff) == 0:
                return (
                    f"Let op het minteken! "
                    f"Je hebt de {var}-termen bij elkaar opgeteld zonder rekening te "
                    f"houden met het minteken. "
                    f"De correcte som is {correct_coeff}{var}, niet {wrong_coeff}{var}."
                )
        except Exception:
            pass
    return None


def _check_wrong_sign_const(prev_expr, new_expr) -> str | None:
    """
    Detect: student flipped the sign of one constant term when combining constants.

    Example:  5x - 4 + 3  →  5x - 7   (correct: 5x - 1)
              3x - 4 + 2x + 3  →  5x - 7   (correct: 5x - 1)

    Phase 1 — Pattern match:
      - There are at least two constant (non-x) terms in prev.
      - The constants have mixed signs.

    Phase 2 — Diagnosis:
      - The new constant term equals the result obtained by flipping the sign
        of exactly one positive (or one negative) constant in prev.
      - We check all single-flip possibilities and report the first match.
    """
    # Phase 1
    consts = _const_terms(prev_expr)
    if len(consts) <= 1:
        return None

    numeric_consts = [c for c in consts if c.is_number]
    has_neg = any(c < 0 for c in numeric_consts)
    has_pos = any(c > 0 for c in numeric_consts)
    if not (has_neg and has_pos):
        return None  # All same sign — no sign confusion

    correct_const = sum(numeric_consts)

    # Phase 2
    try:
        # Constant term of the new expression (x^0 coefficient)
        new_const = sum(
            (t for t in Add.make_args(expand(new_expr)) if t.is_number),
            S.Zero  # identity element for sum
        )

        # Check every single-flip scenario
        for c in numeric_consts:
            wrong = correct_const - 2 * c   # flip the sign of term c
            if simplify(new_const - wrong) == 0:
                return (
                    f"Let op het teken bij de getallen! "
                    f"De correcte som van de losse getallen is {correct_const}, "
                    f"niet {wrong}."
                )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Exported ErrorChecker instances
# ---------------------------------------------------------------------------

wrong_sign_combining_x = ErrorChecker(
    id="wrong_sign_combining_x",
    description=(
        "Student dropped a minus sign when combining x-terms "
        "(treated subtraction as addition)."
    ),
    check=_check_wrong_sign_x,
)

wrong_sign_combining_const = ErrorChecker(
    id="wrong_sign_combining_const",
    description=(
        "Student flipped the sign of a constant term when combining constants."
    ),
    check=_check_wrong_sign_const,
)
