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

from sympy import Add, Mul, Pow, expand, simplify, latex as _sym_latex
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
                    f"Het moet {correct_coeff}{var} zijn, niet {wrong_coeff}{var}."
                )
        except Exception:
            pass
    return None

def _check_add_unlike_terms(prev_expr, new_expr) -> str | None:
    # Phase 1: step is correct → nothing to report
    if simplify(prev_expr - new_expr) == 0:
        return None

    def _eval_term(term):
        # Re-evaluate a single product to flatten nested Mul structures that
        # evaluate=False parsing can create (e.g. Mul(-1, Mul(5, p**2))).
        if isinstance(term, Mul):
            return Mul(*term.args)
        return term

    def split(term):
        return _eval_term(term).as_coeff_Mul()

    def monom_coeffs(expr):
        """Sum of numeric coefficients per monomial key.
        Works for any polynomial: x, p**2, p*q, constants, etc."""
        m = {}
        for term in Add.make_args(expr):
            coeff, monom = split(term)
            m[monom] = m.get(monom, S.Zero) + coeff
        return m

    def individual_coeffs(expr, monom_key):
        """Individual (unevaluated) numeric coefficients for one monomial key."""
        return [split(term)[0]
                for term in Add.make_args(expr)
                if split(term)[1] == monom_key]

    prev_map = monom_coeffs(prev_expr)
    new_map  = monom_coeffs(new_expr)

    def _has_bracket(monom):
        """True when monom is or contains an unevaluated sum — i.e. unexpanded brackets."""
        if isinstance(monom, Add):
            return True
        if hasattr(monom, 'args'):
            return any(isinstance(a, Add) for a in monom.args)
        return False

    # This checker should not fire on expressions with unexpanded brackets.
    if any(_has_bracket(k) for k in prev_map) or any(_has_bracket(k) for k in new_map):
        return None

    all_keys = set(prev_map) | set(new_map)
    delta = {k: new_map.get(k, S.Zero) - prev_map.get(k, S.Zero) for k in all_keys}

    if not all(v.is_number for v in delta.values()):
        return None

    gained = {k: v for k, v in delta.items() if v > 0}
    lost   = {k: v for k, v in delta.items() if v < 0}

    if not gained or not lost:
        return None

    # Phase 2: find unlike-term transfers and build feedback
    def lx(monom, coeff):
        val = coeff if monom == S.One else coeff * monom
        return r'\(' + _sym_latex(val) + r'\)'

    messages = []
    for sink, gain in gained.items():
        # Skip if a literal term already at the new total existed in prev
        if sink != S.One:
            if any(simplify(c - new_map.get(sink, S.Zero)) == 0
                   for c in individual_coeffs(prev_expr, sink)):
                continue
        for source, loss in lost.items():
            if sink == source:
                continue
            if simplify(gain + loss) != 0:
                continue

            if new_map.get(sink, S.Zero) == S.Zero:
                true_sink, true_source, true_gain = source, sink, loss
            else:
                true_sink, true_source, true_gain = sink, source, gain

            ts_ind_prev = individual_coeffs(prev_expr, true_sink)
            ts_ind_new  = individual_coeffs(new_expr,  true_sink)
            ts_prev     = prev_map.get(true_sink, S.Zero)
            ts_new      = new_map.get(true_sink, S.Zero)

            affected = next(
                (c for c in ts_ind_prev
                 if any(simplify(c + true_gain - nj) == 0 for nj in ts_ind_new)),
                ts_prev,
            )
            wrong_coeff = simplify(affected + true_gain) if affected != ts_prev else ts_new

            sink_val   = affected if true_sink   == S.One else affected   * true_sink
            source_val = true_gain if true_source == S.One else true_gain * true_source

            source_term = lx(true_source, true_gain)
            sink_orig   = lx(true_sink,   affected)
            sink_wrong  = lx(true_sink,   wrong_coeff)
            addition    = r'\(' + _sym_latex(sink_val + source_val) + r'\)'
            messages.append(
                f"Je hebt {sink_orig} en {source_term} bij elkaar opgeteld, "
                f"maar dat zijn ongelijksoortige termen. "
                f"{addition} kan niet worden geschreven als {sink_wrong}."
            )

    return " ".join(messages) if messages else None

def _check_wrong_variable(prev_expr, new_expr) -> str | None:
    prev_vars = prev_expr.free_symbols
    new_vars  = new_expr.free_symbols
    introduced = new_vars - prev_vars
    dropped    = prev_vars - new_vars
    if len(introduced) == 1 and len(dropped) == 1:
        new_var = next(iter(introduced))
        old_var = next(iter(dropped))
        return (
            f"Je hebt variabele {new_var} gebruikt, "
            f"maar de uitdrukking heeft {old_var} als variabele."
        )
    return None


def _check_wrong_sign_const(prev_expr, new_expr) -> str | None:
    """
    Detect: student flipped the sign of one constant term when combining constants.

    Example:  5x - 4 + 3,  5x - 7   (correct: 5x - 1)
              3x - 4 + 2x + 3 , 5x - 7   (correct: 5x - 1)

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

    correct_const = simplify(sum(numeric_consts))

    # Phase 2
    try:
        # Constant term of the new expression (x^0 coefficient)
        new_const = sum(
            (t for t in Add.make_args(expand(new_expr)) if t.is_number),
            S.Zero  # identity element for sum
        )

        # Check flip scenarios
        for c in numeric_consts:
            wrong = simplify(correct_const - 2 * c)   # flip the sign of term c
            if simplify(new_const - wrong) == 0:
                if not prev_expr.free_symbols:
                    return (
                        "Let op! Er klopt een minteken niet."
                    )
                return (
                    f"Let op het minteken bij de getallen! "
                    f"het moet {correct_const} zijn, "
                    f"niet {wrong}."
                )
    except Exception:
        pass
    return None


def _check_sign_error_in_product_of_negatives(prev_expr, new_expr) -> str | None:
    """
    Detects if student multiplies factors that include an even number of negatives
    but writes a negative result.
    """
    # Phase 1a: step must be incorrect
    if simplify(prev_expr - new_expr) == 0:
        return None

    # Prev must be a single product 
    if len(Add.make_args(prev_expr)) > 1:
        return None

    # the correct simplified result must also be a single term
    correct = expand(prev_expr)
    if len(Add.make_args(correct)) != 1:
        return None

    # the correct result must have a positive numeric coefficient
    correct_coeff = correct.as_coeff_Mul()[0]
    if not (correct_coeff.is_number and correct_coeff > 0):
        return None

    # new must be exactly the negation of the correct result
    if simplify(correct + new_expr) != 0:
        return None

    # prev must visibly contain at least 2 negative numeric factors
    neg_factors = [f for f in Mul.make_args(prev_expr) if f.is_number and f < 0]
    if len(neg_factors) < 2:
        return None

    # build feedback
    if len(neg_factors) == 2:
        a, b = neg_factors
        return (
            f"Let op de tekens! "
            f"Min keer min is plus! "
            f"{a} × {b} = {a * b}. "
            f"Het resultaat is positief, niet negatief."
        )
    return (
        f"Let op de mintekens! "
        f"Het resultaat is positief, niet negatief."
    )

# ---------------------------------------------------------------------------
# Exported ErrorChecker instances
# ---------------------------------------------------------------------------

wrong_variable = ErrorChecker(
    id="wrong_variable",
    description=(
        "Leerling introduceerd niet bestaande variabele"
    ),
    check=_check_wrong_variable,
)
wrong_sign_combining_x = ErrorChecker(
    id="wrong_sign_combining_x",
    description=(
        "Leerling wisselt - en + om bij een variabele"
    ),
    check=_check_wrong_sign_x,
)

wrong_sign_combining_const = ErrorChecker(
    id="wrong_sign_combining_const",
    description=(
        "Leerling wisselt - en + om bij een constante."
    ),
    check=_check_wrong_sign_const,
)

wrong_unlike_term_addition = ErrorChecker(
    id="wrong_unlike_term_addition",
    description=(
        "Leerling heeft verschillende variabelen, of variabelen en constanten opgeteld."
    ),
    check=_check_add_unlike_terms,
)

sign_error_in_product_of_negatives = ErrorChecker(
    id="sign_error_in_product_of_negatives",
    description=(
        "Leerling vergeet dat min keer min is plus"
    ),
    check=_check_sign_error_in_product_of_negatives,
)

