"""
Template-based checker for error patterns and strategy patterns.

Two matching modes are available, selected by the JSON fields used.

# Option 1 - all scalar parameters

All named parameters are plain scalars (numbers).  Matching works by
expanding both templates and the actual equations to polynomials in x,
then solving the resulting coefficient equations.

    {
        "params":        ["a", "b", "c"],
        "prev_template": "a*x + b = c",
        "new_template":  "a*x = c + b",
        ...
    }

# Option 2 - scalar params + expr params

Use this when a parameter should match subexpressions

   scalar_params  - match any expression that does NOT contain x, similar to params in option 1.
   expr_params    - match any SymPy expression, including ones that contain x, this is used for subsexpression like brackets.
"""

import json
import os
from typing import Optional, Tuple, List, Callable

from sympy import symbols, solve, expand, Eq, Poly, simplify, Wild
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from .base import ErrorChecker
from ..parser import x
from ..validator import StrategyRating

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

_RATING_MAP = {
    'OPTIMAL':          StrategyRating.OPTIMAL,
    'SUBOPTIMAL':       StrategyRating.SUBOPTIMAL,
    'COUNTERPRODUCTIVE': StrategyRating.COUNTERPRODUCTIVE,
}

# Type alias: a strategy check function returns (message, rating) or None
StrategyCheckResult = Tuple[str, StrategyRating]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_template_equation(template_str: str, param_syms: dict) -> Optional[Eq]:
    """Parse 'LHS = RHS' with x and named parameter symbols in scope."""
    s = template_str.replace('==', '=').replace('\u2212', '-').replace('^', '**')
    parts = s.split('=')
    if len(parts) != 2:
        return None
    local_dict = {'x': x, **param_syms}
    try:
        lhs = parse_expr(parts[0].strip(), local_dict=local_dict,
                         transformations=_TRANSFORMS)
        rhs = parse_expr(parts[1].strip(), local_dict=local_dict,
                         transformations=_TRANSFORMS)
        return Eq(lhs, rhs)
    except Exception:
        return None


def _coeff_equations(template_expr, actual_expr) -> Optional[list]:
    """
    Match polynomial coefficients of template_expr against actual_expr.

    Returns a list of equations (template_coeff - actual_coeff = 0) for each
    power of x, or None if the polynomial degrees don't match.
    """
    t = expand(template_expr)
    a = expand(actual_expr)
    try:
        t_poly = Poly(t, x)
        a_poly = Poly(a, x)
    except Exception:
        return None
    if t_poly.degree() != a_poly.degree():
        return None
    deg = t_poly.degree()
    return [t_poly.nth(i) - a_poly.nth(i) for i in range(deg + 1)]


def _try_match(template_prev: Eq, template_new: Eq,
               actual_prev: Eq, actual_new: Eq,
               param_list: list) -> Optional[dict]:
    """
    Try to match both templates against both actual equations in a fixed
    LHS/RHS orientation. Returns a solved param dict or None.
    """
    all_eqs: list = []
    for t_side, a_side in [
        (template_prev.lhs, actual_prev.lhs),
        (template_prev.rhs, actual_prev.rhs),
        (template_new.lhs,  actual_new.lhs),
        (template_new.rhs,  actual_new.rhs),
    ]:
        eqs = _coeff_equations(t_side, a_side)
        if eqs is None:
            return None
        all_eqs.extend(eqs)

    try:
        solutions = solve(all_eqs, param_list, dict=True)
    except Exception:
        return None

    if not solutions:
        return None

    solution = solutions[0]
    if len(solution) != len(param_list):
        return None
    return solution


def _solve_params(template_prev: Eq, template_new: Eq,
                  actual_prev: Eq, actual_new: Eq,
                  param_syms: dict) -> Optional[dict]:
    """
    Solve for all parameters by simultaneously matching both templates.

    Tries all four combinations of (swap / no-swap) on the actual equations
    so that templates match regardless of whether the student or item wrote
    the equation as  LHS = RHS  or  RHS = LHS.

    Returns a substitution dict {Symbol: value}, or None if no unique match.
    """
    param_list = list(param_syms.values())

    prev_flipped = Eq(actual_prev.rhs, actual_prev.lhs)
    new_flipped  = Eq(actual_new.rhs,  actual_new.lhs)

    for ap, an in [
        (actual_prev,  actual_new),
        (prev_flipped, actual_new),
        (actual_prev,  new_flipped),
        (prev_flipped, new_flipped),
    ]:
        result = _try_match(template_prev, template_new, ap, an, param_list)
        if result is not None:
            return result

    return None


def _build_message(message_template: str, solved: dict,
                   computed: dict, param_syms: dict) -> str:
    """
    Fill the message template with solved parameter values and any
    computed derived values defined in the 'computed' JSON field.
    """
    named = {sym.name: val for sym, val in solved.items()}

    for name, expr_str in computed.items():
        expr_str = expr_str.replace('^', '**')
        try:
            expr = parse_expr(expr_str, local_dict=param_syms,
                              transformations=_TRANSFORMS)
            named[name] = simplify(expr.subs(solved))
        except Exception:
            named[name] = '?'

    try:
        return message_template.format(**named)
    except KeyError:
        return message_template


# Wild based pattern matching

def _make_wilds(scalar_names: list, expr_names: list) -> dict:
    """
    Create Wild symbols for Wild-based templates.

    scalar_names → Wild(name, exclude=[x])  - matches scalars only (no x)
    expr_names   → Wild(name)               - matches any sub-expression
    """
    wilds = {}
    for name in scalar_names:
        wilds[name] = Wild(name, exclude=[x])
    for name in expr_names:
        wilds[name] = Wild(name)
    return wilds


def _match_eq_wilds(template_eq: Eq, actual_eq: Eq) -> Optional[dict]:
    """
    Try to match actual_eq against a Wild-based template_eq.

    Returns {Wild: value} or None.  Tries both LHS/RHS orientations so
    the template fires regardless of how the student wrote the equation.
    """
    for a_lhs, a_rhs in [
        (actual_eq.lhs, actual_eq.rhs),
        (actual_eq.rhs, actual_eq.lhs),
    ]:
        lhs_m = a_lhs.match(template_eq.lhs)
        if lhs_m is None:
            continue
        rhs_m = a_rhs.match(template_eq.rhs)
        if rhs_m is None:
            continue

        # Merge both sides, checking that shared Wilds agree
        combined = dict(lhs_m)
        ok = True
        for wild, val in rhs_m.items():
            if wild in combined:
                if expand(combined[wild] - val) != 0:
                    ok = False
                    break
            else:
                combined[wild] = val
        if ok:
            return combined

    return None


def _solve_params_wilds(template_prev: Eq, template_new: Eq,
                        actual_prev: Eq, actual_new: Eq) -> Optional[dict]:
    """
    Match a Wild-based template pair against an actual step transition.

    Strategy
    --------
    1. Match the *prev* template against actual_prev using Wild.match() to
       determine all parameter values.  This is reliable because all params
       are constrained by the structure of the prev equation.
    2. Substitute those values into the *new* template to produce the exact
       equation the error predicts.
    3. Check whether that predicted equation is algebraically equal to
       actual_new (trying both LHS/RHS orientations).

    This avoids the ambiguity that arises when SymPy tries to independently
    match a new template whose RHS is a derived expression like  b - a
    (which has infinitely many Wild solutions on its own).

    Returns {Symbol: value} or None.
    """
    prev_m = _match_eq_wilds(template_prev, actual_prev)
    if prev_m is None:
        return None

    # Build the predicted new equation by substituting solved param values
    predicted_lhs = expand(template_new.lhs.subs(prev_m))
    predicted_rhs = expand(template_new.rhs.subs(prev_m))

    actual_lhs = expand(actual_new.lhs)
    actual_rhs = expand(actual_new.rhs)

    lhs_ok = expand(predicted_lhs - actual_lhs) == 0
    rhs_ok = expand(predicted_rhs - actual_rhs) == 0
    if lhs_ok and rhs_ok:
        return {symbols(w.name): val for w, val in prev_m.items()}

    # Also try the flipped orientation of the actual new equation
    lhs_ok = expand(predicted_lhs - actual_rhs) == 0
    rhs_ok = expand(predicted_rhs - actual_lhs) == 0
    if lhs_ok and rhs_ok:
        return {symbols(w.name): val for w, val in prev_m.items()}

    return None


# ---------------------------------------------------------------------------
# Template class
# ---------------------------------------------------------------------------

class Template:
    """
    A defined error or strategy pattern.

    Parameters
    ----------
    template_type : str
        "error" - fires during error diagnosis (incorrect steps).
        "strategy" - fires during strategy assessment (correct but suboptimal steps).
    id, description, message : str
        Stable identifier, human description, and feedback text.
    prev_template, new_template : str
        Equation templates using x and named parameters, e.g. "a*(x+b) = c".
    params : list[str]
        Names of scalar parameters used in the templates. Use this if no scalar/expr params are used
    scalar_params : list[str]
        Names of scalar paramters used in the templates
    expr_params : list[str]
        Names of expr paramaters used in the templates 
    computed : dict, optional
        Extra expressions to evaluate and inject into the message,
        e.g. {"ab": "a*b"} makes {ab} available in the message.
    rating : str, optional
        Strategy rating for type="strategy": "COUNTERPRODUCTIVE" | "SUBOPTIMAL" | "OPTIMAL".
    """

    def __init__(self, template_type: str, id: str, description: str,
                 message: str, prev_template: str, new_template: str,
                 params: list = None,
                 scalar_params: list = None, expr_params: list = None,
                 computed: dict = None, rating: str = None):
        self.template_type = template_type
        self.id            = id
        self.description   = description
        self.message       = message
        self._prev_str     = prev_template
        self._new_str      = new_template
        self._computed     = computed or {}
        self.rating        = _RATING_MAP.get(rating or '', StrategyRating.COUNTERPRODUCTIVE)

        # Determine matching mode
        if scalar_params is not None or expr_params is not None:
            # Mode 2: Wild-based matching
            self._use_wilds  = True
            self._wilds      = _make_wilds(scalar_params or [], expr_params or [])
            self._param_syms = self._wilds
        else:
            # Mode 1: polynomial coefficient matching (original behaviour)
            self._use_wilds  = False
            self._param_syms = {name: symbols(name) for name in (params or [])}

        # _build_message always needs plain Symbol objects (not Wilds) so that
        # computed expressions like "b/a" can be parsed and substituted correctly.
        all_names = list((params or []) + (scalar_params or []) + (expr_params or []))
        self._msg_param_syms = {name: symbols(name) for name in all_names}

    def _match(self, prev_eq: Eq, new_eq: Eq) -> Optional[dict]:
        """Try to match both equations against templates. Returns solved params or None."""
        try:
            if self._use_wilds:
                t_prev = _parse_template_equation(self._prev_str, self._wilds)
                t_new  = _parse_template_equation(self._new_str,  self._wilds)
                if t_prev is None or t_new is None:
                    return None
                return _solve_params_wilds(t_prev, t_new, prev_eq, new_eq)
            else:
                t_prev = _parse_template_equation(self._prev_str, self._param_syms)
                t_new  = _parse_template_equation(self._new_str,  self._param_syms)
                if t_prev is None or t_new is None:
                    return None
                return _solve_params(t_prev, t_new, prev_eq, new_eq, self._param_syms)
        except Exception:
            return None

    def as_error_checker(self) -> ErrorChecker:
        """Wrap as an ErrorChecker: (prev_eq, new_eq) -> str | None."""
        def _check(prev_eq: Eq, new_eq: Eq) -> Optional[str]:
            solved = self._match(prev_eq, new_eq)
            if solved is None:
                return None
            return _build_message(self.message, solved, self._computed, self._msg_param_syms)

        return ErrorChecker(id=self.id, description=self.description, check=_check)

    def as_strategy_check(self) -> Callable:
        """Return a strategy check function: (prev_eq, new_eq) -> (msg, rating) | None."""
        def _check(prev_eq: Eq, new_eq: Eq) -> Optional[StrategyCheckResult]:
            solved = self._match(prev_eq, new_eq)
            if solved is None:
                return None
            msg = _build_message(self.message, solved, self._computed, self._msg_param_syms)
            return msg, self.rating

        return _check


# Json Loader

def _load_template(path: str) -> Optional[Template]:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return Template(
            template_type=data.get('type', 'error'),
            id=data['id'],
            description=data['description'],
            message=data['message'],
            prev_template=data['prev_template'],
            new_template=data['new_template'],
            params=data.get('params'),
            scalar_params=data.get('scalar_params'),
            expr_params=data.get('expr_params'),
            computed=data.get('computed', {}),
            rating=data.get('rating'),
        )
    except Exception as e:
        print(f"Warning: could not load template {os.path.basename(path)}: {e}")
        return None


def load_templates(
    templates_dir: str,
) -> Tuple[List[ErrorChecker], List[Callable]]:
    """
    Load all *.json templates from templates_dir.

    Returns
    -------
    error_checkers : list[ErrorChecker]
        Ready to use in Goal.default_error_checks.
    strategy_checks : list[callable]
        Each callable is (prev_eq, new_eq) -> (message, StrategyRating) | None.
    """
    error_checkers:  List[ErrorChecker] = []
    strategy_checks: List[Callable]     = []

    if not os.path.isdir(templates_dir):
        return error_checkers, strategy_checks

    all_paths = []
    for dirpath, dirnames, filenames in os.walk(templates_dir):
        dirnames.sort()  # traverse subdirectories in alphabetical order
        for filename in sorted(filenames):
            if filename.endswith('.json'):
                all_paths.append(os.path.join(dirpath, filename))

    for path in all_paths:
        template = _load_template(path)
        if template is None:
            continue

        if template.template_type == 'error':
            error_checkers.append(template.as_error_checker())
        elif template.template_type == 'strategy':
            strategy_checks.append(template.as_strategy_check())

    return error_checkers, strategy_checks


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
