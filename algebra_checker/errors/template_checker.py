"""
Template-based checker for error patterns and strategy patterns.

Teachers define patterns as pairs of equation templates using named scalar
parameters (a, b, c, …) and the variable x. The system matches actual student
steps by simultaneously solving polynomial-coefficient equations derived from
both templates.

Template JSON format
--------------------
{
    "id":            "distribution_error",
    "type":          "error",             # "error" | "strategy"
    "rating":        "COUNTERPRODUCTIVE", # strategy only: OPTIMAL|SUBOPTIMAL|COUNTERPRODUCTIVE
    "params":        ["a", "b", "c"],
    "prev_template": "a*(x+b) = c",
    "new_template":  "a*x + b = c",
    "computed":      {"ab": "a*b"},       # optional extra values for the message
    "description":   "Student didn't distribute the outer factor",
    "message":       "When expanding {a}*(x+{b}) multiply every term by {a}. Result: {a}*x + {ab} = {c}."
}

Matching algorithm
------------------
1. Parse prev_template and new_template with the named parameters as SymPy symbols.
2. Expand both templates and the actual equations to polynomials in x.
3. Equate coefficients of each power of x (from both LHS and RHS of both equations)
   to get a system of equations in the parameters.
4. Solve the system with SymPy. A unique solution means the templates match.
5. Substitute the solved values into the message (and any "computed" expressions).

Limitation: parameters must be scalars (numbers). Sub-expression wildcards
(e.g. A + B = C) require SymPy Wild matching and are a future extension.
"""

import json
import os
from typing import Optional, Tuple, List, Callable

from sympy import symbols, solve, expand, Eq, Poly, simplify, Symbol
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


def _coeff_equations(template_expr, actual_expr, param_list: list) -> Optional[list]:
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
        eqs = _coeff_equations(t_side, a_side, param_list)
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

    return solution


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


# ---------------------------------------------------------------------------
# Template class
# ---------------------------------------------------------------------------

class Template:
    """
    A single teacher-defined error or strategy pattern.

    Parameters
    ----------
    template_type : str
        "error" — fires during error diagnosis (incorrect steps).
        "strategy" — fires during strategy assessment (correct but suboptimal steps).
    id, description, message : str
        Stable identifier, human description, and feedback text.
    prev_template, new_template : str
        Equation templates using x and named parameters, e.g. "a*(x+b) = c".
    params : list[str]
        Names of scalar parameters used in the templates, e.g. ["a", "b", "c"].
    computed : dict, optional
        Extra expressions to evaluate and inject into the message,
        e.g. {"ab": "a*b"} makes {ab} available in the message.
    rating : str, optional
        Strategy rating for type="strategy": "COUNTERPRODUCTIVE" | "SUBOPTIMAL" | "OPTIMAL".
    """

    def __init__(self, template_type: str, id: str, description: str,
                 message: str, prev_template: str, new_template: str,
                 params: list, computed: dict = None, rating: str = None):
        self.template_type = template_type
        self.id            = id
        self.description   = description
        self.message       = message
        self._prev_str     = prev_template
        self._new_str      = new_template
        self._param_syms   = {name: symbols(name) for name in params}
        self._computed     = computed or {}
        self.rating        = _RATING_MAP.get(rating or '', StrategyRating.COUNTERPRODUCTIVE)

    def _match(self, prev_eq: Eq, new_eq: Eq) -> Optional[dict]:
        """Try to match both equations against templates. Returns solved params or None."""
        try:
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
            return _build_message(self.message, solved, self._computed, self._param_syms)

        return ErrorChecker(id=self.id, description=self.description, check=_check)

    def as_strategy_check(self) -> Callable:
        """Return a strategy check function: (prev_eq, new_eq) -> (msg, rating) | None."""
        def _check(prev_eq: Eq, new_eq: Eq) -> Optional[StrategyCheckResult]:
            solved = self._match(prev_eq, new_eq)
            if solved is None:
                return None
            msg = _build_message(self.message, solved, self._computed, self._param_syms)
            return msg, self.rating

        return _check


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

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
            params=data['params'],
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

    for filename in sorted(os.listdir(templates_dir)):
        if not filename.endswith('.json'):
            continue
        template = _load_template(os.path.join(templates_dir, filename))
        if template is None:
            continue

        if template.template_type == 'error':
            error_checkers.append(template.as_error_checker())
        elif template.template_type == 'strategy':
            strategy_checks.append(template.as_strategy_check())

    return error_checkers, strategy_checks


TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')
