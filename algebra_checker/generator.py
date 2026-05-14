"""
Generates concrete question instances from parameterized item templates.

Templates use {{param}} syntax in 'display', 'sympy_str', and 'expected_answer'.
Adding a 'params' key to an item opts it in to instantiation items without 'params' are unchanged.

Parameter formats
----------------------
{"int": [lo, hi]}                      random integer in [lo, hi] inclusive
{"int": [lo, hi], "step": k}           random multiple of k in [lo, hi]
{"int": [lo, hi], "exclude": [v, ...]} integer in [lo, hi], never the listed values
{"fraction": {"lcd": n}}               irreducible proper fraction p/q where q divides n
                                       and gcd(p, q) = 1 — guarantees the denominator
                                       never changes after SymPy simplification.
                                       Optional: "negative": true  →  -p/q
                                                 "exclude_noemer": [param, ...]  →  skip used denominators
{"choices": [v, ...]}                  pick uniformly from the explicit list;
                                       supports "exclude": [param, ...] to skip values
"sympy_expression_string"              derived value evaluated with already-resolved
                                       params via SymPy — must appear after its deps

Parameters are resolved from top to bottom (order of decleration)

Template  formats
{{param}}        raw value, sign included          →  -5
{{param:abs}}    absolute value                    →   5
{{param:term}}   sign + space + abs value          → + 5  or  - 5   (inline LaTeX term)
{{param:frac}}   LaTeX fraction if rational        \\frac{4}{3}   else plain integer

The 'form' field (optional, recommended)
    "form": "a(x + b) = c"
    "form": "ax + b = c"
    "form": "(x + p)(x + q) = 0"

Example: linear equation with guaranteed integer solution
---------------------------------------------------------
{
  "id": "lin_01",
  "form": "ax + b = c",
  "display": "{{a}}x {{b:term}} = {{c}}",
  "sympy_str": "{{a}}*x + {{b}} = {{c}}",
  "goal": "solve_equation",
  "params": {
    "a":     {"int": [2, 9]},
    "x_sol": {"int": [1, 8]},
    "b":     {"int": [1, 9]},
    "c":     "a * x_sol + b"
  }
}

Example: bracket form (crisis item) with guaranteed integer solution
---------------------------------------------------------------------
{
  "id": "lin_01_bracket",
  "form": "a(x + b) = c",
  "display": "{{a}}(x + {{b}}) = {{c}}",
  "sympy_str": "{{a}}*(x + {{b}}) = {{c}}",
  "goal": "solve_equation",
  "params": {
    "a":     {"int": [2, 9]},
    "x_sol": {"int": [1, 8]},
    "b":     {"int": [1, 9]},
    "c":     "a * (x_sol + b)"
  }
}

Example: quadratic factored form with integer roots
----------------------------------------------------
{
  "id": "quad_01",
  "form": "(x + p)(x + q) = 0",
  "display": "(x {{p:term}})(x {{q:term}}) = 0",
  "sympy_str": "(x + {{p}}) * (x + {{q}}) = 0",
  "goal": "solve_quadratic_equation",
  "params": {
    "p": {"int": [-7, 7], "exclude": [0]},
    "q": {"int": [-7, 7], "exclude": [0]}
  }
}

Example: direct answer — slope may be a fraction
-------------------------------------------------
{
  "id": "slope_01",
  "form": "slope through two points",
  "display": "\\text{Find the slope through } ({{x1}}, {{y1}}) \\text{ and } ({{x2}}, {{y2}})",
  "sympy_str": "slope",
  "goal": "direct_answer",
  "expected_answer": "({{y2}} - {{y1}}) / ({{x2}} - {{x1}})",
  "params": {
    "x1": {"int": [0, 4]},
    "y1": {"int": [0, 4]},
    "dx": {"int": [1, 5]},
    "dy": {"int": [1, 8]},
    "x2": "x1 + dx",
    "y2": "y1 + dy"
  }
}
"""

import hashlib
import random
import re
from math import gcd
from typing import Any

from sympy import Integer, Rational, latex, sympify


def _item_seed(student_id: str, item_id: str) -> int:
    payload = f"{student_id}:{item_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little")


def _resolve_params(spec: dict, rng: random.Random) -> dict[str, Any]:
    """
    Resolve a params spec dict to concrete values, in declaration order.

    Integer params are stored as Python int.
    Derived params are stored as SymPy numbers (may be Rational for fractions).
    """
    values: dict[str, Any] = {}

    for name, s in spec.items():
        if isinstance(s, str):
            subs = {k: Integer(v) if isinstance(v, int) else v
                    for k, v in values.items()}
            print(s, subs)
            result = sympify(s).subs(subs)
            print(result, name)
            values[name] = int(result) if result == int(result) else result

        elif isinstance(s, dict) and "fraction" in s:
            spec    = s["fraction"]
            lcd     = spec["lcd"]
            neg     = spec.get("negative", False)
            exclude_raw = spec.get("exclude_noemer", [])
            exclude_q   = {int(values[e].q) if isinstance(e, str) and hasattr(values[e], 'q') else (int(values[e]) if isinstance(e, str) else e) for e in exclude_raw}

            # Denominators: divisors of lcd that are >= 2 and not excluded
            divisors = [d for d in range(2, lcd + 1) if lcd % d == 0 and d not in exclude_q]
            if not divisors:
                raise ValueError(f"No valid denominator for fraction param '{name}' (lcd={lcd})")
            q = rng.choice(divisors)

            # Numerators: p in [1, q-1] coprime with q → fraction is irreducible by construction
            valid_p = [p for p in range(1, q) if gcd(p, q) == 1]
            p = rng.choice(valid_p)
            values[name] = Rational(-p, q) if neg else Rational(p, q)

        elif isinstance(s, dict) and "choices" in s:
            exclude_raw = s.get("exclude", [])
            exclude     = {values[e] if isinstance(e, str) else e for e in exclude_raw}
            choices     = [values[c] if isinstance(c, str) else c
                           for c in s["choices"] if c not in exclude]
            if not choices:
                raise ValueError(f"No valid choices for param '{name}'")
            values[name] = rng.choice(choices)

        elif isinstance(s, dict) and "int" in s:
            lo, hi = s["int"]
            # Check if bounds are sympy strings:
            lo = values[lo] if isinstance(lo, str) else lo 
            hi = values[hi] if isinstance(hi, str) else hi 

            step = s.get("step", 1)
            # Check if step is sympy string:
            step = values[step] if isinstance(step, str) else step
            
            exclude_raw = set(s.get("exclude", []))

            exclude = {values[ex] if isinstance(ex, str) else ex for ex in exclude_raw}

            choices = [v for v in range(lo, hi + 1, step) if v not in exclude]
            if not choices:
                raise ValueError(
                    f"No valid choices for param '{name}' after applying exclude={exclude}"
                )
            values[name] = rng.choice(choices)

        else:
            raise ValueError(f"Unknown param spec for '{name}': {s!r}")

    return values

_PLACEHOLDER = re.compile(r"\{\{(\w+)(?::(\w+))?\}\}")


def _render_value(val: Any, values: dict[str, Any]) -> Any:
    """Recursively render {{param}} templates in a nested JSON-like value."""
    if isinstance(val, str):
        return _render(val, values)
    if isinstance(val, list):
        return [_render_value(v, values) for v in val]
    if isinstance(val, dict):
        return {k: _render_value(v, values) for k, v in val.items()}
    return val


_TERM_AFTER_SIGN = re.compile(r"[+\-]\s*\{\{\w+:term\}\}")


def _render(template: str, values: dict[str, Any], sympy_mode: bool = False) -> str:
    """
    Replace {{param}} and {{param:modifier}} placeholders.

    Modifiers
    ---------
    (none)   raw value, sign included (e.g. -5, or 4/3 for a Rational)
    abs      absolute value (e.g. 5)
    term     sign + space + absolute value (e.g. + 5 or - 5)
             use this to append a term inline: {{a}}x {{b:term}} = {{c}}
             NEVER precede :term with a literal + or -; it includes its own sign
    frac     LaTeX \frac{}{} if the value is a non-integer Rational,
             otherwise the plain integer string

    Leading constants: use raw {{b}} — renders as -2 or 2, never +2

    sympy_mode: when True, negative raw values are wrapped in parentheses so
                that e.g. -{{a}}**3 with a=-1 becomes -(-1)**3 instead of --1**3.
                Use for sympy_str, expected_answer, and if_answer fields.
    """
    if _TERM_AFTER_SIGN.search(template):
        raise ValueError(
            f"Template has a literal sign immediately before a :term placeholder — "
            f":term already includes the sign.\n"
            f"  template: {template!r}\n"
            f"  Fix: '+ {{{{b:term}}}}' → '{{{{b:term}}}}',  '- {{{{b:term}}}}' → '{{{{b:term}}}}'"
        )

    def replace(m: re.Match) -> str:
        name = m.group(1)
        mod = m.group(2)
        val = values[name]

        is_negative = (val < 0)
        abs_val = abs(val)

        if mod == "abs":
            return str(abs_val)
        if mod == "term":
            return f"+ {abs_val}" if not is_negative else f"- {abs_val}"
        if mod == "frac":
            if isinstance(val, Rational) and val.q != 1:
                return latex(val)
            return str(int(val) if val == int(val) else val)
        if mod == "frac_term":
            if isinstance(val, Rational) and val.q != 1:
                return f"+ {latex(abs_val)}" if not is_negative else f"- {latex(abs_val)}"
            val = str(int(abs_val) if abs_val == int(abs_val) else abs_val)
            return f"+ {val}" if not is_negative else f"- {val}"
        # Default: raw value — wrap negatives in parens when rendering for SymPy
        # to avoid e.g. -{{a}}**3 with a=-1 producing --1**3 instead of -(-1)**3.
        if sympy_mode and is_negative:
            return f"({val})"
        return str(val)

    return _PLACEHOLDER.sub(replace, template)



def instantiate_item(item: dict, student_id: str) -> dict:
    """
    Return a concrete item dict with all {{param}} placeholders resolved.

    Items without a 'params' key are returned unchanged.

    The returned dict also contains '_param_values' (the resolved param dict)
    which can be logged for debugging or analytics.

    Crisis sub-items (crisis_tuple.pre_crisis) are recursively instantiated
    if they carry their own 'params' key.
    """
    if "params" not in item:
        if isinstance(item.get("context"), list):
            result = dict(item)
            result["context"] = ' '.join(item["context"])
            return result
        return item

    seed = _item_seed(student_id, item["id"])
    rng = random.Random(seed)
    values = _resolve_params(item["params"], rng)

    result = dict(item)

    if "context" in item:
        ctx = item["context"]
        result["context"] = ' '.join(_render(c, values) for c in ctx) if isinstance(ctx, list) else _render(ctx, values)

    if "display" in item:
        display = item["display"]
        result["display"] =  ' '.join(_render(c, values) for c in display) if isinstance(display, list) else _render(display, values)
        
    if "sympy_str" in item:
        result["sympy_str"] = _render(item["sympy_str"], values, sympy_mode=True)

    if "expected_answer" in item:
        result["expected_answer"] = _render(item["expected_answer"], values, sympy_mode=True)

    if "wrong_answer_hints" in item:
        result["wrong_answer_hints"] = [
            {**hint, "if_answer": _render(hint["if_answer"], values, sympy_mode=True)} if "if_answer" in hint else hint
            for hint in item["wrong_answer_hints"]
        ]

    if "graph" in item:
        result["graph"] = _render_value(item["graph"], values)

    if "parts" in item:
        rendered_parts = []
        for part in item["parts"]:
            rendered_part = dict(part)
            rendered_part["display"] = _render(part["display"], values)
            rendered_part["sympy_str"] = _render(part["sympy_str"], values, sympy_mode=True)
            if "expected_answer" in part:
                rendered_part["expected_answer"] = _render(part["expected_answer"], values, sympy_mode=True)
            if "wrong_answer_hints" in part:
                rendered_part["wrong_answer_hints"] = [
                    {**hint, "if_answer": _render(hint["if_answer"], values, sympy_mode=True)} if "if_answer" in hint else hint
                    for hint in part["wrong_answer_hints"]
                ]
            rendered_parts.append(rendered_part)
        result["parts"] = rendered_parts

    result["_param_values"] = values

    # Recurse into crisis sub-items
    if "crisis_tuple" in item:
        tup = dict(item["crisis_tuple"])
        if "pre_crisis" in tup:
            tup["pre_crisis"] = instantiate_item(tup["pre_crisis"], student_id)
        result["crisis_tuple"] = tup

    return result
