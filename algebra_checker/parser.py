"""
Parser for student-entered mathematical expressions and equations.
 
Handles common student notations:
  - Implicit multiplication: 2x, 3(x+1)
  - Fractions: 2/3x, (x+1)/(x-2)
  - Equations with = sign
  - Mixed numbers and decimals
"""
 
import re
from typing import Union, Tuple
 
import sympy
from sympy import (
    Symbol, Eq, sympify, Rational,
    Expr, S, symbols, sqrt, Abs
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
 
x = Symbol('x')
 
 
class ParseError(Exception):
    """Raised when student input cannot be parsed."""
    def __init__(self, raw_input: str, reason: str = ""):
        self.raw_input = raw_input
        self.reason = reason
        super().__init__(f"Cannot parse '{raw_input}': {reason}")
 
 
# ---------------------------------------------------------------------------
# Preprocessing: normalise student notation 
# ---------------------------------------------------------------------------
 
def _preprocess(text: str) -> str:
    """Clean up student input into something SymPy can handle."""
    s = text.strip()
 
    # Replace common unicode / typography
    s = s.replace('×', '*').replace('·', '*').replace('−', '-')
    s = s.replace('^', '**')
 
    # Replace ² and ³ with **2 and **3
    s = s.replace('²', '**2').replace('³', '**3')
 
    # Handle cases like "2x" -> "2*x", "3(x" -> "3*(x"
    # SymPy's implicit_multiplication_application handles most of this,
    # but we help with a few edge cases.
 
    # ")(": insert multiplication between adjacent parens
    s = re.sub(r'\)\s*\(', ')*(', s)
 
    # A number directly before a letter: 2x -> 2*x  (handled by SymPy transforms)
    # A number directly before (: 3( -> 3*(
    s = re.sub(r'(\d)\s*\(', r'\1*(', s)
 
    # A closing paren before a letter: )x -> )*x
    s = re.sub(r'\)\s*([a-zA-Z])', r')*\1', s)
 
    return s
 
 
# Transformations for the SymPy parser
_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
 
 
def parse_expr_safe(text: str) -> Expr:
    """
    Parse a single mathematical expression from student text.
 
    Returns a SymPy Expr.
    Raises ParseError on failure.
    """
    preprocessed = _preprocess(text)
    try:
        expr = parse_expr(preprocessed, transformations=_TRANSFORMS, local_dict={'x': x}, evaluate=False)
        return expr
    except Exception as e:
        raise ParseError(text, str(e))
 
 
def parse_equation(text: str) -> Eq:
    """
    Parse an equation of the form 'LHS = RHS'.
 
    Returns a SymPy Eq object.
    Raises ParseError if there is not exactly one '=' sign.
    """
    # Handle == as well
    text = text.replace('==', '=')
 
    parts = text.split('=')
    if len(parts) != 2:
        raise ParseError(text, f"Expected exactly one '=' sign, found {len(parts)-1}")
 
    lhs = parse_expr_safe(parts[0])
    rhs = parse_expr_safe(parts[1])
    return Eq(lhs, rhs)
 
 
def parse_student_input(text: str) -> Union[Eq, Expr]:
    """
    Auto-detect whether the student entered an equation or expression.
 
    Returns Eq if '=' present, otherwise Expr.
    """
    text_clean = text.replace('==', '=')
    if '=' in text_clean:
        return parse_equation(text)
    return parse_expr_safe(text)
 
 
# ---------------------------------------------------------------------------
# Convenience: extract LHS/RHS from equations
# ---------------------------------------------------------------------------
 
def equation_sides(eq: Eq) -> Tuple[Expr, Expr]:
    """Return (lhs, rhs) of an equation."""
    return eq.lhs, eq.rhs