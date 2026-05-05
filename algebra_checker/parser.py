"""
Parser for student-entered mathematical expressions and equations.
"""
 
import re
from typing import Union, Tuple, List

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
        super().__init__(f"Er is iets mis met je invoer '{raw_input}': {reason}")
 
def _preprocess(text: str) -> str:
    """Clean up student input into something SymPy can handle."""
    s = text.strip()
 
    # Replace common unicode / typography
    s = s.replace('×', '*').replace('·', '*').replace('−', '-')
    s = s.replace('^', '**')

    # Decimal comma: 1,5 → 1.5  (only between digits, leaves f(a,b) intact)
    s = re.sub(r'(\d),(\d)', r'\1.\2', s)
 
    # Replace ² and ³ with **2 and **3
    s = s.replace('²', '**2').replace('³', '**3')
 
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
    if text == '':
        raise ParseError(text, f"aan beide kanten van het = teken moet iets staan.")
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
        raise ParseError(text, f"in een vergelijking moet een = teken staan.")
    
    lhs = parse_expr_safe(parts[0])
    rhs = parse_expr_safe(parts[1])
    return Eq(lhs, rhs, evaluate=False)
 
 
def parse_student_input(text: str) -> Union[Eq, Expr]:
    """
    Auto-detect whether the student entered an equation or expression.
 
    Returns Eq if '=' present, otherwise Expr.
    """
    text_clean = text.replace('==', '=')
    if '=' in text_clean:
        return parse_equation(text)
    return parse_expr_safe(text)
 
 
def equation_sides(eq: Eq) -> Tuple[Expr, Expr]:
    """Return (lhs, rhs) of an equation."""
    return eq.lhs, eq.rhs


_DISJUNCTION_RE = re.compile(r'∨|\s+(?:v|of)\s+')


def is_disjunction(text: str) -> bool:
    """Return True if text contains a logical-or separator between equations."""
    return bool(_DISJUNCTION_RE.search(text.replace('==', '=')))


def parse_disjunction(text: str) -> List[Eq]:
    """
    Parse a disjunction like 'x+3=0 v x-2=0' into a list of Eq objects.

    Raises ParseError if any branch cannot be parsed as an equation.
    """
    parts = _DISJUNCTION_RE.split(text)
    if len(parts) < 2:
        raise ParseError(text, "er mist een of-teken in je antwoord.")
    return [parse_equation(part.strip()) for part in parts]