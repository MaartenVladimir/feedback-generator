from sympy import Eq, expand, factor, Rationals, symbols

from algebra_checker.parser import parse_equation, parse_expr, parse_expr_safe, parse_student_input, ParseError, x
from algebra_checker.validator import validate_step, StepStatus

def demo_parsing():
    tests = [
        "3x + 5",
        "2(x+3)",
        "3x +3 = 11",
        "(x+2)^2 = 12"
    ]

    for t in tests:
        result = parse_student_input(t)
        print(f"  Input:  {t:30s}  ->  {result}  (type: {type(result).__name__})")

def demo_validator():
    problem = "3x + 5 = 0"
    
    prev = parse_equation(problem)
    while True:
        print(prev)
        t = input()
        parsed = parse_student_input(t)
        res = validate_step(prev, parsed, "solve")
        print(res)
        if not res.is_correct:
            print(res.message)
        if res.status == StepStatus.COMPLETE:
            print("correct!")
            break
        prev = parsed



if __name__ == "__main__":
    demo_validator()