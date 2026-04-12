"""
Goal: student fills in a single correct answer (no step-by-step solving).

Used for questions like "find the slope through two points" where the student
only needs to produce a final numeric or algebraic answer.

Item JSON must include:
    "goal": "direct_answer",
    "expected_answer": "2"          # any SymPy-parseable expression
    "expected_answer_display": "2"  # optional LaTeX for the feedback message

Optionally:
    "answer_label": "rc"            # variable name shown in the input prompt
"""

from sympy import simplify, expand

from ..parser import ParseError, parse_expr_safe
from ..validator import StepResult, StepStatus
from .base import Goal


def _equivalent(a, b) -> bool:
    try:
        diff = simplify(expand(a) - expand(b))
        return diff == 0 or simplify(diff) == 0
    except Exception:
        return False


class DirectAnswerGoal(Goal):
    """
    Goal: produce the single correct answer for a question.

    check_step() ignores prev_raw (there are no intermediate steps).
    It parses the student's input and compares it symbolically to
    item_context['expected_answer'].
    """

    default_error_checks = []  # no step-level error patterns apply here

    @property
    def description(self) -> str:
        return "Fill in the correct answer"

    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        expected_raw = self.item_context.get('expected_answer', '').strip()
        if not expected_raw:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message="This question has no expected answer configured.",
            )

        try:
            student_expr = parse_expr_safe(new_raw)
        except ParseError as e:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message=f"Could not parse your answer: {e.reason}",
            )

        try:
            expected_expr = parse_expr_safe(expected_raw)
        except ParseError:
            return StepResult(
                status=StepStatus.PARSE_ERROR,
                is_correct=False,
                message="Question configuration error: expected answer could not be parsed.",
            )

        if _equivalent(student_expr, expected_expr):
            return StepResult(
                status=StepStatus.COMPLETE,
                is_correct=True,
                message="Correct!",
                canonical_form=str(expected_expr),
            )

        for hint in self.item_context.get('wrong_answer_hints', []):
            try:
                hint_expr = parse_expr_safe(hint['if_answer'])
                if _equivalent(student_expr, hint_expr):
                    return StepResult(
                        status=StepStatus.INCORRECT,
                        is_correct=False,
                        message=hint['message'],
                    )
            except Exception:
                continue

        return StepResult(
            status=StepStatus.INCORRECT,
            is_correct=False,
            message="Dat klopt niet. Probeer het nog eens.",
        )
