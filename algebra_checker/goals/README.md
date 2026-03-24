# Goals

A **Goal** represents one type of mathematical task a student can work on — for example, solving an equation or factoring an expression. Each goal owns the full pipeline for one student step: parsing, correctness checking, completion detection, and strategic feedback.

## How a Goal works

A Goal subclass implements two things:

1. **`check_step(prev_raw, new_raw) -> StepResult`** — the full check for one student step. It receives two raw strings (previous state and new state), and returns a `StepResult` with correctness, completion, error diagnosis, and strategy feedback.

2. **`description`** — a short human-readable label shown in the UI (e.g. `"Solve the equation for x"`).

Error detection is handled by declaring **`default_error_checks`** — a list of `ErrorChecker` instances that `diagnose_error()` runs in order when a step is incorrect. See [`../errors/README.md`](../errors/README.md).

```
check_step()
  ├── parse prev_raw and new_raw
  ├── validate correctness          ← validator.py (algebraic equivalence)
  ├── if incorrect: diagnose_error()  ← runs default_error_checks in order
  ├── if correct: check completion    ← goal-specific logic (owned here)
  └── if correct and not complete: assess_strategy()  ← optional, goal-specific
```

## Adding a new goal

**Just add a new file.** The package auto-discovers all `Goal` subclasses — no changes to `__init__.py` or any other file are needed.

### Minimal example

```python
# algebra_checker/goals/factor_expression.py
from sympy import factor, Expr
from sympy import expand

from ..parser import ParseError, parse_expr_safe
from ..validator import validate_expression_step, StepResult, StepStatus
from .base import Goal


def _is_fully_factored(expr: Expr) -> bool:
    return factor(expr) == expr


class FactorExpressionGoal(Goal):
    """Goal: fully factor a polynomial expression."""

    default_error_checks = []  # add ErrorCheckers here as needed

    @property
    def description(self) -> str:
        return "Factor the expression completely"

    def check_step(self, prev_raw: str, new_raw: str) -> StepResult:
        # 1. Parse
        try:
            prev_expr = parse_expr_safe(prev_raw)
            new_expr = parse_expr_safe(new_raw)
        except ParseError as e:
            return StepResult(status=StepStatus.PARSE_ERROR, is_correct=False, message=str(e))

        # 2. Correctness (algebraic equivalence)
        result = validate_expression_step(prev_expr, new_expr)
        if not result.is_correct:
            error_id, diagnosis = self.diagnose_error(prev_expr, new_expr)
            result.error_id = error_id
            result.error_diagnosis = diagnosis
            result.message = diagnosis
            return result

        # 3. Completion (owned by this goal)
        if _is_fully_factored(new_expr):
            result.status = StepStatus.COMPLETE
            result.message = "Correct! The expression is fully factored."

        return result
```

### Checklist for a new goal

- [ ] Create `algebra_checker/goals/<your_goal>.py`
- [ ] Subclass `Goal` and implement `check_step()` and `description`
- [ ] Call `validate_equation_step` or `validate_expression_step` from `validator.py` for correctness
- [ ] Handle completion in `check_step()` after the correctness check
- [ ] Declare `default_error_checks` with any relevant `ErrorChecker` instances
- [ ] Add strategy feedback if applicable (see `solve_equation.py` for an example)

## Using goals

```python
from algebra_checker import get_goal, get_all_goals

# Look up by name
goal = get_goal("SolveEquationGoal")()
result = goal.check_step("2x + 1 = 5", "2x = 4")

# See all available goals
print(get_all_goals())  # {"SolveEquationGoal": <class ...>, "FactorExpressionGoal": <class ...>}

# Instantiate with extra error checks for a specific problem
from algebra_checker.errors.equation_errors import sign_error_on_move
goal = get_goal("SolveEquationGoal")(extra_checks=[sign_error_on_move])
```
