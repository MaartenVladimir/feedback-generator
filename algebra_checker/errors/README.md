# Errors

An **ErrorChecker** is a reusable, named check that inspects two consecutive student states and either returns a specific feedback message or `None` if the error pattern is not present.

Error checkers are goal-agnostic — they detect algebraic mistakes that can occur across multiple goal types, and are shared by importing them into whichever goals need them.

## How an ErrorChecker works

```python
@dataclass(frozen=True)
class ErrorChecker:
    id: str          # stable snake_case identifier used in system logs
    description: str # one-line human description of the error pattern
    check: CheckFn   # (prev, new) -> str | None
```

The `check` function follows a **two-phase pattern**:

1. **Pattern match** — does this transition look like the error? If not, return `None` immediately so the next checker gets a turn.
2. **Diagnosis** — the error is confirmed; return a specific, helpful message.

```python
def _check_my_error(prev, new) -> str | None:
    # Phase 1: does this look like the error?
    if <pattern does not match>:
        return None

    # Phase 2: confirm and describe the mistake
    return "Specific message explaining what went wrong and what should have happened."
```

Returning `None` is the correct behaviour when a check doesn't apply. Checkers run in order; the first one that returns a message wins.

## Adding a new error checker

Create it in an existing file or a new file in this directory. There is no registration step — checkers are referenced directly by the goals that use them.

### Example

```python
# algebra_checker/errors/equation_errors.py  (or a new file)
from sympy import Eq, expand, simplify
from ..parser import x
from .base import ErrorChecker


def _check_partial_multiplication(prev_eq: Eq, new_eq: Eq) -> str | None:
    # Phase 1: was this a multiplication step? (RHS changed by a numeric factor)
    try:
        factor = simplify(new_eq.rhs / prev_eq.rhs)
        if not factor.is_number or factor == 1:
            return None
    except Exception:
        return None

    # Phase 2: was the same factor NOT applied to every LHS term?
    lhs_ratio = simplify(new_eq.lhs / prev_eq.lhs)
    if simplify(lhs_ratio - factor) != 0:
        return (
            f"It looks like you multiplied by {factor} but not every term. "
            f"Both sides must be multiplied by the same value throughout."
        )
    return None


partial_multiplication = ErrorChecker(
    id="partial_multiplication",
    description="Student multiplied only some terms instead of both sides completely.",
    check=_check_partial_multiplication,
)
```

### Checklist for a new error checker

- [ ] Write a `_check_<name>(prev, new) -> str | None` function
- [ ] Phase 1: return `None` if the pattern doesn't match (don't assume)
- [ ] Phase 2: return a message that says what went wrong *and* what should have happened
- [ ] Wrap it in an `ErrorChecker(id=..., description=..., check=...)` instance
- [ ] Use a stable `id` in `snake_case` — this is what gets recorded in system logs
- [ ] Add the instance to `default_error_checks` in any Goal that should use it

## Attaching error checkers to a goal

Goals declare which checkers they use via `default_error_checks`:

```python
from ..errors.equation_errors import sign_error_on_move, wrong_division, partial_multiplication

class SolveEquationGoal(Goal):
    default_error_checks = [sign_error_on_move, wrong_division, partial_multiplication]
```

Order matters — checkers run in sequence and the first match wins. Put the most specific checks first.

The DLE can also inject extra checks at instantiation time without modifying the goal:

```python
goal = SolveEquationGoal(extra_checks=[my_problem_specific_check])
```
