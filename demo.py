from algebra_checker.validator import StepStatus, StrategyRating
from algebra_checker.goals.solve_equation import SolveEquationGoal

PROBLEMS = [
    "x/2 + x/3 = 5",
    "x/2 + 1 = 3",
    "(x + 1)/3 = 2",
    "x/4 + x/6 = 5",
]


def print_feedback(result):
    if result.status == StepStatus.PARSE_ERROR:
        print(f"  [Parse error] {result.message}")
        return

    if result.status == StepStatus.COMPLETE:
        print("  [Complete] Problem solved!")
        return

    if not result.is_correct:
        print(f"  [Incorrect] {result.error_diagnosis or result.message}")
        return

    rating_label = {
        StrategyRating.OPTIMAL: "Correct",
        StrategyRating.SUBOPTIMAL: "Correct, but suboptimal",
        StrategyRating.COUNTERPRODUCTIVE: "Correct, but counterproductive",
    }.get(result.strategy_rating, "Correct")

    print(f"  [{rating_label}]", end="")
    if result.strategy_message:
        print(f" {result.strategy_message}", end="")
    print()


def pick_problem():
    print("\nChoose a problem:")
    for i, p in enumerate(PROBLEMS, 1):
        print(f"  {i}. {p}")
    print(f"  {len(PROBLEMS) + 1}. Enter your own")

    while True:
        choice = input("\n> ").strip()
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(PROBLEMS):
                return PROBLEMS[n - 1]
            if n == len(PROBLEMS) + 1:
                return input("Enter equation: ").strip()
        print("  Invalid choice, try again.")


def run():
    print("=" * 60)
    print("  Algebra step checker — interactive demo")
    print("  Type 'quit' to exit, 'restart' for a new problem")
    print("=" * 60)

    while True:
        problem = pick_problem()
        goal = SolveEquationGoal()
        prev = problem

        print(f"\n  Solve: {problem}")
        print("  Enter your steps one at a time.\n")

        while True:
            try:
                step = input("  > ").strip()
            except EOFError:
                return

            if step.lower() == "quit":
                return
            if step.lower() == "restart":
                break
            if not step:
                continue

            result = goal.check_step(prev, step)
            print_feedback(result)

            if result.status == StepStatus.COMPLETE:
                print()
                break

            if result.is_correct:
                prev = step


if __name__ == "__main__":
    run()
