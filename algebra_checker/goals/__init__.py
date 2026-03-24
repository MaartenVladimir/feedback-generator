"""
Goal package — auto-discovers all Goal subclasses.

To add a new goal:
  1. Create a new file in this directory (e.g. factor_expression.py).
  2. Subclass Goal and implement check_step() and description.
  3. That's it — no changes to __init__.py or any other file.

Goals are accessible via get_goal("ClassName") or get_all_goals().
"""

import importlib
import pkgutil
from pathlib import Path

from .base import Goal


def _load_all_goal_modules() -> None:
    """Import every module in this package so Goal subclasses register themselves."""
    package_dir = str(Path(__file__).parent)
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        if module_name != "base":
            importlib.import_module(f".{module_name}", package=__name__)


def _all_subclasses(cls: type) -> list[type]:
    result = []
    for sub in cls.__subclasses__():
        result.append(sub)
        result.extend(_all_subclasses(sub))
    return result


def get_all_goals() -> dict[str, type[Goal]]:
    """Return all registered Goal subclasses, keyed by class name."""
    _load_all_goal_modules()
    return {cls.__name__: cls for cls in _all_subclasses(Goal)}


def get_goal(name: str) -> type[Goal]:
    """Look up a Goal subclass by name. Raises KeyError if not found."""
    goals = get_all_goals()
    if name not in goals:
        raise KeyError(f"No goal named '{name}'. Available: {sorted(goals)}")
    return goals[name]


__all__ = ["Goal", "get_all_goals", "get_goal"]
