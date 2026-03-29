from .base import ErrorChecker
from .template_checker import load_templates, TEMPLATES_DIR

error_checkers, strategy_checks = load_templates(TEMPLATES_DIR)

__all__ = [
    "ErrorChecker",
    "error_checkers",
    "strategy_checks",
    "load_templates",
    "TEMPLATES_DIR",
]
