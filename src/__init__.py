"""Root package for Internal Support Copilot."""

from src.core.logging_utils import configure_logging
from src.core.observability import configure_observability

configure_logging()
configure_observability()
