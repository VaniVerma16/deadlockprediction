"""Beginner-friendly runtime deadlock detection prototype."""

from .graph import build_snapshot
from .inference import classify_snapshot

__all__ = ["build_snapshot", "classify_snapshot"]
