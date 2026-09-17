"""Metadata-only inspection of KGlobal simulation cases."""

from .case import KGlobalCase
from .index import CaseIndex, Segment, parse_filename, scan_case
from .parameters import Parameters, parse_parameters, read_parameters

__all__ = [
    "KGlobalCase", "CaseIndex", "Segment", "parse_filename", "scan_case",
    "Parameters", "parse_parameters", "read_parameters",
]
