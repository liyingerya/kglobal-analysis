"""KGlobal metadata inspection and dimension-aware movie reconstruction."""

from .case import KGlobalCase
from .format import MovieFormat, VariableSpec, VolumeLayout, STANDARD_MOVIE_FORMAT
from .index import CaseIndex, Segment, parse_filename, scan_case
from .parameters import Parameters, parse_parameters, read_parameters
from .segment import BxSegment, MovieSegment
from .times import TimeMetadataError
from .series import BxSeries, MovieSeries

__all__ = [
    "KGlobalCase", "CaseIndex", "Segment", "parse_filename", "scan_case",
    "Parameters", "parse_parameters", "read_parameters", "BxSegment", "TimeMetadataError", "BxSeries",
    "MovieSegment", "MovieSeries", "MovieFormat", "VariableSpec", "VolumeLayout", "STANDARD_MOVIE_FORMAT",
]
