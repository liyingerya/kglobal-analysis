"""KGlobal metadata inspection and dimension-aware movie reconstruction."""

from .case import KGlobalCase
from .distribution import ReducedDistribution, DistributionError, DistributionMetadataError
from .energy import EnergySpectrum, EnergySpectrumError, EnergyMetadataError
from .dataset import MovieAlignmentError
from .format import MovieFormat, VariableSpec, VolumeLayout, STANDARD_MOVIE_FORMAT
from .index import CaseIndex, Segment, parse_filename, scan_case
from .manifest import MovieCaseReport, MovieValidationIssue
from .parameters import Parameters, parse_parameters, read_parameters
from .segment import BxSegment, MovieSegment
from .times import TimeMetadataError
from .series import BxSeries, MovieSeries

__all__ = [
    "ReducedDistribution", "DistributionError", "DistributionMetadataError",
    "EnergySpectrum", "EnergySpectrumError", "EnergyMetadataError", "MovieCaseReport", "MovieValidationIssue", "MovieAlignmentError", "KGlobalCase", "CaseIndex", "Segment", "parse_filename", "scan_case",
    "Parameters", "parse_parameters", "read_parameters", "BxSegment", "TimeMetadataError", "BxSeries",
    "MovieSegment", "MovieSeries", "MovieFormat", "VariableSpec", "VolumeLayout", "STANDARD_MOVIE_FORMAT",
]

# User-facing orchestration; specialist science stays in its own modules.
from .workflows import (
    particle_quantity_catalog, particle_quantity_series, firehose_series,
    magnetic_flux_series, particle_map_series, firehose_map_series,
)

__all__ += [
    'particle_quantity_catalog', 'particle_quantity_series', 'firehose_series',
    'magnetic_flux_series', 'particle_map_series', 'firehose_map_series',
]
