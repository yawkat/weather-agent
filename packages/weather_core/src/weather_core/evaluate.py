"""Which model run samples came from (feeds the model table and attribution)."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SourceInfo:
    source: str  # e.g. "ecmwf-ens"
    model: str  # e.g. "IFS ENS 0.25°"
    provider: str  # e.g. "ECMWF"
    run: datetime
    members: int
