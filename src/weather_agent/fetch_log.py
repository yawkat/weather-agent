from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated

from micronaut.data.annotation import GeneratedValue, Id, MappedEntity
from micronaut.data.jdbc.annotation import JdbcRepository
from micronaut.data.model.query.builder.sql import Dialect
from micronaut.data.repository import CrudRepository


@dataclass
@MappedEntity("fetch_log")
class FetchLogEntry:
    """One batch of downloaded model data. Times are naive UTC: Pyronaut maps datetimes to naive Java types."""
    id: Annotated[int | None, Id, GeneratedValue]
    source: str
    run_utc: datetime
    bytes: int
    fetched_at_utc: datetime


def naive_utc(t: datetime) -> datetime:
    return t.astimezone(timezone.utc).replace(tzinfo=None)


@JdbcRepository(dialect=Dialect.POSTGRES)
class FetchLogRepository(CrudRepository[FetchLogEntry, int]):
    pass
