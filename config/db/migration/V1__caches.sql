-- Shared, non-personal state only: no users, places or routes are stored.
-- Timestamps are UTC, stored without time zone (Pyronaut maps Python datetimes to naive Java types).
CREATE TABLE geocode_cache (
    query          TEXT             PRIMARY KEY,
    lat            DOUBLE PRECISION NOT NULL,
    lon            DOUBLE PRECISION NOT NULL,
    label          TEXT             NOT NULL,
    fetched_at_utc TIMESTAMP        NOT NULL
);

CREATE TABLE fetch_log (
    id             BIGSERIAL PRIMARY KEY,
    source         TEXT      NOT NULL,
    run_utc        TIMESTAMP NOT NULL,
    bytes          BIGINT    NOT NULL,
    fetched_at_utc TIMESTAMP NOT NULL
);
