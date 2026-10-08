"""Place names → coordinates via Nominatim, within its usage policy.

https://operations.osmfoundation.org/policies/nominatim/: at most one request per second, an identifying
User-Agent, no autocomplete, cache results, and credit "© OpenStreetMap contributors". Requests are serialised
behind a lock and spaced by `min_interval`. Results, including misses, are kept in an in-memory LRU cache, so
it starts empty after a restart.
"""

import json
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlencode

ATTRIBUTION = "© OpenStreetMap contributors (ODbL 1.0), geocoding by Nominatim"
MAX_QUERY_CHARS = 200
MAX_QUERIES = 10
MAX_CACHED = 10_000  # names, found or not; an entry is a few hundred bytes


@dataclass(frozen=True)
class Place:
    query: str  # normalised, the cache key
    lat: float
    lon: float
    label: str


class GeocodeError(ValueError):
    """Bad input or upstream failure; reported to the client."""


def normalize(query: str) -> str:
    """Cache key: NFC, case-folded, whitespace collapsed. "  Köln " and "köln" share an entry."""
    return " ".join(unicodedata.normalize("NFC", query).casefold().split())


class Geocoder:
    def __init__(self, get: Callable[[str], bytes | None], base_url: str, accept_language: str = "en",
                 min_interval: float = 1.0, max_cached: int = MAX_CACHED,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        """`get(url)` returns the body, or None on 404; it must send the identifying User-Agent."""
        self._get = get
        self._base_url = base_url.rstrip("/")
        self._accept_language = accept_language
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._max_cached = max_cached
        self._request_lock = threading.Lock()  # one request at a time, as the policy asks
        self._next_request = float("-inf")
        self._cache_lock = threading.Lock()
        self._cache: dict[str, Place | None] = {}  # None = not found; insertion order is recency, oldest first

    def resolve_many(self, text: str) -> dict:
        """"Cologne; Bonn" → one result per name, plus attribution."""
        names = [n.strip() for n in text.split(";") if n.strip()]
        if not names:
            raise GeocodeError("query is empty")
        if len(names) > MAX_QUERIES:
            raise GeocodeError(f"at most {MAX_QUERIES} places per call")
        results = []
        for name in names:
            place = self.resolve(name)
            if place is None:
                results.append({"query": name, "error": "not found"})
            else:
                results.append({
                    "query": name,
                    "lat": round(place.lat, 5),
                    "lon": round(place.lon, 5),
                    "label": place.label,
                    # Ready for places("…") in forecast queries; the name is the caller's, so it can't contain ';' or '@'.
                    "place": f"{name.replace('@', ' ')}@{place.lat:.5f},{place.lon:.5f}",
                })
        return {"results": results, "attribution": ATTRIBUTION}

    def resolve(self, name: str) -> Place | None:
        if len(name) > MAX_QUERY_CHARS:
            raise GeocodeError(f"place names are limited to {MAX_QUERY_CHARS} characters")
        key = normalize(name)
        if not key:
            raise GeocodeError("query is empty")
        found, place = self._cached(key)
        if found:
            return place
        with self._request_lock:
            # Another caller may have fetched it while we waited.
            found, place = self._cached(key)
            if found:
                return place
            place = self._fetch(key)
            with self._cache_lock:
                self._cache[key] = place
                while len(self._cache) > self._max_cached:
                    del self._cache[next(iter(self._cache))]
            return place

    def _cached(self, key: str) -> tuple[bool, Place | None]:
        with self._cache_lock:
            if key not in self._cache:
                return False, None
            place = self._cache.pop(key)  # re-insert as most recently used
            self._cache[key] = place
            return True, place

    def _fetch(self, key: str) -> Place | None:
        wait = self._next_request - self._clock()
        if wait > 0:
            self._sleep(wait)
        url = f"{self._base_url}/search?" + urlencode({
            "q": key, "format": "jsonv2", "limit": 1, "accept-language": self._accept_language})
        try:
            body = self._get(url)
        finally:
            # Spacing counts from the end of the request, so a slow response never leads to a burst.
            self._next_request = self._clock() + self._min_interval
        if body is None:
            raise GeocodeError("geocoder unavailable (HTTP 404)")
        try:
            hits = json.loads(body)
            if not hits:
                return None
            hit = hits[0]
            return Place(query=key, lat=float(hit["lat"]), lon=float(hit["lon"]), label=str(hit["display_name"]))
        except (ValueError, KeyError, TypeError, IndexError) as e:
            raise GeocodeError(f"unexpected geocoder response: {type(e).__name__}") from None
