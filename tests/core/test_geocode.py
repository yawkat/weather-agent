import json

import pytest

from weather_core.geocode import ATTRIBUTION, GeocodeError, Geocoder, normalize

COLOGNE = [{"lat": "50.938361", "lon": "6.959974", "display_name": "Köln, Nordrhein-Westfalen, Deutschland"}]


class Fake:
    """Nominatim stand-in on a fake clock: records request times and URLs."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.now = 0.0
        self.requests = []  # (time, url)

    def get(self, url):
        self.requests.append((self.now, url))
        self.now += 0.3  # response time
        for name, hits in self.responses.items():
            if f"q={name}&" in url:
                return json.dumps(hits).encode()
        return b"[]"

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds

    def geocoder(self, **kwargs):
        return Geocoder(self.get, "https://nominatim.example/", clock=lambda: self.now, sleep=self.sleep, **kwargs)


def test_resolves_and_caches():
    fake = Fake({"cologne": COLOGNE})
    geocoder = fake.geocoder()
    result = geocoder.resolve_many(" Cologne ")
    assert result["attribution"] == ATTRIBUTION
    assert "OpenStreetMap contributors" in ATTRIBUTION
    [hit] = result["results"]
    assert hit == {"query": "Cologne", "lat": 50.93836, "lon": 6.95997,
                   "label": "Köln, Nordrhein-Westfalen, Deutschland", "place": "Cologne@50.93836,6.95997"}
    [(_, url)] = fake.requests
    assert url.startswith("https://nominatim.example/search?q=cologne&format=jsonv2&limit=1")
    # Cached whatever the spelling.
    assert geocoder.resolve_many("COLOGNE")["results"][0]["lat"] == 50.93836
    assert len(fake.requests) == 1


def test_requests_are_at_least_a_second_apart():
    fake = Fake({"cologne": COLOGNE, "bonn": COLOGNE, "aachen": COLOGNE})
    fake.geocoder().resolve_many("Cologne; Bonn; Aachen; cologne")
    times = [t for t, _ in fake.requests]
    assert len(times) == 3
    # Spacing counts from the end of the previous request (0.3 s here).
    assert all(b - a >= 1.3 - 1e-9 for a, b in zip(times, times[1:]))


def test_misses_are_cached():
    fake = Fake({})
    geocoder = fake.geocoder()
    assert geocoder.resolve_many("Nowhere")["results"] == [{"query": "Nowhere", "error": "not found"}]
    assert geocoder.resolve("nowhere") is None
    assert len(fake.requests) == 1


def test_least_recently_used_is_evicted():
    fake = Fake({"a": COLOGNE, "b": COLOGNE, "c": COLOGNE})
    geocoder = fake.geocoder(max_cached=2)
    geocoder.resolve("a")
    geocoder.resolve("b")
    geocoder.resolve("a")  # now b is the oldest
    geocoder.resolve("c")  # evicts b
    assert len(fake.requests) == 3
    geocoder.resolve("a")
    assert len(fake.requests) == 3
    geocoder.resolve("b")
    assert len(fake.requests) == 4


def test_input_limits():
    geocoder = Fake({}).geocoder()
    with pytest.raises(GeocodeError, match="empty"):
        geocoder.resolve_many(" ; ")
    with pytest.raises(GeocodeError, match="at most 10"):
        geocoder.resolve_many(";".join(f"p{i}" for i in range(11)))
    with pytest.raises(GeocodeError, match="200 characters"):
        geocoder.resolve("x" * 201)


def test_bad_response():
    fake = Fake({"broken": [{"lat": "north"}]})
    with pytest.raises(GeocodeError, match="unexpected geocoder response"):
        fake.geocoder().resolve("broken")


def test_place_string_is_safe_for_forecast():
    fake = Fake({"a%40b": COLOGNE})
    assert fake.geocoder().resolve_many("a@b")["results"][0]["place"] == "a b@50.93836,6.95997"


def test_normalize():
    assert normalize("  Freíburg   im  BREISGAU ") == normalize("Freíburg im Breisgau")
