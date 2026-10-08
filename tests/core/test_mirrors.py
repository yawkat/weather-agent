"""Fallback between the origin and its mirrors."""

import pytest

from weather_core.sources.base import Download, NotPublished, SourceError
from weather_core.sources.mirrors import Mirrors

ORIGIN, A, B = "https://origin.test", "https://a.test", "https://b.test"


class Hosts:
    """Fake fetcher: each host is "ok", "missing" (404) or "down" (fails after retries)."""

    def __init__(self, **state):
        self.state = {f"https://{host}.test": s for host, s in state.items()}
        self.calls = []

    def _answer(self, url):
        self.calls.append(url)
        state = self.state[url[:url.index("/", len("https://"))]]
        if state == "down":
            raise SourceError(f"{url}: HTTP 429")
        return None if state == "missing" else b"body"

    def get(self, url):
        return self._answer(url)

    def download_many(self, requests):
        for r in requests:
            if self._answer(r.url) is None:
                raise NotPublished(f"{r.url}: HTTP 404")
        return [1] * len(requests)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def test_origin_is_used_while_it_works(clock):
    hosts = Hosts(origin="ok", a="ok", b="ok")
    assert Mirrors(ORIGIN, [A, B], clock=clock).get(hosts, "f") == b"body"
    assert hosts.calls == [f"{ORIGIN}/f"]


def test_origin_404_is_final(clock):
    hosts = Hosts(origin="missing", a="ok", b="ok")
    assert Mirrors(ORIGIN, [A, B], clock=clock).get(hosts, "f") is None
    assert hosts.calls == [f"{ORIGIN}/f"]


def test_failing_origin_is_skipped_until_cooldown_ends(clock):
    hosts = Hosts(origin="down", a="ok", b="ok")
    mirrors = Mirrors(ORIGIN, [A, B], cooldown_s=300, clock=clock)
    assert mirrors.get(hosts, "f") == b"body"
    assert mirrors.get(hosts, "g") == b"body"
    assert hosts.calls == [f"{ORIGIN}/f", f"{A}/f", f"{A}/g"]
    clock.now = 301
    hosts.state[ORIGIN] = "ok"
    hosts.calls.clear()
    assert mirrors.get(hosts, "h") == b"body"
    assert hosts.calls == [f"{ORIGIN}/h"]


def test_mirror_404_asks_the_next_mirror_but_not_cooling_hosts(clock):
    hosts = Hosts(origin="down", a="missing", b="ok")
    mirrors = Mirrors(ORIGIN, [A, B], clock=clock)
    assert mirrors.get(hosts, "f") == b"body"
    hosts.state[B] = "missing"
    hosts.calls.clear()
    # The origin is cooling down: a mirror lagging behind doesn't make us retry it.
    assert mirrors.get(hosts, "g") is None
    assert hosts.calls == [f"{A}/g", f"{B}/g"]


def test_cooling_hosts_are_tried_when_nothing_else_answers(clock):
    hosts = Hosts(origin="down", a="down", b="down")
    mirrors = Mirrors(ORIGIN, [A, B], clock=clock)
    with pytest.raises(SourceError):
        mirrors.get(hosts, "f")
    clock.now = 10
    hosts.state[ORIGIN] = "ok"
    hosts.calls.clear()
    assert mirrors.get(hosts, "g") == b"body"
    assert hosts.calls == [f"{ORIGIN}/g"]  # failed longest ago, so tried first


def test_download_falls_back_with_the_whole_batch(clock):
    hosts = Hosts(origin="down", a="ok", b="ok")
    downloads = [Download("x.grib2", [(0, 9)], "/dev/null"), Download("y.grib2", None, "/dev/null")]
    assert Mirrors(ORIGIN, [A, B], clock=clock).download_many(hosts, downloads) == [1, 1]
    assert hosts.calls == [f"{ORIGIN}/x.grib2", f"{A}/x.grib2", f"{A}/y.grib2"]


DOWNLOADS = [Download("x.grib2", None, "/dev/null")]


def test_download_after_mirror_lag_tries_cooling_hosts_without_muting_the_mirror(clock):
    hosts = Hosts(origin="down", a="ok", b="ok")
    mirrors = Mirrors(ORIGIN, [A, B], clock=clock)
    assert mirrors.get(hosts, "f.index") == b"body"  # the origin starts cooling down
    hosts.state.update({ORIGIN: "ok", A: "missing", B: "missing"})
    hosts.calls.clear()
    # The index came from somewhere, so the file exists: the cooling origin is asked after both mirrors' 404s.
    assert mirrors.download_many(hosts, DOWNLOADS) == [1]
    assert hosts.calls == [f"{A}/x.grib2", f"{B}/x.grib2", f"{ORIGIN}/x.grib2"]
    assert [b for b, cooling in mirrors._order() if not cooling] == [A, B]


def test_download_404_from_the_origin_is_final(clock):
    hosts = Hosts(origin="missing", a="ok", b="ok")
    with pytest.raises(NotPublished):
        Mirrors(ORIGIN, [A, B], clock=clock).download_many(hosts, DOWNLOADS)
    assert hosts.calls == [f"{ORIGIN}/x.grib2"]


def test_each_repeated_download_is_reserved(clock):
    hosts = Hosts(origin="down", a="down", b="ok")
    reserved = []
    assert Mirrors(ORIGIN, [A, B], clock=clock).download_many(hosts, DOWNLOADS, lambda: reserved.append(1)) == [1]
    assert len(reserved) == 2  # the first attempt is the caller's own reservation


def test_exhausted_budget_stops_the_fallback(clock):
    hosts = Hosts(origin="down", a="ok", b="ok")

    def reserve():
        raise SourceError("per-query limit")

    with pytest.raises(SourceError, match="per-query limit"):
        Mirrors(ORIGIN, [A, B], clock=clock).download_many(hosts, DOWNLOADS, reserve)
    assert hosts.calls == [f"{ORIGIN}/x.grib2"]


def test_only_https():
    with pytest.raises(ValueError):
        Mirrors(ORIGIN, ["http://plain.test"])
