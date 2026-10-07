from datetime import datetime, timedelta, timezone

import numpy as np

from weather_core.store import FieldStore

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)


def test_keeps_the_newest_two_runs(tmp_path):
    store = FieldStore(tmp_path)
    for i in range(3):
        store.put("ecmwf-ens", RUN + timedelta(hours=6 * i), "tp", 3, np.full((2, 4), float(i)))
    # The 00z run is gone; 06z (e.g. a long run that 12z's short range can't replace) is still there.
    assert store.runs("ecmwf-ens") == ["20261007T0600Z", "20261007T1200Z"]
    assert store.get("ecmwf-ens", RUN + timedelta(hours=6), "tp", 3)[0, 0] == 1.0
    assert not store.has("ecmwf-ens", RUN, "tp", 3)


def test_older_run_being_written_is_kept(tmp_path):
    store = FieldStore(tmp_path)
    for i in (1, 2):
        store.put("ecmwf-ens", RUN + timedelta(hours=6 * i), "tp", 3, np.zeros((2, 4)))
    # A query needing the older 00z run writes it; its own writes must not evict it.
    store.put("ecmwf-ens", RUN, "tp", 3, np.ones((2, 4)))
    store.put("ecmwf-ens", RUN, "tp", 6, np.ones((2, 4)))
    assert store.get("ecmwf-ens", RUN, "tp", 3)[0, 0] == 1.0
    assert store.has("ecmwf-ens", RUN, "tp", 6)
