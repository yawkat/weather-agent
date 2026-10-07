import numpy as np
import pytest

from weather_core.variables import apparent_temperature, available, derive, relative_humidity


def test_wind_speed_and_direction_from_components():
    # u = 0, v = -10: wind blowing towards the south, i.e. from the north (0°).
    out = derive({"wind_u": np.array([[0.0, 10.0]]), "wind_v": np.array([[-10.0, 0.0]])})
    assert out["wind"].tolist() == [[10.0, 10.0]]
    # u = 10, v = 0: blowing towards the east, from the west (270°).
    assert out["wind_dir"] == pytest.approx(np.array([[0.0, 270.0]]))


def test_headwind_on_route():
    # North wind, rider heading north then south.
    out = derive({"wind_u": np.array([[0.0, 0.0]]), "wind_v": np.array([[-10.0, -10.0]])},
                 bearing=np.array([0.0, 180.0]))
    assert out["headwind"] == pytest.approx(np.array([[10.0, -10.0]]))


def test_relative_humidity():
    assert relative_humidity(20.0, 20.0) == pytest.approx(100.0)
    assert relative_humidity(20.0, 10.0) == pytest.approx(52.6, abs=0.5)


def test_apparent_temperature():
    # Wind chill at 0 °C and 20 km/h is about -5 °C.
    assert apparent_temperature(np.array([0.0]), np.array([20.0]))[0] == pytest.approx(-5.2, abs=0.2)
    # Mild weather: no adjustment.
    assert apparent_temperature(np.array([18.0]), np.array([20.0]), np.array([60.0]))[0] == 18.0
    # 32 °C at 60 % humidity feels like ~37 °C.
    assert apparent_temperature(np.array([32.0]), np.array([5.0]), np.array([60.0]))[0] == pytest.approx(37, abs=1)


def test_available_variables():
    names = available({"t2m", "wind_u", "wind_v", "precip"}, route=True)
    assert {"wind", "wind_dir", "feels_like", "headwind", "precip"} <= names
    assert "rh" not in names
    assert "headwind" not in available({"wind_u", "wind_v"}, route=False)
