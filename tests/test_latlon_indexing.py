from my_vessel.bathy.grid import latlon_to_rc, rc_to_latlon


def test_rc_ll_roundtrip():
    bounds = (60.0, 20.0, 61.0, 21.0)
    shape = (100, 100)
    r, c = latlon_to_rc(60.5, 20.5, bounds, shape)
    lat, lon = rc_to_latlon(r, c, bounds, shape)
    assert abs(lat - 60.5) < 0.02 and abs(lon - 20.5) < 0.02
