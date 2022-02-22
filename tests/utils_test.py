from itertools import product

from xfit.utils import coordinate_slice, da_filter, gen_coord_combo
from xfit.utils import gen_sim_da, combine_new_ds_dim

import numpy as np
import xarray as xr

from numpy.testing import assert_array_equal
from xarray.testing import assert_equal

da1 = xr.DataArray(
    np.arange(3 * 4 * 5).reshape(3, 4, 5),
    dims=["a", "b", "c"],
    coords={"a": [1, 2, 3], "b": [11, 22, 33, 44], "c": [111, 222, 333, 444, 555]},
)

da2 = xr.DataArray(
    2 * np.arange(3 * 4 * 5).reshape(3, 4, 5),
    dims=["a", "b", "c"],
    coords={"a": [1, 2, 3], "b": [11, 22, 33, 44], "c": [111, 222, 333, 444, 555]},
)

da12 = xr.DataArray(
    np.concatenate(
        (
            np.arange(3 * 4 * 5).reshape(1, 3, 4, 5),
            2 * np.arange(3 * 4 * 5).reshape(1, 3, 4, 5),
        ),
        axis=0,
    ),
    dims=["new", "a", "b", "c"],
    coords={
        "a": [1, 2, 3],
        "b": [11, 22, 33, 44],
        "c": [111, 222, 333, 444, 555],
        "new": [9, 99],
    },
)


def test_coordinate_slice():
    actual1 = coordinate_slice(da1, "c", 200, 500)
    actual2 = coordinate_slice(da1, "c", 222, 500)
    actual3 = coordinate_slice(da1, "c", 200, 444)

    expected = np.array([222, 333, 444])

    assert np.all(actual1 == expected)
    assert np.all(actual2 == expected)
    assert np.all(actual3 == expected)


def test_da_filter():
    actual = da_filter(da1, selections={"a": 1})
    expected = da1.sel(a=[1])
    assert expected.equals(actual)

    actual = da_filter(da1, selections={"a": [1, 2]})
    expected = da1.sel(a=[1, 2])
    assert expected.equals(actual)

    actual = da_filter(da1, omissions={"b": 22})
    expected = da1.sel(b=[11, 33, 44])
    assert expected.equals(actual)

    actual = da_filter(da1, omissions={"b": [22, 44]})
    expected = da1.sel(b=[11, 33])
    assert expected.equals(actual)

    actual = da_filter(da1, ranges={"c": (100, 300)})
    expected = da1.sel(c=[111, 222])
    assert expected.equals(actual)

    actual = da_filter(da1, ranges={"c": (111, 300)})
    expected = da1.sel(c=[111, 222])
    assert expected.equals(actual)

    actual = da_filter(da1, ranges={"c": (100, 333)})
    expected = da1.sel(c=[111, 222, 333])
    assert expected.equals(actual)

    actual = da_filter(da1, ranges={"c": (200, np.inf)})
    expected = da1.sel(c=[222, 333, 444, 555])
    assert expected.equals(actual)

    actual = da_filter(da1, ranges={"c": (-np.inf, 400)})
    expected = da1.sel(c=[111, 222, 333])
    assert expected.equals(actual)


def test_gen_coord_combo():
    actual_d, actual_v = gen_coord_combo(da1)
    expected_v = product([1, 2, 3], [11, 22, 33, 44], [111, 222, 333, 444, 555])
    expected_d = ["a", "b", "c"]
    assert expected_d.sort() == actual_d.sort()
    # TODO: ensure dim ordering lines up
    assert all([x == y for x, y in zip(actual_v, expected_v)])

    actual_d, actual_v = gen_coord_combo(da1, drop_dims="c")
    expected_v = product([1, 2, 3], [11, 22, 33, 44])
    expected_d = ["a", "b"]
    assert expected_d.sort() == actual_d.sort()
    # TODO: ensure dim ordering lines up
    assert all([x == y for x, y in zip(actual_v, expected_v)])


def test_gen_sim_da():
    new_dvars = ["first", "second"]
    actual = gen_sim_da(da1)

    assert np.all(np.isnan(actual))
    assert list(actual.coords) == list(da1.coords)
    for c in actual.coords:
        assert_equal(actual.coords[c], da1.coords[c])

    actual2 = gen_sim_da(da1, drop_dims=["c"])
    assert np.all(np.isnan(actual2))
    expected_coords = [c for c in da1.coords if c != "c"]
    assert list(actual2.coords) == list(expected_coords)
    for c in actual2.coords:
        assert_equal(actual2.coords[c], da1.coords[c])

    d_vals = np.array([12.0, 13.0, 14.0])
    actual3 = gen_sim_da(da1, new_coords={"d": d_vals})
    assert np.all(np.isnan(actual3))
    expected_coords = ["d", "a", "b", "c"]
    assert set(actual3.coords) == set(expected_coords)
    for c in da1.coords:
        assert_equal(actual3.coords[c], da1.coords[c])
    assert_array_equal(actual3.coords["d"].values, d_vals)


def test_combine_new_ds_dim():
    expected = combine_new_ds_dim({9: da1, 99: da2}, "new")
    assert_equal(expected, da12)

    expected2 = combine_new_ds_dim([da1, da2], "new", [9, 99])
    assert_equal(expected2, da12)
