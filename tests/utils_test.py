from itertools import product

from xfit.utils import coordinate_slice, da_filter, gen_coord_combo
from xfit.utils import gen_copy_ds, combine_new_ds_dim

import numpy as np
import xarray as xr

from xarray.testing import assert_equal, assert_allclose

da1 = xr.DataArray(
    np.arange(3*4*5).reshape(3,4,5), 
    dims=['a','b','c'], 
    coords={
        'a': [1,2,3], 
        'b': [11,22,33,44],
        'c': [111,222,333,444,555]
        }
    )

da2 = xr.DataArray(
    2*np.arange(3*4*5).reshape(3,4,5), 
    dims=['a','b','c'], 
    coords={
        'a': [1,2,3], 
        'b': [11,22,33,44],
        'c': [111,222,333,444,555]
        }
    )

da12 = xr.DataArray(
    np.concatenate(
        (np.arange(3*4*5).reshape(1,3,4,5), 2*np.arange(3*4*5).reshape(1,3,4,5)),
        axis=0
    ),
    dims=['new','a','b','c'], 
    coords={
        'a': [1,2,3], 
        'b': [11,22,33,44],
        'c': [111,222,333,444,555],
        'new': [9, 99]
        }
    )

def test_coordinate_slice():
    actual1 = coordinate_slice(da1, 'c', 200, 500)
    actual2 = coordinate_slice(da1, 'c', 222, 500)
    actual3 = coordinate_slice(da1, 'c', 200, 444)
    
    expected = np.array([222,333,444])
    
    assert np.all(actual1 == expected)
    assert np.all(actual2 == expected)
    assert np.all(actual3 == expected)
    
    
def test_da_filter():
    actual = da_filter(da1, selections={'a': 1})
    expected = da1.sel(a=[1])
    assert expected.equals(actual)
    
    actual = da_filter(da1, selections={'a': [1,2]})
    expected = da1.sel(a=[1,2])
    assert expected.equals(actual)
    
    actual = da_filter(da1, omissions={'b': 22})
    expected = da1.sel(b=[11,33,44])
    assert expected.equals(actual)
    
    actual = da_filter(da1, omissions={'b': [22,44]})
    expected = da1.sel(b=[11,33])
    assert expected.equals(actual)
    
    actual = da_filter(da1, ranges={'c': (100,300)})
    expected = da1.sel(c=[111,222])
    assert expected.equals(actual)
    
    actual = da_filter(da1, ranges={'c': (111,300)})
    expected = da1.sel(c=[111,222])
    assert expected.equals(actual)
    
    actual = da_filter(da1, ranges={'c': (100,333)})
    expected = da1.sel(c=[111,222,333])
    assert expected.equals(actual)
    
    actual = da_filter(da1, ranges={'c': (200,np.inf)})
    expected = da1.sel(c=[222,333,444,555])
    assert expected.equals(actual)
    
    actual = da_filter(da1, ranges={'c': (-np.inf,400)})
    expected = da1.sel(c=[111,222,333])
    assert expected.equals(actual)
    
    
def test_gen_coord_combo():
    actual_d, actual_v = gen_coord_combo(da1)
    expected_v = product([1,2,3], [11,22,33,44], [111,222,333,444,555])
    expected_d = ['a', 'b', 'c']
    assert expected_d.sort() == actual_d.sort()
    # TODO: ensure dim ordering lines up
    assert all([x == y for x, y in zip(actual_v, expected_v)])
    
    actual_d, actual_v = gen_coord_combo(da1, drop_dims='c')
    expected_v = product([1,2,3], [11,22,33,44])
    expected_d = ['a', 'b']
    assert expected_d.sort() == actual_d.sort()
    # TODO: ensure dim ordering lines up
    assert all([x == y for x, y in zip(actual_v, expected_v)])
    
    
def test_gen_copy_ds():
    new_dvars = ['first', 'second']
    actual = gen_copy_ds(da1, new_dvars)
    
    assert all([x==y for x, y in zip(actual.dims, da1.dims)])
    assert all([x==y for x, y in zip(actual.coords, da1.coords)])
    assert all([x==y for x, y in zip(new_dvars, actual.data_vars)])
    for c in actual.coords:
        assert_equal(actual.coords[c], da1.coords[c])


def test_combine_new_ds_dim():
    expected = combine_new_ds_dim({9: da1, 99: da2}, "new")
    
    assert_equal(expected, da12)