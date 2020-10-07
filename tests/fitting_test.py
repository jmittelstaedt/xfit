from xfit.fitting import fit_dataArray, fit_dataArray_models

import numpy as np
import xarray as xr

from xarray.testing import assert_equal, assert_allclose

bs = xr.DataArray(np.linspace(0,10,6), coords={'b_true': np.linspace(0,10,6)}, dims='b_true')
xs = xr.DataArray(np.linspace(0,10,11), coords={'x': np.linspace(0,10,11)}, dims='x')

data = xr.DataArray(
    np.ones((1,10))*np.arange(5).reshape(5,1), 
    dims=['b_true','x'], 
    coords={
        'b_true': np.arange(5),
        'x': np.arange(10)
    }
    )

expected_popt = xr.DataArray(np.arange(5).reshape(1,5), 
                             coords={'param': ['b'], 'b_true': np.arange(5)},
                             dims=['param', 'b_true'])

expected_perr = xr.DataArray(np.zeros((1,5)), 
                             coords={'param': ['b'], 'b_true': np.arange(5)},
                             dims=['param', 'b_true'])

expected_pcov = xr.DataArray(np.zeros((1,1,5)), 
                             coords={'param_cov': ['b'], 'param': ['b'], 'b_true': np.arange(5)},
                             dims=['param_cov', 'param', 'b_true'])

expected_xda = data.coords['x']
expected_yda = data
expected_yerr_da = xr.full_like(expected_yda, np.nan, float)

def const(x, b):
    return b

def const_guess(x, y, **kwargs):
    return np.mean(y)

const_params = ['b']

def test_basic_fit_dataArray():
    actual = fit_dataArray(data, const, const_guess, const_params, 'x')
    
    expected = xr.Dataset(
        {
            'popt': expected_popt,
            'perr': expected_perr,
            'pcov': expected_pcov,
            'xda': expected_xda,
            'yda': expected_yda,
            'yerr_da': expected_yerr_da
        },
        attrs={'fit_func': const, 'param_names': ['b'], 'xname': 'x', 'yname': None})
    
    assert_equal(expected, actual)