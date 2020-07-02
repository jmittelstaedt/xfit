# XFit: Curve Fitting in Xarray

[Xarray](https://xarray.pydata.org/en/stable/) is a package providing labelled
N-dimensional arrays. 
This makes it convenient for storing data in a variety of application domains. 
Xfit attempts to provide an easy way to fit data contained in an xarray object
to an arbitrary nonlinear function using [scipy's curve fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).

# Usage

We'll briefly describe the two main fitting methods here.

## Fitting DataArray's

We first import the necessary packages and make some dummy data

```python
import numpy as np
import xarray as xr
from xfit import *

# define slopes, intercepts and independent variable coordinates
ms = xr.DataArray(np.arange(6), coords = {'m_true': np.arange(6)}, dims='m_true')
bs = xr.DataArray(np.linspace(0,10,6), coords={'b_true': np.linspace(0,10,6)}, dims='b_true')
xs = xr.DataArray(np.linspace(0,20,101), coords={'x': np.linspace(0,20,101)}, dims='x')

# create data
data_da = xs*ms + bs
data_da.values += 5*np.random.rand(101, 6, 6) # adding some noise
data_da.name = 'data'
```

We now have a `DataArray` with three dimensions: one each for slope 
and intercept and one for the independent variable. 
For each slope and intercept, the data is a line along `x` with the 
slope and intercept described by the coordinate. 
To fit this, we will need three things: the functional form which we
want to fit, a function to generate guesses of the parameters, and 
a list of the names of the various parameters:

```python
def linfunc(x, m, b):
    return x*m + b

def linfunc_guess(x, y, **kwargs):
    m_guess = (y[-1] - y[0])/(x[-1] - x[0])
    b_guess = y[0] - m_guess*x[0]
    return m_guess, b_guess

param_names = ['m', 'b']
```

With these defined, we can now perform the actual fitting, which 
will be done with the `fit_dataArray` function:

```python
fit = fit_dataArray(
    data_da,        # data to fit
    linfunc,        # function to fit to
    linfunc_guess,  # function for making guesses
    lin_params,     # parameter names
    'x'             # dimension to fit over
)
```

`fit` is now a `fitResult` object, containing the results of fitting our data.
The main results are contained in `fit.fit_ds`, which is a `Dataset` with two 
`DataArray`'s per parameter: one containing the value of the parameter and
another containing its error.
The dimensions and coordinates of this new `Dataset` are the same as the `data_da`
`DataArray`, excluding the dimension that we fit over.
This object has other features, but we will talk about them elsewhere.

## Fitting Datasets

The process for fitting `Dataset`'s is quite similar. 
Assuming we have the same setup as above, we make a `Dataset` using the `DataArray`
from before, and fit it:

```python
data_ds = xr.Dataset(data_vars={'lindata': data_da})

fit2 = fit_dataset(
    data_ds,        # Dataset containing the data to fit
    linfunc,        # Function to fit
    linfunc_guess,  # Function generating initial parameter guesses
    lin_params,     # Parameter names
    'x',            # Dimension name
    'lindata'       # Data variable name containing the dataArray to fit
)
```

and now `fit2` is the same kind of object as `fit`, and should actually
be identical in this simple example.

# Installation

For now, this package is not on pypi so you will need to clone and install
locally for now.
The dependencies are:
- xarray
- numpy
- scipy
- pyplot
- python > 3.6