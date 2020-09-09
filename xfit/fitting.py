from inspect import getfullargspec
from typing import (
    TYPE_CHECKING,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Hashable
)


import numpy as np
import xarray as xr
from scipy.optimize import leastsq, curve_fit


from .utils import da_filter, gen_coord_combo, gen_copy_ds
from .models import fitModel
from .fitResult import fitResult, fitResultModel


if TYPE_CHECKING:
    from xarray import DataArray, Dataset


def make_fit_dataArray_guesses(
    yda: 'DataArray', 
    guess_func: Callable[[Sequence[float], Sequence[float]], Sequence[float]], 
    param_names: Sequence[str], 
    xname: str, 
    xda: Optional['DataArray'] = None, 
    guess_func_help_params: Mapping[str, float] = {}
    ):
    """
    creates a dataset of guesses of param_names given ``guess_func``. To be used
    in :func:`~.fit_dataArray`.

    Parameters
    ----------
    yda : xarray.DataArray
        data array containing data to fit to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - numpy array of x data
        - 1D numpy array of y data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        All arguments must be accepted, not all must be used.
        As a hint, if designing for unexpected dims you can include **kwargs at
        the end. This will accept any keword arguments not explicitly defined
        by you and just do nothing with them.
        Must return a list of guesses to the parameters, in the order given in
        param_names
    param_names : list of str
        list of fit parameter names, in the order they are returned
        by ``guess_func``
    xname : str
        the name of the  ``dim`` of ``da`` to be fit along
    guess_func_help_params : dict
        Dictionary of any "constant" parameters to help in generating fit
        guesses. Passed as keyword arguments to ``guess_func``.

    Returns
    -------
    xarray.Dataset
        A Dataset with param_names as data_vars containing all guesses, and all
        ``dims`` of ``ds`` besides xname with the same coordinates, unless
        otherwise specified in ``**selections``.
    """

    xdims = xda.dims

    combo_dims, coord_combos = gen_coord_combo(yda, [xname])
    
    # Generate empty dataset to contain parameter guesses
    guess_ds = gen_copy_ds(yda, param_names, [xname])

    # apply guess_func for each coord combo and record param guesses
    for combo in coord_combos:
        selection_dict = dict(zip(combo_dims, combo))
        xselection_dict = {k: v for k, v in selection_dict.items() if k in xdims}

        # load x/y data for this coordinate combination
        ydata = yda.sel(selection_dict).values
        xdata = xda.sel(xselection_dict).values

        # Deal with any possible spurious data
        if np.all(np.isnan(ydata)):
            # there is no meaningful data. Fill guesses with nan's
            for i, pname in enumerate(param_names):
                guess_ds[pname].loc[selection_dict] = np.nan
            continue
        else:
            # remove bad datapoints
            good_pts = np.logical_and(np.isfinite(ydata), np.isfinite(xdata))
            xdata = xdata[good_pts]
            ydata = ydata[good_pts]

        # generate guesses
        guesses = guess_func(xdata, ydata, **guess_func_help_params, **selection_dict)

        # record fit parameters and their errors
        for i, pname in enumerate(param_names):
            guess_ds[pname].loc[selection_dict] = guesses[i]

    return guess_ds

# TODO: make another using leastsq? (to allow for broader cost functions)
def fit_dataArray(
    yda: 'DataArray', 
    fit_func: Callable[[Sequence[float], float], Sequence[float]], 
    guess_func: Callable[[Sequence[float], Sequence[float]], Sequence[float]], 
    param_names: Sequence[str], 
    xname: str, 
    xda: Optional['DataArray'] = None, 
    yname: Optional[str] = None,
    yerr_da: Optional['DataArray'] = None,
    bootstrap_samples=0, # TODO: remove bootstraping
    guess_func_help_params: Mapping[str, float] = {}, 
    ignore_faliure: bool = False,
    selections: Mapping[str, Union[Hashable, Sequence[Hashable]]] = {}, 
    omissions: Mapping[str, Union[Hashable, Sequence[Hashable]]] = {}, 
    ranges: Mapping[str, Tuple[float, float]] = {},
    **kwargs) -> fitResult:
    """
    Fits values in a data array to a function. Returns an
    :class:`~.fitResult` object

    Parameters
    ----------
    yda : xarray.DataArray
        Dataset containing data to be fit.
    fit_func : function
        function to fit data to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - numpy array of x data
        - 1D numpy array of y data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        As a hint, if designing for unexpected dims you can include ``**kwargs``
        at the end. This will accept any keword arguments not explicitly defined
        by you and just do nothing with them.
        All arguments must be accepted, not all must be used. fit results
        Must return a list of guesses to the parameters, in the order given in
        ``param_names``
    param_names : list of str
        list of fit parameter names, in the order they are returned
        by guess_func
    xname : str
        the name of the ``dim`` of ``da`` to be fit along
    xda : xarray.DataArray or None
        If given, dataArray containing data to use as the dependent variable
        in the fits. Must include ``xname`` among its dims and have a 
        subset of the coords of ``yda``
    yname : str or None
        Optional. The name of the y data being fit over.
    yerr_da : xarray.DataArray or None
        Optional. If provided, must be a data array containing errors in the
        data contained in ``da``. Must have the same coordinates as ``da``
    guess_func_help_params : dict
        Dictionary of any "constant" parameters to help in generating fit
        guesses. Passed as keyword arguments to ``guess_func``.
    ignore_faliure : bool
        if True, will ignore a ``RuntimeError`` from curve_fit that the
        optimal parameters were not found, and will fill with nan and print
        a message
    selections : dict
        Key is dimension name, values are coordinate values which we want to keep
    omissions : dict
        Key is dimension name, values are coordinate values which we want to omit
    ranges : dict
        Key is dimension name, values are a 2-iterable of a lower and upper limit
        (inclusive) which we want to keep.
    **kwargs
        can be:
        - names of ``dims`` of ``da``
        values should eitherbe single coordinate values or lists of coordinate
        values of those ``dims``. Only data with coordinates given by selections
        are fit to . If no selections given, everything is fit to.
        - kwargs of ``curve_fit``

    Returns
    -------
    fitResult
        Object containing all results of fitting and some convenience methods.
    """
    
    selections = {} if selections is None else selections
    omissions = {} if omissions is None else omissions
    ranges = {} if ranges is None else ranges

    if xda is None:
        xda = yda.coords[xname]

    # Account for selections in keyword arguments
    for kw in kwargs:
        if kw in yda.dims:
            selections[kw] = kwargs[kw]
    
    xselections = {dim: sel for dim, sel in selections.items() if dim in xda.dims}
    xomissions = {dim: sel for dim, sel in omissions.items() if dim in xda.dims}
    xranges = {dim: sel for dim, sel in ranges.items() if dim in xda.dims}
    
    xda = da_filter(xda, selections=xselections, omissions=xomissions, ranges=xranges)
    yda = da_filter(yda, selections=selections, omissions=omissions, ranges=ranges)
    if yerr_da is not None:
        yerr_da = da_filter(yerr_da, selections=selections, omissions=omissions, ranges=ranges)
    guesses = make_fit_dataArray_guesses(
        yda,
        guess_func,
        param_names,
        xname,
        xda,
        guess_func_help_params
    )

    # Determine which kwargs can be passed to curve_fit
    cf_argspec = getfullargspec(curve_fit)
    lsq_argspec = getfullargspec(leastsq)
    good_args = cf_argspec.args + lsq_argspec.args
    cf_kwargs = {k: v for k, v in kwargs.items() if k in good_args}

    full_param_names = param_names + [pname+'_err' for pname in param_names]

    # Get the selection and empty fit dataset
    fit_ds = gen_copy_ds(yda, full_param_names, [xname])
    combo_dims, coord_combos = gen_coord_combo(yda, [xname])

    # Do the fitting
    for combo in coord_combos:
        selection_dict = dict(zip(combo_dims, combo))
        xselection_dict = {k: v for k,v in selection_dict.items() if k in xda.dims}

        # load x/y data for this coordinate combination
        ydata = yda.sel(selection_dict).values
        xdata = xda.sel(xselection_dict).values
        if yerr_da is not None:
            yerr = yerr_da.sel(selection_dict).values
        else:
            yerr = None

        # load fit parameter guesses for this coordinate combination
        guess = [float(guesses[pname].sel(selection_dict).values) for pname in param_names]
        guess = np.array(guess)

        # Deal with any possible spurious data
        if np.all(np.isnan(ydata)):
            # there is no meaningful data. Fill fit results with nan's
            for i, pname in enumerate(param_names):
                fit_ds[pname].loc[selection_dict] = np.nan
                fit_ds[pname+'_err'].loc[selection_dict] = np.nan
            continue
        else:
            # remove bad datapoints
            good_pts = np.logical_and(np.isfinite(ydata), np.isfinite(xdata))
            if yerr_da is not None:
                good_pts = np.logical_and(good_pts, np.isfinite(yerr))
                yerr = yerr[good_pts]
            xdata = xdata[good_pts]
            ydata = ydata[good_pts]

        if ydata.size < len(param_names):
            print("Less finite datapoints than parameters at : ", selection_dict)
            for i, pname in enumerate(param_names):
                fit_ds[pname].loc[selection_dict] = np.nan
                fit_ds[pname+'_err'].loc[selection_dict] = np.nan
            continue

        # fit
        try:
            asig = True if yerr_da is not None else False
            popt, pcov = curve_fit(fit_func, xdata, ydata, guess, yerr, absolute_sigma=asig, **cf_kwargs)
            perr = np.sqrt(np.diag(pcov)) # from curve_fit documentation
        except RuntimeError:
            if ignore_faliure:
                popt = np.ones(len(param_names))*np.nan
                perr = np.ones(len(param_names))*np.nan
            else:
                raise

        if bootstrap_samples > 0 and not np.all(np.isnan(popt)): # TODO: figure out a better way of handling points with large errors
            fit_ys = fit_func(xdata, *popt)
            residuals = ydata - fit_ys

            if yerr_da is not None:
                ys_weights = 1/yerr**2
                ys_weights = ys_weights / ys_weights.sum()
            else:
                ys_weights = None
            popts = []
            for _ in range(bootstrap_samples):
                # draw with replacement, weighting each residual by 1/error**2 so points
                # with large errors are suppressed
                residual_sample = np.random.choice(residuals, size=ydata.size, replace=True, p=ys_weights)
                y_sample = fit_ys + residual_sample
                popt_samp, _ = curve_fit(fit_func, xdata, y_sample, popt, **cf_kwargs)
                popts.append(popt_samp)
            popts = np.array(popts)
            popt = popts.mean(0)
            perr = popts.std(0)

        # record fit parameters and their errors
        for i, pname in enumerate(param_names):
            fit_ds[pname].loc[selection_dict] = popt[i]
            fit_ds[pname+'_err'].loc[selection_dict] = perr[i]

    return fitResult(
        fit_ds,
        yda,
        xda,
        fit_func,
        guess_func,
        param_names,
        xname,
        yname,
        yerr_da
    )


def fit_dataset(
    ds: 'Dataset', 
    fit_func: Callable[[Sequence[float], float], Sequence[float]], 
    guess_func: Callable[[Sequence[float], Sequence[float]], Sequence[float]], 
    param_names: Sequence[str], 
    xname: str,
    yname: str,
    xda_name: Optional[str] = None,
    yerr_name: Optional[str] = None,
    bootstrap_samples=0, # TODO: remove bootstraping
    guess_func_help_params: Mapping[str, float] = {}, 
    ignore_faliure: bool = False,
    selections: Mapping[str, Union[Hashable, Sequence[Hashable]]] = {}, 
    omissions: Mapping[str, Union[Hashable, Sequence[Hashable]]] = {}, 
    ranges: Mapping[str, Tuple[float, float]] = {},
    **kwargs
    ) -> fitResult:
    """
    Fits values in a dataset to a function. Returns an
    :class:`~.fitResult` object.

    Convenience function which calls :func:`~.fit_dataArray`.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing data to be fit.
    fit_func : function
        function to fit data to
    guess_func : function
        function to generate guesses of parameters. Arguments must be:
        - numpy array of x data
        - 1D numpy array of y data
        - keyword arguments, with the keywords being all dims of ds besides
        xname. Values passed will be individual floats of coordinate values
        corresponding to those dims.
        As a hint, if designing for unexpected dims you can include **kwargs at
        the end. This will accept any keword arguments not explicitly defined
        by you and just do nothing with them.
        All arguments must be accepted, not all must be used.fit results
        Must return a list of guesses to the parameters, in the order given in
        ``param_names``
    param_names : list of str
        list of fit parameter names, in the order they are returned
        by guess_func
    xname : str
        the name of the ``dim`` of ``ds`` to be fit along
    yname : str
        the name of the  containing data to be fit to
    xda_name : str
        Name of the data variable which contains the x data
    yerr_name : str
        Optional. the name of the ``data_var`` of ``ds`` containing errors in data
        to be fit.
    bootstrap_samples : int
        Number of boostrap samples to draw to get beter statistics on the
        parameters and their errors. If zero no boostrap resampling is done
    guess_func_help_params : dict
        Dictionary of any "constant" parameters to help in generating fit
        guesses. Passed as keyword arguments to ``guess_func``.
    ignore_faliure : bool
        if True, will ignore a ``RuntimeError`` from curve_fit that the
        optimal parameters were not found, and will fill with nan and print
        a message
    selections : dict
        Key is dimension name, values are coordinate values which we want to keep
    omissions : dict
        Key is dimension name, values are coordinate values which we want to omit
    ranges : dict
        Key is dimension name, values are a 2-iterable of a lower and upper limit
        (inclusive) which we want to keep.
    **kwargs
        can be:
        - names of ``dims`` of ``da``
        values should eitherbe single coordinate values or lists of coordinate
        values of those ``dims``. Only data with coordinates given by selections
        are fit to . If no selections given, everything is fit to.
        - kwargs of ``curve_fit``

    Returns
    -------
    fitResult
        Result of the fitting
    """
    
    yerr_da = None if yerr_name is None else ds[yerr_name]
    
    xda = None if xda_name is None else ds[xda_name]

    fit_da = ds[yname]

    return fit_dataArray(fit_da, fit_func, guess_func, param_names, xname, xda,
                         yname, yerr_da, bootstrap_samples,
                         guess_func_help_params, ignore_faliure, selections,
                         omissions, ranges, **kwargs)


def fit_dataArray_models(
    da: 'DataArray', 
    models: Union[fitModel, Sequence[fitModel]], 
    xname: str, # TODO: Explicitly add other arguments
    **kwargs):
    """
    fits a dataArray to a collection of models
    
    Parameters
    ----------
    da : xr.DataArray
        DataArray to fit over
    models : fitModel or list of fitModel
        Models to use in fit. Will be combined additively
    xname : str
        dimension to fit over
    """
    
    if isinstance(models, fitModel):
        models = [models]
    
    # make list of all unique parameters, with intersection of bounds if same 
    # parameter in multiple models
    all_params = []
    for m in models:
        for param in m.params:
            for p in all_params:
                if p == param:
                    p.intersect(param)
                    break
            else: # if not found in all_params
                all_params.append(param)

    def full_func(x, *args):
        cumsum = 0
        for m in models:
            mparams = [args[all_params.index(p)] for p in m.params]
            cumsum += m(x, *mparams)
        return cumsum

    # mean of estimate from each model TODO: should we do median instead?
    def full_guess(x, y, **kwargs):
        guesses = []
        
        # generate parameter guesses from each model
        for m in models:
            model_guesses = np.atleast_1d(m.guess(x, y, **kwargs))
            guesses.append(dict(zip([p.name for p in m.params], model_guesses)))
        
        # take the mean of all guesses for each parameter
        final_guesses = []
        for param in all_params:
            count = 0
            cumsum = 0
            for guess in guesses:
                if param.name in guess:
                    count += 1
                    cumsum += guess[param.name]
                else:
                    pass
            final_guesses.append(cumsum/count)
            
        return final_guesses
    
    bounds = [[], []]
    for p in all_params:
        bounds[0].append(p.bounds[0])
        bounds[1].append(p.bounds[1])
    
    return fitResultModel.from_fitResult(
        fit_dataArray(da, full_func, full_guess, [p.name for p in all_params], xname, bounds=tuple(bounds), **kwargs),
         models
        )


def fit_dataset_models(ds, models, xname, yname, yerr_name=None, **kwargs):
    """
    Fits a dataset to a collection of models, combined additively
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing data to fit
    models : list of fitModel
        Models to fit to
    xname : str
        Name of coordinate to use as the independent variable
    yname : str
        name of dataArray to use as the dependent variable
    yerr_name : str
        name of dataArray to use as the errors in the dependent variable
    """
    
    if yerr_name is not None:
        if yerr_name not in ds.data_vars:
            raise AttributeError('%s is not a data_var!'%yerr_name)
        else:
            yerr_da = ds[yerr_name]
    else:
        yerr_da=None

    fit_da = ds[yname]

    return fit_dataArray_models(fit_da, models, xname,
                         yname=yname, yerr_da=yerr_da, **kwargs)
