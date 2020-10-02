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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from .utils import da_filter, gen_coord_combo


if TYPE_CHECKING:
    from xarray import DataArray, Dataset


def plot_fits(fit_ds, overlay_data = True, pts_per_plot = 200,
              selections=None, omissions=None, ranges=None, **kwargs):
    """
    Plots the results from fitting of `fit_ds`

    Parameters
    ----------
    fit_ds : xr.Dataset
        Dataset which has been created from a fitting method. Necessary so that
        the expected attributes and data variables.
    overlay_data : bool
        whether to overlay the actual data on top of the
        corresponding fit. Error bars applied if available.
    pts_per_plot : int
        How many points to use for the ``fit_func`` domain.
    selections : dict
        Key is dimension name, values are coordinate values which we want to keep
    omissions : dict
        Key is dimension name, values are coordinate values which we want to omit
    ranges : dict
        Key is dimension name, values are a 2-iterable of a lower and upper limit
        (inclusive) which we want to keep.
    **kwargs
        Can either be:
        - names of ``dims`` of ``fit_ds``. values should eitherbe single coordinate
        values or lists of coordinate values of those ``dims``. Only data with
        coordinates given by selections are plotted. If no selections given,
        everything is plotted.
        - kwargs passed to ``plot`` or ``errorbar``, as appropriate

    Returns
    -------
    None
        Just plots the requested fits.
    """

    xlabel = fit_ds.xda.name if fit_ds.xda.name is not None else fit_ds.attrs['xname']

    selections = {} if selections is None else selections
    omissions = {} if omissions is None else omissions
    ranges = {} if ranges is None else ranges
    
    for kw in kwargs:
        if kw in fit_ds.yda.dims:
            selections[kw] = kwargs[kw]
   
    xselections = {dim: sel for dim, sel in selections.items() if dim in fit_ds.xda.dims}
    xomissions = {dim: sel for dim, sel in omissions.items() if dim in fit_ds.xda.dims}
    xranges = {dim: sel for dim, sel in ranges.items() if dim in fit_ds.xda.dims}

    yselections = {dim: sel for dim, sel in selections.items() if dim in fit_ds.yda.dims}
    yomissions = {dim: sel for dim, sel in omissions.items() if dim in fit_ds.yda.dims}
    yranges = {dim: sel for dim, sel in ranges.items() if dim in fit_ds.yda.dims}

    pselections = {dim: sel for dim, sel in selections.items() if dim in fit_ds.popt.dims and dim != "param"}
    pomissions = {dim: sel for dim, sel in omissions.items() if dim in fit_ds.popt.dims and dim != "param"}
    pranges = {dim: sel for dim, sel in ranges.items() if dim in fit_ds.popt.dims and dim != "param"}

    xda = da_filter(fit_ds.xda, selections=xselections, omissions=xomissions, ranges=xranges)
    yda = da_filter(fit_ds.yda, selections=yselections, omissions=yomissions, ranges=yranges)
    pda = da_filter(fit_ds.popt, selections=pselections, omissions=pomissions, ranges=pranges)

    if not np.all(np.isnan(fit_ds.yerr_da)):
        yeda = da_filter(fit_ds.yerr_da, selections=yselections, omissions=yomissions, ranges=yranges)
    else:
        yeda = None

    combo_dims, coord_combos = gen_coord_combo(yda, fit_ds.attrs['xname'])
    
    # dimensions to show in the title
    dims_with_many_values = [dim for dim in pda.dims if len(pda[dim])>1 and dim != "param"]

    # Determine which kwargs can be passed to plot
    if np.all(np.isnan(fit_ds.yerr_da)):
        plot_argspec = getfullargspec(Line2D)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
    else:
        ebar_argspec = getfullargspec(plt.errorbar)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}

    for combo in coord_combos:
        selection_dict = dict(zip(combo_dims, combo))
        xselection_dict = {k: v for k, v in selection_dict.items() if k in xda.dims}

        data_dom = xda.sel(xselection_dict).values.copy()
        fit_dom = np.linspace(data_dom.min(), data_dom.max(), pts_per_plot)

        fit_params = pda.sel(selection_dict).values
        if np.all(np.isnan(fit_params)):
            continue

        fit_range = fit_ds.attrs['fit_func'](fit_dom, *fit_params)

        if overlay_data:
            data_range = yda.sel(selection_dict).values.copy()
            if yeda is not None:
                yerr = yeda.sel(selection_dict).values.copy()
                plt.errorbar(data_dom, data_range, yerr, **plot_kwargs)
            else:
                plt.plot(data_dom, data_range, **plot_kwargs)

        plt.plot(fit_dom, fit_range)
        # add labels and make the title reflect the current selection
        plt.xlabel(xlabel)
        if fit_ds.attrs['yname'] is not None:
            plt.ylabel(fit_ds.attrs['yname'])
        title_str = ''
        for dim in dims_with_many_values:
            title_str += f'{dim}: {selection_dict[dim]}, '
        try:
            plt.title(title_str[:-2])
        except:
            plt.title('')
        plt.show()

def plot_model_fits(fit_ds, plot_models='all', plot_total=True, background_models=[],
                    overlay_data=True, pts_per_plot=200, show_legend=True, 
                    selections=None, omissions=None, ranges=None, **kwargs):
    """
    Plots individual models
    
    Parameters
    ----------
    plot_models : str or list of str
        names of models to plot
    plot_total : bool
        wheter to also plot the sum of all of the models
    background_models : str or list of str
        models to use as a background, being added to each of the individual
        models to be plotted
    overlay_data : bool
        whether to overlay the raw data
    pts_per_plot : int
        Number of points to use in the fit curves
    show_legend : bool
        whether to show the legend on the plot        
    """
    
    xlabel = fit_ds.xda.name if fit_ds.xda.name is not None else fit_ds.attrs['xname']
    
    # check that all models are valid
    if plot_models == 'all':
        plot_models = list(x for x in fit_ds.attrs['models'].keys() if x not in background_models)
    elif isinstance(plot_models, str):
        plot_models = [plot_models]
    if isinstance(background_models, str):
        background_models = [background_models]
    for m in (plot_models+background_models):
        if m not in fit_ds.attrs['models']:
            raise ValueError(f"{m} is not a model of this system! Included models are: {fit_ds.attrs['models'].keys()}")
    
    bg_models = [fit_ds.attrs['models'][mod] for mod in background_models]
    fg_models = [fit_ds.attrs['models'][mod] for mod in plot_models]

    selections = {} if selections is None else selections
    omissions = {} if omissions is None else omissions
    ranges = {} if ranges is None else ranges
    
    for kw in kwargs:
        if kw in fit_ds.yda.dims:
            selections[kw] = kwargs[kw]
   
    xselections = {dim: sel for dim, sel in selections.items() if dim in fit_ds.xda.dims}
    xomissions = {dim: sel for dim, sel in omissions.items() if dim in fit_ds.xda.dims}
    xranges = {dim: sel for dim, sel in ranges.items() if dim in fit_ds.xda.dims}

    yselections = {dim: sel for dim, sel in selections.items() if dim in fit_ds.yda.dims}
    yomissions = {dim: sel for dim, sel in omissions.items() if dim in fit_ds.yda.dims}
    yranges = {dim: sel for dim, sel in ranges.items() if dim in fit_ds.yda.dims}

    pselections = {dim: sel for dim, sel in selections.items() if dim in fit_ds.popt.dims and dim != "param"}
    pomissions = {dim: sel for dim, sel in omissions.items() if dim in fit_ds.popt.dims and dim != "param"}
    pranges = {dim: sel for dim, sel in ranges.items() if dim in fit_ds.popt.dims and dim != "param"}

    xda = da_filter(fit_ds.xda, selections=xselections, omissions=xomissions, ranges=xranges)
    yda = da_filter(fit_ds.yda, selections=yselections, omissions=yomissions, ranges=yranges)
    pda = da_filter(fit_ds.popt, selections=pselections, omissions=pomissions, ranges=pranges)

    if not np.all(np.isnan(fit_ds.yerr_da)):
        yeda = da_filter(fit_ds.yerr_da, selections=yselections, omissions=yomissions, ranges=yranges)
    else:
        yeda = None

    combo_dims, coord_combos = gen_coord_combo(yda, fit_ds.attrs['xname'])
    
    # dimensions to show in the title
    dims_with_many_values = [dim for dim in pda.dims if len(pda[dim])>1 and dim != "param"]

    # Determine which kwargs can be passed to plot
    if np.all(np.isnan(fit_ds.yerr_da)):
        plot_argspec = getfullargspec(Line2D)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
    else:
        ebar_argspec = getfullargspec(plt.errorbar)
        plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}

    for combo in coord_combos:
        selection_dict = dict(zip(combo_dims, combo))
        xselection_dict = {k: v for k, v in selection_dict.items() if k in xda.dims}

        all_params = {pn : float(pda.sel(**selection_dict, param=pn).values) for pn in fit_ds.attrs['param_names']}
        
        data_dom = xda.sel(xselection_dict).values.copy()
        fit_dom = np.linspace(data_dom.min(), data_dom.max(), pts_per_plot)
        
        if np.all(np.isnan(list(all_params.values()))):
            continue
        
        background = np.zeros(fit_dom.size)
        for m in bg_models:
            bgparams = [all_params[p.name] for p in m.params]
            background += m(fit_dom, *bgparams)
        
        if overlay_data:
            data_range = yda.sel(selection_dict).values.copy()
            if yeda is not None:
                yerr = yeda.sel(selection_dict).values.copy()
                plt.errorbar(data_dom, data_range, yerr, **plot_kwargs)
            else:
                plt.plot(data_dom, data_range, **plot_kwargs)
        
        if plot_total:
            modparams = pda.sel(selection_dict).values
            plt.plot(fit_dom, fit_ds.attrs['fit_func'](fit_dom, *modparams), label='total')
        
        for m in fg_models:
            modparams = [all_params[p.name] for p in m.params]
            plt.plot(fit_dom, m(fit_dom, *modparams) + background, label=m.name)
        
        # add labels and make the title reflect the current selection
        plt.xlabel(xlabel)
        if fit_ds.attrs['yname'] is not None:
            plt.ylabel(fit_ds.attrs['yname'])
            
        title_str = ''
        for dim in dims_with_many_values:
            title_str += f'{dim}: {selection_dict[dim]}, '
        try:
            plt.title(title_str[:-2]) # get rid of trailing comma and space
        except:
            plt.title('')
        
        # add legend if requested
        if show_legend:
            plt.legend()
        
        plt.show()