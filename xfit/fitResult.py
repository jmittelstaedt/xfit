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


class fitResult:
    """
    Class containing the results of :func:`~.fit_dataset`.
    Given for convenience to give to plotting functions.

    Parameters
    ----------
    fit_ds : xarray.Dataset
        the dataset which resulted from fitting
    main_da : xarray.DataArray
        the data array which was fit over
    main_xda : xarray.DataArray or None
        the data array of x values which were fit over
    fit_func : function
        the function which was used to fit the data
    guess_func : function
        the function which was used to generate initial parameter
        guesses
    param_names : list of str
        names of the parameters of ``fit_func``, in the order it
        accepts them
    xname : str
        the name of the``dim`` of ``main_da`` which was fit over
    yname : str or None
        Name of the y data which was fit over
    yerr_da : xarray.dataArray or None
        The data array containing y error information for the data in ``main_da``

    Attributes
    ----------
    fit_ds : xarray.Dataset
        the dataset containing all of the found fit parameters and errors
    main_da : xarray.DataArray
        the data array which was fit over
    coords : xarray.coords
        ``coords`` of :attr:`~.fitResult.fit_ds`
    dims : xarray.dims
        ``dims`` of :attr:`~.fitResult.fit_ds`
    data_vars : xarray.data_vars
        ``data_vars`` of :attr:`~.fitResult.fit_ds`
    fit_func : function
        function which was fit
    guess_func : function
        function used for generating guesses
    param_names : list of str
        Names of parametrs fit to
    xname : str
        name of :attr:`~.fitResult.main_ds` ``dim``
        which was fit over
    yname : str or None
        the name of the y data which was fit over
    yerr_da : xarray.dataArray or None
        The data array containing y error information for the data in ``main_da``
    """

    def __init__(self, fit_ds, main_da, main_xda, fit_func, guess_func, param_names,
                 xname, yname=None, yerr_da=None):
        """
        Saves variables and extracts ``coords`` and ``dims`` for more convenient
        access.

        Parameters
        ----------
        fit_ds : xarray.Dataset
            the dataset which resulted from fitting
        main_da : xarray.DataArray
            the data array which was fit over
        main_xda : xarray.DataArray or None
            the data array of x values which were fit over
        fit_func : function
            the function which was used to fit the data
        guess_func : function
            the function which was used to generate initial parameter
            guesses
        param_names : list of str
            names of the parameters of ``fit_func``, in the order it
            accepts them
        xname : str
            the name of the``dim`` of ``main_da`` which was fit over
        yname : str or None
            the name of the y data which was fit over
        yerr_da : xarray.dataArray or None
            The data array containing y error information for the data in ``main_da``
        """

        self.fit_ds = fit_ds
        self.coords = self.fit_ds.coords
        self.dims = self.fit_ds.dims
        self.data_vars = self.fit_ds.data_vars
        # QUESTION: should we store parameter guesses?
        self.main_da = main_da
        self.main_xda = main_xda
        self.fit_func = fit_func
        self.guess_func = guess_func
        self.param_names = param_names
        self.xname = xname
        self.yname = yname
        self.yerr_da = yerr_da

    def __repr__(self):
        return super().__repr__()+'\n'+self.fit_ds.__repr__()

    def __getattr__(self, name):
        """
        Assume unknown attribute accesses are meant for fit dataset
        """
        return getattr(self.fit_ds, name)

    # TODO: implement selections, omissions, ranges
    def plot_fits(self, overlay_data = True, hide_large_errors = True,
                  pts_per_plot = 200, **kwargs):
        """
        Plots the results from fitting of
        :attr:`~baseAnalysis.fitResult.fit_ds`.

        Parameters
        ----------
        overlay_data : bool
            whether to overlay the actual data on top of the
            corresponding fit. Error bars applied if available.
        hide_large_errors : bool
            Whether to hide very large error bars
        pts_per_plot : int
            How many points to use for the ``fit_func`` domain.
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
        
        xlabel = self.main_xda.name if self.main_xda.name is not None else self.xname

        selections = {dim: coords for dim, coords in kwargs.items() if dim in self.fit_ds.dims}
        
        coord_combos = gen_coord_combo(self.fit_ds, selections=selections)

        # Determine which kwargs can be passed to plot
        if self.yerr_da is None:
            plot_argspec = getfullargspec(Line2D)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
        else:
            ebar_argspec = getfullargspec(plt.errorbar)
            plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}

        for combo in coord_combos:
            selection_dict = dict(zip(self.fit_ds.dims, combo))
            xselection_dict = {k: v for k, v in selection_dict.items() if k in self.main_xda.dims}
            selected_ds = self.fit_ds.sel(selection_dict)
            
            # Find the domain to fit over
            data_dom = self.main_xda.sel(xselection_dict).values.copy()
            fit_dom = np.linspace(data_dom.min(), data_dom.max(), pts_per_plot)

            # extract the fit parameters
            fit_params = [float(selected_ds[param].values) for param
                          in self.param_names]

            # don't plot if there is no meaningful data
            if np.all(np.isnan(fit_params)):
                continue

            # fit the function and plot
            fit_range = self.fit_func(fit_dom, *fit_params)
            # overlay data if requested
            if overlay_data:
                data_range = self.main_da.sel(selection_dict).values.copy()
                # plot errorbars if available
                if self.yerr_da is not None:
                    yerr = self.yerr_da.sel(selection_dict).values.copy()
                    num_pts = yerr.size
                    errlims = np.zeros(num_pts).astype(bool)
                    if hide_large_errors: # hide outliers if requested
                        data_avg = np.mean(data_range)
                        data_std = np.std(data_range)
                        for i, err in enumerate(yerr):
                            if err > 5*data_std:
                                yerr[i] = data_std*.5 # TODO: Find some better way of marking this
                                errlims[i] = True
                        for i, val in enumerate(data_range):
                            if np.abs(val - data_avg) > 5*data_std:
                                data_range[i] = data_avg
                                yerr[i] = data_std*0.5
                                errlims[i] = True
                    plt.errorbar(data_dom, data_range, yerr, lolims=errlims,
                                 uplims=errlims, **plot_kwargs)
                else:
                    plt.plot(data_dom, data_range, **plot_kwargs)

            plt.plot(fit_dom, fit_range)
            # add labels and make the title reflect the current selection
            plt.xlabel(xlabel)
            if self.yname is not None:
                plt.ylabel(self.yname)
            title_str = ''
            for item in selection_dict.items():
                title_str += '{}: {}, '.format(*item)
            plt.title(title_str[:-2]) # get rid of trailing comma and space
            plt.show()

class fitResultModel(fitResult):
    """
    represents fitting to a set of models
    """
    
    @staticmethod
    def from_fitResult(a, models):
        return fitResultModel(models, a.fit_ds, a.main_da, a.main_xda, a.fit_func,
                                a.guess_func, a.param_names, a.xname, a.yname,
                                a.yerr_da)
    
    def __init__(self, models, fit_ds, main_da, main_xda, fit_func, guess_func, param_names,
                 xname, yname=None, yerr_da=None):
        super().__init__(fit_ds, main_da, main_xda, fit_func, guess_func, param_names,
                 xname, yname, yerr_da)
        
        self.models = models
        self.modelsd = {m.name : m for m in models}
    
    # TODO: implement selections, omissions, ranges
    def plot_model_fits(self, plot_models='all', plot_total=True, background_models=[],
                            overlay_data=True, hide_large_errors=True,
                            pts_per_plot=200, show_legend=True, **kwargs):
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
            hide_large_errors : bool
                wheter to try to hide large errorbars
            pts_per_plot : int
                Number of points to use in the fit curves
            show_legend : bool
                whether to show the legend on the plot        
            """
            
            xlabel = self.main_xda.name if self.main_xda.name is not None else self.xname
            
            # check that all models are valid
            if plot_models == 'all':
                plot_models = list(x for x in self.modelsd.keys() if x not in background_models)
            elif isinstance(plot_models, str):
                plot_models = [plot_models]
            if isinstance(background_models, str):
                background_models = [background_models]
            for m in (plot_models+background_models):
                if m not in self.modelsd:
                    raise ValueError(f"{m} is not a model of this system! Included models are: {self.modelsd.keys()}")
            
            bg_models = [self.modelsd[mod] for mod in background_models]
            fg_models = [self.modelsd[mod] for mod in plot_models]

            
            selections = {dim: coords for dim, coords in kwargs.items() if dim in self.fit_ds.dims}
            
            coord_combos = gen_coord_combo(self.fit_ds, selections=selections)

            # Determine which kwargs can be passed to plot
            if self.yerr_da is None:
                plot_argspec = getfullargspec(Line2D)
                plot_kwargs = {k: v for k, v in kwargs.items() if k in plot_argspec.args}
            else:
                ebar_argspec = getfullargspec(plt.errorbar)
                plot_kwargs = {k: v for k, v in kwargs.items() if k in ebar_argspec.args}

            for combo in coord_combos:
                selection_dict = dict(zip(self.fit_ds.dims, combo))
                xselection_dict = {k: v for k, v in selection_dict.items() if k in self.main_xda.dims}
                selected_ds = self.fit_ds.sel(selection_dict)
                
                # extract selected parameters
                all_params = {pn : float(selected_ds[pn].values) for pn in self.param_names}
                
                data_dom = self.main_xda.sel(xselection_dict).values.copy()
                fit_dom = np.linspace(data_dom.min(), data_dom.max(), pts_per_plot)
                
                # don't plot if there is no meaningful data
                if np.all(np.isnan(list(all_params.values()))):
                    continue
                
                # generate the background offset
                background = np.zeros(fit_dom.size)
                for m in bg_models:
                    bgparams = [all_params[p.name] for p in m.params]
                    background += m(fit_dom, *bgparams)
                
                # overlay data if requested
                if overlay_data:
                    data_range = self.main_da.sel(selection_dict).values.copy()
                    # plot errorbars if available
                    if self.yerr_da is not None:
                        yerr = self.yerr_da.sel(selection_dict).values.copy()
                        num_pts = yerr.size
                        errlims = np.zeros(num_pts).astype(bool)
                        if hide_large_errors: # hide outliers if requested
                            data_avg = np.mean(data_range)
                            data_std = np.std(data_range)
                            for i, err in enumerate(yerr):
                                if err > 5*data_std:
                                    yerr[i] = data_std*.5 # TODO: Find some better way of marking this
                                    errlims[i] = True
                            for i, val in enumerate(data_range):
                                if np.abs(val - data_avg) > 5*data_std:
                                    data_range[i] = data_avg
                                    yerr[i] = data_std*0.5
                                    errlims[i] = True
                        plt.errorbar(data_dom, data_range, yerr, lolims=errlims,
                                    uplims=errlims, **plot_kwargs)
                    else:
                        plt.plot(data_dom, data_range, **plot_kwargs)
                
                # plot the full model, if requested
                if plot_total:
                    modparams = [all_params[pn] for pn in self.param_names]
                    plt.plot(fit_dom, self.fit_func(fit_dom, *modparams), label='total')
                
                # plot the models
                for m in fg_models:
                    modparams = [all_params[p.name] for p in m.params]
                    plt.plot(fit_dom, m(fit_dom, *modparams) + background, label=m.name)
                
                # add labels and make the title reflect the current selection
                plt.xlabel(xlabel)
                if self.yname is not None:
                    plt.ylabel(self.yname)
                title_str = ''
                for item in selection_dict.items():
                    title_str += '{}: {}, '.format(*item)
                plt.title(title_str[:-2]) # get rid of trailing comma and space
                
                # add legend if requested
                if show_legend:
                    plt.legend()
                
                plt.show()
