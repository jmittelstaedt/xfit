from typing import (
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


import numpy as np

class fitParameter:
    """
    parameter used in fitting
    
    Attributes
    ----------
    name : str
        name of the parameter
    bounds : 2-tuple of float
        a tuple of lower and upper bounds on the parameter
    """


    def __init__(
        self, 
        name: str, 
        bounds: Tuple[float, float] = (-np.inf, np.inf)
        ):
        
        self.name = name
        if bounds[0] > bounds[1]:
            raise ValueError("Lower bound must be less than upper bound!")
        self.bounds = bounds


    def __eq__(self, other: 'fitParameter'):
        """ equality determined by name only. maybe bad... idk """
        return self.name == other.name


    def __repr__(self):
        return f"<fitParameter {self.name}, {self.bounds}>"


    # TODO: what to do if bounds don't intersect?
    def intersect(self, other: 'fitParameter') -> 'fitParameter':
        """
        Returns a new fitParameter with bounds which are the intersection
        of the initial ones. Names must be the same, and will be the name of the
        result.
        """
        if self.name == other.name:
            return fitParameter(
                self.name,
                (
                    max(self.bounds[0], other.bounds[0]),
                    min(self.bounds[1], other.bounds[1])
                )
            )


class fitModel:
    """
    Model to fit to
    
    Attributes
    ----------
    name : str
        name of the model
    func : callable
        function which represents the model. First argument must be the 
        dependent variable, and the rest the model parameters
    guess : callable
        function which generates parameter guesses. If not given, guesses
        are all 1. Must take an array for x and y values, and allow for keyword
        arguments which will be dimension names where coordinate values for that
        dimension will be passed.
    params : list of str or fitParameter
        Parameters of the model, in the order accepted by the model function
    bounds : tuple
        bounds for parameters, in the order accepted by model function. As would
        be passed to ``scipy.optimize.curve_fit``. Only used if fit parameters
        do not already have bounds.
    """
    
    def __init__(
        self, 
        name: str, 
        func: Callable[[Sequence[float], float], Sequence[float]], 
        params: Sequence[Union[fitParameter, str]], 
        guess: Optional[Callable[[Sequence[float], Sequence[float]], Sequence[float]]] = None, 
        bounds: Union[Tuple[float, float], Sequence[Tuple[float, float]]] = (-np.inf, np.inf)
        ):
        
        self.name = name
        self.func = func
        
        # create parameter list
        if isinstance(params[0], fitParameter):
            self.params = params
        elif isinstance(params[0], str):
            self.params = []
            # make proper bounds
            for i, param in enumerate(params):
                try:
                    lobound = bounds[0][i]
                except TypeError:
                    lobound = bounds[0]
                try:
                    upbound = bounds[1][i]
                except TypeError:
                    upbound = bounds[1]
                self.params.append(fitParameter(param, (lobound, upbound)))
        else:
            raise ValueError("Must have good values for the parameters")

        # bounds tuple
        self.bounds = ( tuple(x.bounds[0] for x in self.params),
                        tuple(x.bounds[1] for x in self.params) )

        if guess is not None:
            self.guess = guess
        else:
            def gf(*args, **kwargs):
                return (1,)*len(self.params)
            self.guess = gf


    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


    def __repr__(self):
        
        rstr = f"<fitModel {self.name}>\nParameters:"
        for p in self.params:
            rstr += '\n  '+str(p)
        
        return rstr+'\n'


    def rename_params(self, rename_dict: Mapping[str, str]) -> 'fitModel':
        """
        Returns a new fitModel with different parameter names
        """
        
        new_params = []
        for p in self.params:
            if p.name in rename_dict:
                new_params.append(fitParameter(rename_dict[p.name], p.bounds))
            else:
                new_params.append(fitParameter(p.name, p.bounds))
        
        return fitModel(self.name, self.func, new_params, self.guess)
