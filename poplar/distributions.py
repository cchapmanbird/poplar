"""
Some standard distributions with vectorised probability density functions (pdfs) and a routine for numerical cumulative density function (CDF) computation and distribution sampling.

If the user requires other distributions, they can implement them as their own class by subclassing the Distribution base class and adding their own methods as required.
"""

import torch
from .utilities import interpolate

class Distribution:
    def __init__(self, limits, npoints=1000, grid_spacing='linear', device="cpu") -> None:
        """Distribution base class, implementing inverse-transform sampling method for the univariate distribution case. This can be subclassed by the user
        to specify generic distributions as required.

        Parameters
        ----------
        limits : list
            [Lower, Upper] limits for this distribution. Used to initialise the CDF computational grid.
        npoints : int, optional
            number of points in CDF computational grid, by default 1000
        grid_spacing : str, optional
            specifies either linear or logarithmic spacing for CDF computational grid, by default 'linear'
        device : str, optional
            pytorch device this distribution operates using, by default "cpu"
        """
        self.limits = torch.as_tensor(limits, device=device)
        if grid_spacing == 'linear':
            self.xvec = torch.linspace(self.limits[0], self.limits[1], npoints, device=device)
        elif grid_spacing == 'logarithmic':
            self.xvec = torch.logspace(torch.log10(self.limits[0]), torch.log10(self.limits[1]), npoints, device=device)
        self.dx = torch.diff(self.xvec, prepend=torch.tensor([self.limits[0],], device=device))
        self.device = device

    def _sanitise_inputs(self, **inputs):
        outs = {}
        for a, b in inputs.items():
            b_tensor = torch.as_tensor(b, device=self.device)
            if b_tensor.ndim == 0:
                b_tensor = b_tensor[None]
            outs[a] = b_tensor
        return outs

    def pdf(self):
        raise NotImplementedError
    
    def cdf(self, **pdf_kwargs):
        pdf_values = self.pdf(self.xvec, **pdf_kwargs)
        renorm = self.dx * pdf_values
        pdf_values *= 1/renorm.sum(axis=-1)
        return torch.cumsum(pdf_values * self.dx, axis=-1)

    def draw_samples(self,  size=1, **pdf_kwargs):
        cdf = self.cdf(**pdf_kwargs)
        if cdf.ndim == 1:
            draws = interpolate(torch.rand(size, device=self.device)[None,:], cdf[None,:], self.xvec[None,:])
        else:
            draws = interpolate(torch.rand((cdf.shape[0],size), device=self.device), cdf, self.xvec)
        return torch.squeeze(draws) 

    def to(self, device):
        self.xvec.to(device)
        self.dx.to(device)
        self.device = device

class FixedLimitsPowerLaw(Distribution):
    """A power law distribution with fixed limits (i.e. the pdf does not take the limits as arguments, they are fixed at initialisation).

    Parameters
    ----------
    limits : list
        [Lower, Upper] limits for this distribution. Used to initialise the CDF computational grid.
    npoints : int, optional
        number of points in CDF computational grid, by default 1000
    grid_spacing : str, optional
        specifies either linear or logarithmic spacing for CDF computational grid, by default 'linear'
    device : str, optional
        pytorch device this distribution operates using, by default "cpu"
    """
    def __init__(self, limits, npoints=1000, grid_spacing='logarithmic', device="cpu") -> None:
        super().__init__(limits, npoints, grid_spacing, device)
    
    def pdf(self, x, **hypers):
        x = torch.as_tensor(x, device=self.device)
        hypers = self._sanitise_inputs(**hypers)
        norm = (1+hypers['lam'])/(self.limits[1]**(1+hypers['lam']) - self.limits[0]**(1+hypers['lam']))
        return torch.squeeze(x[None,:]**hypers['lam'][:,None] * norm[:,None])

class FixedLimitsTruncatedGaussian(Distribution):
    """A truncated normal distribution with fixed limits (i.e. the pdf does not take the limits as arguments, they are fixed at initialisation).

    Parameters
    ----------
    limits : list
        [Lower, Upper] limits for this distribution. Used to initialise the CDF computational grid.
    npoints : int, optional
        number of points in CDF computational grid, by default 1000
    grid_spacing : str, optional
        specifies either linear or logarithmic spacing for CDF computational grid, by default 'linear'
    device : str, optional
        pytorch device this distribution operates using, by default "cpu"
    """
    def __init__(self, limits, npoints=1000, grid_spacing='linear', device="cpu") -> None:
        super().__init__(limits, npoints, grid_spacing, device)
    
    def pdf(self, x, **hypers):
        x = torch.as_tensor(x, device=self.device)
        hypers = self._sanitise_inputs(**hypers)

        norm = 2**0.5 / torch.pi**0.5 / hypers["sigma"]
        norm /= torch.erf((self.limits[1] - hypers["mu"]) / 2**0.5 / hypers["sigma"]) + torch.erf((hypers["mu"] - self.limits[0]) / 2**0.5 / hypers["sigma"]) 
        
        prob = torch.exp(-(x[None,:] - hypers["mu"][:,None])**2 / (2 * hypers["sigma"][:,None]**2)) 
        prob *= norm[:,None]  
        return torch.squeeze(prob)

class VariableLimitsPowerLaw(Distribution):
    """A power law distribution with variable limits (i.e. the pdf takes the limits as arguments).

    Parameters
    ----------
    limits : list
        [Lower, Upper] limits for this distribution. Used to initialise the CDF computational grid.
    npoints : int, optional
        number of points in CDF computational grid, by default 1000
    grid_spacing : str, optional
        specifies either linear or logarithmic spacing for CDF computational grid, by default 'linear'
    device : str, optional
        pytorch device this distribution operates using, by default "cpu"
    """
    def __init__(self, limits, npoints=1000, grid_spacing='logarithmic', device="cpu") -> None:
        super().__init__(limits, npoints, grid_spacing, device)
    
    def pdf(self, x, **hypers):
        x = torch.as_tensor(x, device=self.device)
        hypers = self._sanitise_inputs(**hypers)

        norm = (1+hypers['lam'])/(hypers['xhigh']**(1+hypers['lam']) - hypers['xlow']**(1+hypers['lam']))
        edges = (x[None,:] > hypers['xlow'][:,None]) * (x[None,:] < hypers['xhigh'][:,None])
        return torch.squeeze(x[None,:]**hypers['lam'][:,None] * norm[:,None] * edges) 

class UniformDistribution(Distribution):
    """A uniform distribution with fixed limits (i.e. the pdf does not take the limits as arguments, they are fixed at initialisation).

    Parameters
    ----------
    limits : list
        [Lower, Upper] limits for this distribution. Used to initialise the CDF computational grid.
    npoints : int, optional
        number of points in CDF computational grid, by default 1000
    grid_spacing : str, optional
        specifies either linear or logarithmic spacing for CDF computational grid, by default 'linear'
    device : str, optional
        pytorch device this distribution operates using, by default "cpu"
    """
    def __init__(self, limits, npoints=1000, grid_spacing='linear', device="cpu") -> None:
        super().__init__(limits, npoints, grid_spacing, device)

    def pdf(self, x):
        return 1 / (self.limits[1] - self.limits[0])