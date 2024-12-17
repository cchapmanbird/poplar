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
        pdf_values = self.pdf(self.xvec[None, :], **pdf_kwargs)
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
        prob = torch.squeeze(torch.pow(x[:,:,None], hypers['lam']) * norm)
        return prob

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
        prob = torch.squeeze(torch.pow(x[:,:,None],hypers['lam']) * norm)
        return prob

class FixedLimitsTruncatedGaussian(Distribution):
    """A truncated normal distribution with fixed limits (i.e. the pdf does not take the limits as arguments, they are fixed at initialisation).

    Ref : http://parker.ad.siu.edu/Olive/ch4.pdf

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

        prob = torch.exp(-(x[:, :, None] - hypers["mu"])**2 / (2 * hypers["sigma"]**2)) 
        prob *= norm 
        return torch.squeeze(prob)

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

class FixedLimitBetaDistribution(Distribution):
    """A Beta distribution with fixed limits (i.e. the pdf does not take the limits as arguments, they are fixed at initialisation).

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

        prob = torch.distributions.Beta(hypers['alpha'], hypers['beta'])
        prob = torch.exp(prob.log_prob(x[:, :, None]))
        return torch.squeeze(prob)


class FixedLimitTruncatedBetaDistribution(Distribution):
    """A Truncated Beta distribution with fixed limits (i.e. the pdf does not take the limits as arguments, they are fixed at initialisation).

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

        alpha = torch.tensor(hypers['alpha'])
        beta = torch.tensor(hypers['beta'])
        B_alpha_beta = torch.exp(torch.lgamma(torch.tensor(alpha)) + torch.lgamma(beta) - torch.lgamma(alpha + beta))  # Beta function

        beta_PDF = (x[:, :, None] ** (alpha - 1)) * ((1 - x[:, :, None]) ** (beta - 1)) / B_alpha_beta
        
        from scipy.stats import beta as sp_beta
        
        cdf_a = sp_beta.cdf(self.limits[0], alpha, beta)
        cdf_b = sp_beta.cdf(self.limits[1], alpha, beta)
        
        norm_factor = cdf_b - cdf_a
        
        prob = beta_PDF / norm_factor
        return torch.squeeze(prob)


class FixedLimits_PowerLawTruncatedGaussian(Distribution):
    """A distribution similar to Peak+PowerLaw with fixed bounds (i.e. the pdf does not take the limits as arguments, they are fixed at initialisation).
    Ref : https://arxiv.org/abs/2010.14533

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
        
        pwerlaw_norm = (1+hypers['lam'])/(self.limits[1]**(1+hypers['lam']) - self.limits[0]**(1+hypers['lam']))

        powerlaw_pdf = torch.squeeze(torch.pow(x[:,:,None], hypers['lam']) * pwerlaw_norm)

        Tgaussian_norm = 2**0.5 / torch.pi**0.5 / hypers["sigma"]
        Tgaussian_norm /= torch.erf((self.limits[1] - hypers["mu"]) / 2**0.5 / hypers["sigma"]) + torch.erf((hypers["mu"] - self.limits[0]) / 2**0.5 / hypers["sigma"]) 

        Tgaussian_prob = torch.exp(-(x[:, :, None] - hypers["mu"])**2 / (2 * hypers["sigma"]**2)) 
        Tgaussian_prob *= Tgaussian_norm 
        Tgaussian_pdf = torch.squeeze(Tgaussian_prob)

        prob = hypers["k1"] * powerlaw_pdf + hypers["k2"] * Tgaussian_pdf
        return prob