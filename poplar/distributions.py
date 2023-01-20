import torch
from .utilities import interpolate

class Distribution:
    def __init__(self, limits, npoints=1000, grid_spacing='linear', device="cpu") -> None:
        self.limits = limits
        if grid_spacing == 'linear':
            self.xvec = torch.linspace(limits[0], limits[1], npoints, device=device)
        elif grid_spacing == 'logarithmic':
            self.xvec = torch.logspace(torch.log10(limits[0]), torch.log10(limits[1]), npoints, device=device)
        self.dx = torch.diff(self.xvec, prepend=torch.tensor([limits[0],], device=device))
        self.device = device

    def pdf(self):
        raise NotImplementedError
        # return torch.squeeze(x[None,:]**lam[:,None])
    
    def cdf(self, *pdf_args):
        pdf_values = self.pdf(self.xvec, *pdf_args)
        renorm = self.dx * pdf_values
        pdf_values *= 1/renorm.sum(axis=-1)
        return torch.cumsum(pdf_values * self.dx, axis=-1)

    def draw_samples(self, *pdf_args, size=1):
        cdf = self.cdf(*pdf_args)
        if cdf.ndim == 1:
            return interpolate(torch.rand(size, device=self.device), cdf, self.xvec)
        else:
            draws = torch.zeros((cdf.shape[0], size), device=self.device)
            for k, per_cdf in enumerate(cdf):
                draws[k] = interpolate(torch.rand(size, device=self.device), per_cdf, self.xvec)
            return draws

    def to(self, device):
        self.xvec.to(device)
        self.dx.to(device)
        self.device = device

class FixedLimitsPowerLaw(Distribution):
    def __init__(self, limits, npoints=1000, grid_spacing='logarithmic', device="cpu") -> None:
        super().__init__(limits, npoints, grid_spacing, device)
    
    def pdf(self, x, lam):
        norm = (1+lam)/(self.limits[1]**(1+lam) - self.limits[0]**(1+lam))
        return torch.squeeze(x[None,:]**lam[:,None] * norm[:,None])

class FixedLimitsTruncatedGaussian(Distribution):
    def __init__(self, limits, npoints=1000, grid_spacing='linear', device="cpu") -> None:
        super().__init__(limits, npoints, grid_spacing, device)
    
    def pdf(self, x, mu, sigma):
        norm = 2**0.5 / torch.pi**0.5 / sigma
        norm /= torch.erf((self.limits[1] - mu) / 2**0.5 / sigma) + torch.erf((mu - self.limits[0]) / 2**0.5 / sigma) 
        
        prob = torch.exp(-(x[None,:] - mu[:,None])**2 / (2 * sigma[:,None]**2)) 
        prob *= norm[:,None]  
        return torch.squeeze(prob)

class VariableLimitsPowerLaw(Distribution):
    def __init__(self, limits, npoints=1000, grid_spacing='logarithmic', device="cpu") -> None:
        super().__init__(limits, npoints, grid_spacing, device)
    
    def pdf(self, x, lam, xlow, xhigh):
        norm = (1+lam)/(xhigh**(1+lam) - xlow**(1+lam))
        edges = (x[None,:] > xlow[:,None]) * (x[None,:] < xhigh[:,None])
        return torch.squeeze(x[None,:]**lam[:,None] * norm[:,None] * edges) 