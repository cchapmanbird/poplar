"""
Rescaling objects to be passed to LinearModel to handle the rescaling of input/output data. 

These are some standard rescalers to provide something easy to use out of the box; for more complex rescaling behaviour, create your own
rescaling class using these examples as a template.
"""


import torch

class IdentityRescaler:
    """A placeholder rescaler that leaves input/output data unchanged. Functions may still be applied to the targets by passing them to yfunctions.

    Parameters
    ----------
    yfunctions : list, optional
        A list containing a function and its inverse to apply to the labels prior to rescaling, by default None (i.e. no function is applied)
    """
    def __init__(self,yfunctions=None) -> None:

        if yfunctions is None:
            yfunctions = [lambda x: x, lambda x: x]
        self.yfunctions = yfunctions

    def normalise(self, data, type):
        if type == "y":
            data = self.yfunctions[0](data)
        return data

    def unnormalise(self, data, type):
        if type == "y":
            out = self.yfunctions[1](data)
        return out

class ZScoreRescaler:
    """Rescales data to the unit normal distribution.

    Parameters
    ----------
    xdata : torch.Tensor
        Input data.
    ydata : torch.Tensor
        Input labels corresponding to xdata.
    yfunctions : list, optional
        A list containing a function and its inverse to apply to the labels prior to rescaling, by default None (i.e. no function is applied)
    """
    def __init__(self, xdata: torch.Tensor, ydata, yfunctions=None) -> None:

        #xy need to be 2d
        if yfunctions is None:
            yfunctions = [lambda x: x, lambda x: x]
        self.yfunctions = yfunctions
        ydata = self.yfunctions[0](ydata)
        self.means = dict(x=xdata.mean(axis=0), y=ydata.mean(axis=0))
        self.stds = dict(x=xdata.std(axis=0), y=ydata.std(axis=0))

    def normalise(self, data, type):
        if type not in ["x", "y"]:
            raise ValueError("Pass either x or y for normalisation")
        else:
            if type == "y":
                data = self.yfunctions[0](data)
            return (data - self.means[type]) / self.stds[type]

    def unnormalise(self, data, type):
        if type not in ["x", "y"]:
            raise ValueError("Pass either x or y for normalisation")
        else:
            out = data * self.stds[type] + self.means[type]
            if type == "y":
                out = self.yfunctions[1](out)
            return out

    def to(self, device):
        for key in self.means.keys():
            self.means[key] = self.means[key].to(device)
            self.stds[key] = self.stds[key].to(device)

class UniformRescaler:
    """Rescales data to the uniform distribution with bounds [-1, 1].

    Parameters
    ----------
    xdata : torch.Tensor
        Input data.
    ydata : torch.Tensor
        Input labels corresponding to xdata.
    yfunctions : list, optional
        A list containing a function and its inverse to apply to the labels prior to rescaling, by default None (i.e. no function is applied)
    """
    def __init__(self, xdata, ydata, yfunctions=None) -> None:
        if yfunctions is None:
            yfunctions = [lambda x: x, lambda x: x]
        self.yfunctions = yfunctions
        #xy need to be 2d
        ydata = self.yfunctions[0](ydata)
        self.mins = dict(x=xdata.min(axis=0),y=ydata.min(axis=0))
        self.maxs = dict(x=xdata.max(axis=0),y=ydata.max(axis=0))

    def normalise(self, data, type):
        if type not in ["x", "y"]:
            raise ValueError("Pass either x or y for normalisation")
        else:
            if type == "y":
                data = self.yfunctions[0](data)
            return 2 * (data - self.mins[type]) / (self.maxs[type] - self.mins[type]) - 1

    def unnormalise(self, data, type):
        if type not in ["x", "y"]:
            raise ValueError("Pass either x or y for normalisation")
        else:
            out = (1 + data) / 2 * (self.maxs[type] - self.mins[type]) + self.mins[type]
            if type == "y":
                out = self.yfunctions[1](out)
            return out

    def to(self, device):
        for key in self.mins.keys():
            self.mins[key] = self.mins[key].to(device)
            self.maxs[key] = self.maxs[key].to(device)
