class IdentityRescaler:
    def __init__(self) -> None:
        pass

    def normalise(self, data, type):
        return data

    def unnormalise(self, data, type):
        return data

class ZScoreRescaler:
    def __init__(self, xdata, ydata, yfunctions=None) -> None:
        #xy need to be 2d
        if yfunctions is None:
            self.yfunctions = [lambda x: x, lambda x: x]
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
            if type == "y":
                data = self.yfunctions[1](data)
            return data * self.stds[type] + self.means[type]

class UniformRescaler:
    def __init__(self, xdata, ydata, yfunctions=None) -> None:
        if yfunctions is None:
            self.yfunctions = [lambda x: x, lambda x: x]
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
            if type == "y":
                data = self.yfunctions[1](data)
            return (1 + data) / 2 * (self.maxs[type] - self.mins[type]) + self.mins[type]
