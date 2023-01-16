import torch
import numpy as np

class PopulationLikelihood:
    def __init__(self, names, bounds, ppop, data, priors=None, selection_function=None, batch_size=None, device="cpu"):
        self.names = names
        self.bounds = bounds
        self.device = device
        self.ppop = ppop
        self.prepare_data(data)
        self.n_posteriors = self.in_shape[0]
        self.samples_per_posterior = self.in_shape[1]
        self.selection_function = selection_function
        self.batch_size = batch_size
        
        if priors is None:
            priors = {}
        self.priors = priors

        for key in names:
            if key not in priors:
                priors[key] = lambda x: -np.log(self.bounds[key][1] - self.bounds[key][0])

    def prepare_data(self, data):
        # pass in as dictionary of lists of numpy arrays or torch tensors
        datadict = {}
        for n in self.names:
            data_list = data[n]
            to_append = torch.zeros((len(data_list), len(data_list[0])), device=self.device)
            for k, tensor in enumerate(data_list):
                tensor = torch.as_tensor(tensor, device=self.device)
                to_append[k] = tensor
            datadict[k] = tensor

    def log_prior(self, x):
        log_p = np.log(self.in_bounds(x)).astype(np.float64)
        for n in self.names:
            log_p += self.priors[n](x)
        return log_p
    
    def log_likelihood(self, x):
        xd_all = dict()
        for n in self.names:
            xd_all[n] = torch.as_tensor(x[n], device=self.device)
        nhyp = xd_all[self.names[0]].size
        batch_size = self.batch_size
        nbatch = np.ceil(nhyp/self.batch_size).astype(np.int32)
        results = torch.zeros(nhyp, device=self.device)
        for k in range(nbatch):
            start = k*batch_size
            end = min((k+1)*batch_size,nhyp)
            xd = dict()
            for n in self.names:
                xd[n] = xd_all[n][start:end]
            log_l = torch.sum(self.per_event(xd), axis=-1)
            selfact = self.selection_factor(xd) if self.selection_function is not None else 0.
            out = (log_l + selfact)
            results[start:end] = out
        return results.cpu().numpy()

    def selection_factor(self,x):
        ln_l = - x["rate"]*self.selection_function(x) + self.n_posteriors * torch.log(x["rate"])
        return ln_l

    def per_event(self, xd):
        ppop_evaluated = self.ppop(self.data, xd)
        return -np.log(self.samples_per_posterior) +torch.log(torch.sum(ppop_evaluated,axis=-1))

