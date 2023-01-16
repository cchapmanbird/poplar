import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import dill as pickle
import numpy as np
from .rescaling import IdentityRescaler
from pathlib import Path
import time

class LinearModel(nn.Module):
    def __init__(self, out_features, neurons, n_layers, activation, name, device="cpu", rescaler=None, out_activation=None, initialisation=xavier_uniform_, dropout=False, batch_norm=False):
        super().__init__()

        if isinstance(neurons, list):
            if len(neurons) != layers:
                raise RuntimeError('Length of neuron vector does not equal number of hidden layers.')
        else:
            neurons = np.ones_like(n_layers)*neurons
            
        self.initial = initialisation
        self.name = name
        if rescaler is None:
            rescaler = IdentityRescaler()
        self.rescaler = rescaler

        layers = []
        for i in range(n_layers):
            layers.append(nn.LazyLinear(neurons[i]))
            if dropout is not False and i > 0:
                layers.append(nn.Dropout(dropout))  
            layers.append(activation())
            if batch_norm and i > 0:
                layers.append(nn.BatchNorm1d(num_features=neurons[i+1]))
                
        layers.append(nn.LazyLinear(out_features))
        if out_activation is not None:
            layers.append(out_activation())
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)

        self.to(device)

    def forward(self, x):
        return self.layers(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.initial(m.weight)

    def save(self, outdir):
        Path(f'/{outdir}/{self.name}/').mkdir(parents=True, exist_ok=True)
        pickle.dump(self,file=f'/{outdir}/{self.name}/model.pth')

    def run_on_dataset(self, inputs, n_batches=1, luminosity_distances=None, runtime=False):
        if self.training is True:
            self.eval()

        normalised_inputs = self.rescaler.normalise(inputs, "x")

        if runtime:
            st = time.perf_counter()

        if n_batches > 1:
            with torch.no_grad():
                out = []
                for _ in range(n_batches):
                    output = self(normalised_inputs)
                    out.append(output)
                output = torch.cat(out)
        else:
            with torch.no_grad():
                output = self(normalised_inputs)

        if runtime:
            et = time.perf_counter()
            total_time = et - st
            per_point = (et - st) / normalised_inputs.shape[0]

        try:
            if output.shape[1] == 1:
                output = output[:,0]        
        except IndexError:
            pass

        outputs = self.rescaler.unnormalise(output, "y")

        if luminosity_distances is not None :
            outputs /= self.luminosity_distances

        if runtime:
            outputs = (outputs, [total_time, per_point])
        return outputs

    def test_threshold_accuracy(self, xdata, ydata, threshold, confusion_matrix=False, **run_kwargs):
        ypred = self.run_on_dataset(xdata, **run_kwargs)        
        out_classified = torch.zeros_like(ypred, device=self.device)
        out_classified[ypred >= threshold] = 1

        truth_classified = torch.zeros_like(ydata)
        truth_classified[ydata >= threshold] = 1

        if not confusion_matrix:
            return 1 - torch.mean(torch.abs(out_classified - truth_classified))
        else:
            confmat = torch.zeros((2,2), device=self.device)
            confmat[0,0] = torch.sum(torch.logical_and(out_classified==0,truth_classified==0))
            confmat[0,1] = torch.sum(torch.logical_and(out_classified==0,truth_classified==1))
            confmat[1,0] = torch.sum(torch.logical_and(out_classified==1,truth_classified==0))
            confmat[1,1] = torch.sum(torch.logical_and(out_classified==1,truth_classified==1))

            return (1-torch.mean(torch.abs(out_classified-truth_classified)),confmat)

def load_model(path):
    model = pickle.load(path)
    model.to(model.device())
    return model