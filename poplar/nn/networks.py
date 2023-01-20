import torch
import torch.nn as nn
import dill as pickle
import numpy as np
from .rescaling import IdentityRescaler
from pathlib import Path
import time
from typing import Union, Any


class LinearModel(nn.Module):
    """LinearModel class implementing a standard multi-layer perceptron with some convenience features for function approximation use. 

    This is a subclass of `torch.nn.Module`.

    Parameters
    ----------
    in_features : int
        Number of features for the input layer of the model.
    out_features : int
        Number of features for the output layer of the model.
    neurons : list
            A list containing the number of neurons in each layer of the model (excluding input/output).
    activation : Any
        The activation function to be used for each hidden layer.
    name : str, optional
        A name for the model, used for file naming. Defaults to "model".
    device : str, optional
        pytorch device to initialise the model to, by default "cpu"
    rescaler : _type_, optional
        An object for rescaling inputs/outputs. by default `IdentityRescaler` (see `mlsel.nn.rescaling.py` for examples)
    out_activation : _type_, optional
        Activation function for the output layer, by default None
    initialisation : _type_, optional
        Function for setting the initial weights of all neurons, by default uses xavier_uniform rescaling
    dropout : float, optional
        Sets the dropout probability for all layers, by default 0 (no dropout).
    batch_norm : bool, optional
        If True, enables batch normalisation between layers, by default False
    """
    def __init__(self, in_features: int, out_features: int, neurons: list, activation: Any, name="model", device="cpu", rescaler=None, out_activation=None, initialisation=None, dropout=0., batch_norm=False):
        super().__init__()
            
        self.initial = initialisation
        self.name = name
        if rescaler is None:
            rescaler = IdentityRescaler()
        self.rescaler = rescaler
        if initialisation is None:
            initialisation = torch.nn.init.xavier_uniform_
        self.initialisation = initialisation

        n_layers = len(neurons)

        layers = [nn.Linear(in_features, neurons[0]), activation()]
        for i in range(1, n_layers-1):
            layers.append(nn.Linear(neurons[i], neurons[i+1]))
            if dropout is not False and i > 0:
                layers.append(nn.Dropout(dropout))  
            layers.append(activation())
            if batch_norm and i > 0:
                layers.append(nn.BatchNorm1d(num_features=neurons[i]))
                
        layers.append(nn.Linear(neurons[-1], out_features))
        if out_activation is not None:
            layers.append(out_activation())
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self._init_weights)

        self.to(device)

    def forward(self, x: torch.Tensor):
        """Computes the output for a set of inputs, and removes extra dimensions in the output.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the model.

        Returns
        -------
        torch.Tensor
            The resulting output tensor, with no dimensions of size 1.
        """
        return torch.squeeze(self.layers(x))

    def _init_weights(self, m):
        """Initialise the neural network weights

        Parameters
        ----------
        m : Any
            A network component to be set, if it is a nn.Linear instance.
        """
        if isinstance(m, nn.Linear):
            self.initial(m.weight)

    def save(self, outdir: str):
        """Saves the model to a pickle file for reloading later.

        Parameters
        ----------
        outdir : str
            Output file directory in which to place the model directory.
        """
        Path(f'{outdir}/{self.name}/').mkdir(parents=True, exist_ok=True)
        with open(f'{outdir}/{self.name}/model.pth', 'wb') as pickle_file:
            pickle.dump(self,pickle_file)

    def run_on_dataset(self, inputs: torch.Tensor, n_batches=1, luminosity_distances=None, runtime=False):
        """Run this model on a set of inputs, applying all necessary rescalings and transformations.

        If the output is distance-normalised, luminosity distances can also be provided to convert these into unnormalised values.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor to run through the model.
        n_batches : int, optional
            Number of batches to process the input data in, by default 1 (the entire dataset)
        luminosity_distances : torch.Tensor, optional
            Set of luminosity distance values to multiply the output data by, by default None
        runtime : bool, optional
            If True, returns timing statistics. By default False

        Returns
        -------
        output: torch.Tensor
            The output of the model after reversing the input scalings.
        timings: list (only returned if runtime is True)
            The time taken for the network [in total, per_datapoint].
        """
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

        outputs = self.rescaler.unnormalise(output, "y")

        if luminosity_distances is not None:
            outputs /= luminosity_distances

        if runtime:
            outputs = (outputs, [total_time, per_point])
        return outputs

    def test_threshold_accuracy(self, xdata: torch.Tensor, ydata: torch.Tensor, threshold: float, confusion_matrix=False, **run_kwargs):
        """_summary_

        Parameters
        ----------
        xdata : torch.Tensor
            Set of input (target) data to be processed by the network.
        ydata : torch.Tensor
            Set of true values to compare the network output with.
        threshold : float
            A threshold value with which to compare the accuracy of each network when operating as a classifier (i.e. 0: below threshold, 1: above threshold)
        confusion_matrix : bool, optional
            If True, outputs the result in confusion matrix format. By default False
        **kwargs
            Keyword arguments passed to run_on_dataset.
        Returns
        -------
        accuracy: double
            Accuracy of the network, normalised to [0,1].
        confmat: torch.Tensor (only returned if confusion_matrix is True)
            Confusion matrix of the network output over the two classes (below threshold, above threshold).
        """
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

def load_model(path: str, device="cpu") -> LinearModel:
    """Load an existing `LinearModel` from file.

    Args:
        path (str): Path to `.pkl` file to be loaded.
        device (str, optional): The PyTorch device to load the model to. Defaults to "cpu".).

    Returns:
        LinearModel: Loaded LinearModel.
    """
    with open(path, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    model.to(device)
    return model