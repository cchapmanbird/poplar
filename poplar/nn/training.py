"""
Training routines and associated helper functions.
"""

import torch
import numpy as np
from sys import stdout
from pathlib import Path
import matplotlib.pyplot as plt
from .plot import loss_plot
import os
from .networks import LinearModel
from typing import Any, Union
import copy

def train(model: LinearModel, data: list, n_epochs: int, n_batches: int, loss_function: Any, optimiser=None, verbose=False, plot=True, update_every=1, n_test_batches=None, save_best=False, scheduler=None, outdir='models'):
    """Train/test loop for an instance of LinearModel. This function allows for some basic monitoring of the training process, including regular loss curve plots
    and a command line output indicating current progress. 
    
    If you need more complex functionality, it is advisable to write your own training function using this as a starting point.

    Parameters
    ----------
    model : LinearModel
        Instance of LinearModel to be trained.
    data : list
        List of torch.Tensors for the training data, training labels, testing data and testing labels respectively.
    n_epochs : int
        Number of epochs to train for.
    n_batches : int
        Number of batches per epoch.
    loss_function : Any
        Loss function to use. It is recommended to use one of the pytorch loss functions (https://pytorch.org/docs/stable/nn.html#loss-functions)
    optimiser : Any, optional
        The pytorch optimiser to use, by default the Adam optimiser with a learning rate of 1e-3 is used. This should be instantiated before passing to this function.
    verbose : bool, optional
        If True, also displays training progress on the command line. By default False
    plot : bool, optional
        If True, loss curves are regularly produced (with interval update_every) and saved in the model directory, by default True
    update_every : int, optional
        Number of epochs between updating the saved model and plotting diagnostic data, by default 1
    n_test_batches : int, optional
        Number of batches to run the testing data in, by default n_batches
    save_best : bool, optional
        If True, saves the network that achieved the lowest validation losses, by default False
    scheduler : Any, optional
        pytorch scheduler to use for learning rate adjustment during training, by default None
    outdir : str, optional
        Output directory for the trained model directory, by default 'models'
    """
    if n_test_batches is None:
        n_test_batches = n_batches

    if optimiser is None:
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    xtrain, ytrain, xtest, ytest = data
    name = model.name
    Path(f'{outdir}/{name}/').mkdir(parents=True, exist_ok=True)
    
    xtest = model.rescaler.normalise(xtest, "x")
    ytest = model.rescaler.normalise(ytest, "y")
    xtrain = model.rescaler.normalise(xtrain, "x")    
    ytrain = model.rescaler.normalise(ytrain, "y")    

    ytrainsize = len(ytrain)
    ytestsize = len(ytest)

    batch_size = ytrainsize // n_batches
    test_batch_size = ytestsize // n_test_batches

    train_losses = []
    test_losses = []

    datasets = {"train": [xtrain, ytrain], "test": [xtest, ytest]}
    lowest_loss = 1e50
    for epoch in range(n_epochs):
        # Print epoch
        for phase in ['train','test']:
            if phase == 'train':
                model.train(True)
                shuffled_inds = torch.randperm(ytrainsize)

                # Set current loss value
                current_loss = 0.0

                # Get and prepare inputs
                inputs, targets = datasets[phase]
                inputs = inputs[shuffled_inds]
                targets = targets[shuffled_inds]

                for i in range(n_batches):
                    # for param in model.parameters():
                    #     param.grad = None
                    optimiser.zero_grad()
                    outputs = model(inputs[i * batch_size:(i+1)*batch_size])
                    loss = loss_function(outputs, targets[i * batch_size: (i+1)*batch_size])
                    loss.backward()
                    optimiser.step()
                    if scheduler is not None:
                        scheduler.step()
                    current_loss += loss.item()

                train_losses.append(current_loss / n_batches)

            else:
                with torch.no_grad():
                    model.train(False)
                    current_loss = 0.0
                    inputs, targets = datasets[phase]
                    inputs = inputs
                    targets = targets

                    for i in range(n_test_batches):
                        outputs = model(inputs[i * test_batch_size: (i+1)*test_batch_size])
                        loss = loss_function(outputs, targets[i * test_batch_size: (i+1)*test_batch_size])
                        current_loss += loss.item()

                    test_losses.append(current_loss / n_test_batches)
            
        if test_losses[-1] < lowest_loss:
            lowest_loss = test_losses[-1]
            if save_best:
                best_model = copy.deepcopy(model)

        if scheduler is not None:
            scheduler.step()

        if verbose:
            stdout.write(f'\rEpoch: {epoch} | Train loss: {train_losses[-1]:.3e} | Test loss: {test_losses[-1]:.3e} (Lowest: {lowest_loss:.3e})')
        
        if update_every is not None:
            if not epoch % update_every:
                model.loss_curves = [train_losses, test_losses]
                if plot:
                    with plt.style.context("seaborn-v0_8"):
                        loss_plot(train_losses, test_losses, filename=f'{outdir}/losses_{epoch}.png')
                if not save_best:
                    model.save(f'{outdir}/{epoch}')
                else:
                    best_model.loss_curves = [train_losses, test_losses]
                    best_model.save(f'{outdir}/{epoch}')
        
    if verbose:
        print('\nTraining complete - saving.')
    if not save_best:
        model.loss_curves = [train_losses, test_losses]
        model.save(f'{outdir}/final_model')
    else:
        best_model.loss_curves = [train_losses, test_losses]
        best_model.save(f'{outdir}/best_final_model')

    if plot:
        with plt.style.context("seaborn-v0_8"):
            loss_plot(train_losses, test_losses, filename=f'{outdir}/losses_{epoch}.png')
    
def train_test_split(data: Union[torch.tensor, np.ndarray, list], ratio: int, device=None, dtype=None):
    """Splits `data` into two instances of `torch.tensor` with sizes of ratio `ratio` along their first axis. Also supports device
    switching and dtype casting.

    Parameters
    ----------
    data : torch.tensor, numpy.ndarray or list
        The tensors/ndarrays (or list of tensors/ndarrays) to be split.
    ratio : int
        The ratio between the sizes of the two output tensors along their first axis.       
    device : str, optional
        device to move tensors to, by default None (maintains device of inputs)
    dtype : optional
        data type of output tensors, by default None (the same dtype as the input is returned)

    Returns
    -------
    list of tensors
        A list of the two split tensors.
    """

    if isinstance(data, list):
        out = []
        for tensor in data:
            cut_ind = int(ratio * tensor.shape[0])
            if device is None:
                try:
                    device = tensor.device
                except AttributeError:
                    device = "cpu"
            if dtype is None:
                dtype = tensor.dtype
            
            tensor = torch.as_tensor(tensor, device=device, dtype=dtype)
            out.extend([tensor[:cut_ind], tensor[cut_ind:]])
        return out

    else:
        cut_ind = int(ratio * data.shape[0])
        if device is None:
            try:
                device = data.device
            except AttributeError:
                device = "cpu"
        if dtype is None:
            dtype = data.dtype
        
        data = torch.as_tensor(data, device=device, dtype=dtype)
        return data[:cut_ind], data[cut_ind:]

