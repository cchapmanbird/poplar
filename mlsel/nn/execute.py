import torch
import numpy as np
from sys import stdout
from pathlib import Path
import matplotlib.pyplot as plt
from .plot import loss_plot

def train(model, data, n_epochs, n_batches, loss_function, optimizer, verbose=False, plot=True, return_losses=False, update_every=1, n_test_batches=None, save_best=False, scheduler=None, outdir='models'):

    if n_test_batches is None:
        n_test_batches = n_batches
    
    xtrain, ytrain, xtest, ytest = data
    name = model.name
    Path(f'/{outdir}/{name}/').mkdir(parents=True, exist_ok=True)
    
    xtest = model.rescaler.normalise(xtest, "x")
    ytest = model.rescaler.normalise(xtest, "y")
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
                    optimizer.zero_grad()
                    outputs = model(inputs[i * batch_size:(i+1)*batch_size])
                    loss = loss_function(outputs, targets[i * batch_size: (i+1)*batch_size])
                    loss.backward()
                    optimizer.step()
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
                model.save(f'{outdir}/{name}/model.pth')
        
        if scheduler is not None:
            scheduler.step()

        if verbose:
            stdout.write(f'\rEpoch: {epoch} | Train loss: {train_losses[-1]:.3e} | Test loss: {test_losses[-1]:.3e} (Lowest: {lowest_loss:.3e})')
        
        if update_every is not None:
            if not epoch % update_every:
                if plot:
                    with plt.style.context("seaborn"):
                        loss_plot(train_losses, test_losses, filename=f'{outdir}/{name}/losses.png')
                if not save_best:
                    model.save(f'{outdir}/{name}/model.pth')

        
    if verbose:
        print('\nTraining complete - saving.')
    if not save_best:
        model.save(outdir)
    if plot:
        with plt.style.context("seaborn"):
            loss_plot(train_losses, test_losses, filename=f'{outdir}/{name}/losses.png')

    out = (model,)
    if return_losses:
        out += (train_losses, test_losses,)
    return out
