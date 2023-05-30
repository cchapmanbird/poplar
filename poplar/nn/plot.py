"""
Plotting routines for visualising network outputs and reporting training/testing diagnostics.
"""

import matplotlib.pyplot as plt
import numpy as np

def loss_plot(train_losses: list, test_losses: list, filename=None):
    """Simple plot routine for producing a loss curve

    Parameters
    ----------
    train_losses : list
        List of training data losses
    test_losses : list
        List of testing data losses
    filename : str, optional
        Output string for the loss curve figure, by default None (in which case, the figure is returned).

    Returns
    -------
    fig (only returned if filename is None)
        The loss curve matplotlib figure.
    """
    epochs = np.arange(len(train_losses))
    fig, ax = plt.subplots()
    ax.semilogy(epochs, train_losses, label='Train')
    ax.semilogy(epochs, test_losses, label='Test')
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
    else:
        return fig