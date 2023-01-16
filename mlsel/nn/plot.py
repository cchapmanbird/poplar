import matplotlib.pyplot as plt
import numpy as np

def loss_plot(train_losses, test_losses, filename=None):
    epochs = np.arange(len(train_losses))
    fig = plt.figure()
    plt.semilogy(epochs, train_losses, label='Train')
    plt.semilogy(epochs, test_losses, label='Test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        return fig