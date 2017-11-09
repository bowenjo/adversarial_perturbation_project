import numpy as np
from data.input_data import load_MNIST
import utils.helperFunctions as hf
import utils.model_utils as utils

"""
Train a LCA model on MNIST
"""

# Load the data
data = load_MNIST('mnistData')

# load pre-trained sparse dictionary phi
phi = np.load('lca_mnist_phi.npz')["weights"]
phi = phi.T 


# Set global parameters
eta = 1e-1 # learning rate
epochs = 30
numSamples = 50000 # number of samples in MNIST test set
batchSize = 100

epochInterval = int(numSamples / batchSize) 
numTrials = epochs * epochInterval

# LCA specific parameters
tau = 50  # proportionality constant in LCA sparse coefficient learning rule
numSteps = 20  # Number of iterations to run LCA
sparsityTradeoff = 0.142857142857  # Lambda parameter that determines how sparse the model will be



def lcaSparsify(data, phi, tau, sparsityTradeoff, numSteps):
    """
    Compute sparse code of input data using the LCA

    Parameters
    ----------
    data : np.ndarray of dimensions (numInputUnits, batchSize) holding a batch of image patches
    phi : np.ndarray of dimensions (numInputUnits, numHiddenUnits) holding sparse coding dictionary
    tau : float for setting time constant for LCA differential equation
    sparsityTradeoff : float indicating Sparse Coding lambda value (also LCA neuron threshold)
    numSteps: int indicating number of inference steps for the LCA model

    Returns
    -------
    a : np.ndarray of dimensions (numHiddenUnits, batchSize) holding thresholded potentials
    """
    b = phi.T @ data  # Driving input
    gramian = phi.T @ phi - np.identity(int(phi.shape[1]))  # Explaining away matrix
    u = np.zeros_like(b)  # Initialize membrane potentials to 0
    for step in range(numSteps):
        a = hf.lcaThreshold(u, sparsityTradeoff)  # Activity vector contains thresholded membrane potentials
        du = b - (gramian.T @ a) - u  # LCA dynamics define membrane update
        u = u + (1.0 / tau) * du  # Update membrane potentials using time constant
    return hf.lcaThreshold(u, sparsityTradeoff)


def LCA_Classifier(numHiddenUnits, numOutputUnits):
    # intitialize weights and bias
    weights = np.random.randn(numHiddenUnits,numOutputUnits) 
    bias = np.random.randn(numOutputUnits,batchSize) 

    for i in range(numTrials):
        # initialize data batch and teacher
        batch = data["train"].next_batch(batchSize)
        x = batch[0].T
        teacher = batch[1].T

        # compute sparse code from pre-trained dictionary
        sparseCode = lcaSparsify(x, phi, tau, sparsityTradeoff, numSteps) 

        # compute output
        z = utils.softmax((weights.T @ sparseCode) + bias)  

        # backpropagation
        modError = z - teacher  
        deltaWeights = -eta * (1 / batchSize) * np.dot(sparseCode,modError.T)  
        deltaBias = -eta * (1 / batchSize) * modError  

        # update weights and bias
        weights += deltaWeights
        bias += deltaBias

        if (i+1) % epochInterval == 0:
            train_acc = (1/batchSize) * utils.Accuracy(z, teacher)
            error = (1/batchSize) * utils.crossEntropy(z, teacher) 
            print("Epoch %s: Training accuracy: %s; Error: %s"%((i+1)/epochInterval,train_acc,error))

            # save weights
            np.savetxt('weights_bias/lca/w_' + str((i+1)/(epochInterval)), weights)
            np.savetxt('weights_bias/lca/b_' + str((i+1)/(epochInterval)), bias)

if __name__ == "__main__":
    LCA_Classifier(400, 10)

