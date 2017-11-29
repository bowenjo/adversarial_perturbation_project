import numpy as np
from training_plots.dynamic_plotter import DynamicPlotter
import os

import sys
sys.path.append('../')
import utils.model_utils as utils
from data.input_data import load_MNIST


"""
Train a Multilayer neural network on MNIST
"""


def MLP(numInputUnits, numHiddenUnits, numOutputUnits):
	# initialize weights & bias
	weightsOne = np.random.randn(numInputUnits, numHiddenUnits) # first layer weights
	weightsTwo = np.random.randn(numHiddenUnits, numOutputUnits)  # second layer weights
	biasOne = np.random.randn(numHiddenUnits, batchSize) # first layer bias
	biasTwo = np.random.randn(numOutputUnits, batchSize)  # second layer bias

	DP = DynamicPlotter(epochs)

	for i in range(numTrials):
		# initialize data and teachers from batch
		batch = data["train"].next_batch(batchSize)

		# transpose samples to have dimension (numInputUnits, batchSize)
		teacher = batch[1].T  
		x = batch[0].T  

		# normalize first layer weights and bias
		weightsOne = utils.normalizeColumnDim(weightsOne)

		# forward pass
		uy = np.dot(weightsOne.T, x) + biasOne  
		y = utils.relu(uy)  
		uz = np.dot(weightsTwo.T, y) + biasTwo  
		z = utils.softmax(uz) 

		# second layer backpropagation
		modErrorZ = z - teacher
		deltaWeightsTwo = -eta * (1/batchSize) * np.dot(y, modErrorZ.T)  # (numHiddenUnits x numOutputUnits) dimensional array
		deltaBiasTwo = -eta * (1/batchSize) * modErrorZ  # (numOutputUnits x batchSize) dimensional array

		# first layer backpropagation
		modErrorY = np.dot(weightsTwo, modErrorZ) * utils.reluDeriv(uy)  # (numHiddenUnits x batchSize) dimensional array
		deltaWeightsOne = -eta * (1/batchSize) * np.dot(x, modErrorY.T)  # (numInputUnits x numHiddenUnits) dimensional array
		deltaBiasOne = -eta *(1/batchSize) * modErrorY  # (numHiddenUnits x batchSize) dimensional array

		# update weights and biases
		weightsOne += deltaWeightsOne
		weightsTwo += deltaWeightsTwo
		biasOne += deltaBiasOne
		biasTwo += deltaBiasTwo

		if (i+1) % (epochInterval) == 0:
			train_acc = (1/batchSize) * utils.Accuracy(z, teacher)
			error = (1/batchSize) * utils.crossEntropy(z, teacher) 
			print("Epoch %s: Training accuracy: %s; Error: %s"%((i+1)/epochInterval,train_acc,error))
			DP.update_plot((i+1)/epochInterval, error, train_acc)

		if (i+1) % (10*epochInterval) == 0:
			# save weights
			if not os.path.exists(weights_bias_dir):
				os.makedirs(weights_bias_dir)

			weights_bias = {'weightsOne': weightsOne, 'weightsTwo': weightsTwo, 'biasOne': biasOne, 'biasTwo':biasTwo, 'model_type': 'MLP'}
			np.save(weights_bias_dir + '/epoch_' + str((i+1)/(epochInterval)), weights_bias)

if __name__ == '__main__':
	# Load the data
	data = load_MNIST('mnistData')

	# Set global parameters
	eta = 1e-1 # learning rate
	epochs = 30
	numSamples = 50000 # number of samples in MNIST test set
	batchSize = 100

	epochInterval = int(numSamples / batchSize) 
	numTrials = epochs * epochInterval

	# Results directories
	weights_bias_dir = '../results/weights_bias/mlp'
	MLP(784,400,10)










