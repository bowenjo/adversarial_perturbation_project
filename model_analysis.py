import numpy as np 
import utils.model_utils as utils
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def test_accuracy(z, ground_truth):
	return np.mean((np.argmax(z, axis=0) ==  np.argmax(ground_truth, axis=0)).astype(int))

def test_confidence(z, ground_truth):
	return np.sum(z * ground_truth, axis=0)

def percent_active_units(a):
	return np.mean((a != 0).astype(int), axis=0)

def recon_SNR(x, phi, a):
	recon = phi @ a
	MSE = np.mean((recon - x)**2, axis=0)
	PSNR = 10*np.log10(1/MSE)
	return PSNR

def record_test_metrics(test_set):
	"""
	Records testing metrics for a given model's weights and bias (test_accuracy, test_confidence, ect.)
	"""
	Tk().withdraw() 
	weights_bias_filename = askopenfilename() 
	WB = np.load(weights_bias_filename).item()

	x = test_set[0].T
	ground_truth = test_set[1].T

	metrics = {}
	if WB['model_type'] == 'MLP':
		# forward pass
		numHiddenUnits, numOutputUnits  = WB['weightsTwo'].shape 
		uy = (WB['weightsOne'].T @ x) + np.mean(WB['biasOne'],axis=1).reshape(numHiddenUnits,1)  # must average across batch dimension in bias
		y = utils.relu(uy)  
		uz = (WB['weightsTwo'].T @ y) + np.mean(WB['biasTwo'],axis=1).reshape(numOutputUnits,1)  # must average across batch dimension in bias
		z = utils.softmax(uz) 

		# record metrics
		metrics['test_accuracy'] = test_accuracy(z, ground_truth)
		metrics['test_confidence'] = test_confidence(z, ground_truth)
		metrics['model_type'] = 'MLP'

	elif WB['model_type'] == 'LCA':
		from models.LCA_classifier import lcaSparsify 

		_, numOutputUnits = WB['weights'].shape
		sparseCode = lcaSparsify(x, WB['phi'], WB['tau'], WB['sparsityTradeoff'], WB['numSteps'])
		z = utils.softmax((WB['weights'].T @ sparseCode) + np.mean(WB['bias'],axis=1).reshape(numOutputUnits,1))

		# record metrics
		metrics['test_accuracy'] = test_accuracy(z, ground_truth)
		metrics['test_confidence'] = test_confidence(z, ground_truth)
		metrics['percent_active_units'] = percent_active_units(sparseCode)
		metrics['recon_SNR'] = recon_SNR(x, WB['phi'], sparseCode)
		metrics['model_type'] = 'LCA'

	elif WB['model_type'] == 'DLCA':
		from models.DLCA_classifier import dlcaSparsify 

		_, numOutputUnits = WB['weights'].shape
		sparseCode = dlcaSparsify(x, WB['phi'], WB['tau'], WB['sparsityTradeoff'], WB['numSteps'], WB['weights'], np.mean(WB['bias'],axis=1).reshape(numOutputUnits,1), WB['feedbackRate'])
		z = utils.softmax((WB['weights'].T @ sparseCode) + np.mean(WB['bias'],axis=1).reshape(numOutputUnits,1))

		# record metrics
		metrics['test_accuracy'] = test_accuracy(z, ground_truth)
		metrics['test_confidence'] = test_confidence(z, ground_truth)
		metrics['percent_active_units'] = percent_active_units(sparseCode)
		metrics['recon_SNR'] = recon_SNR(x, WB['phi'], sparseCode)
		metrics['model_type'] = 'DLCA'

	else:
		raise ValueError(str(model_type) + " is not a recognized model type.")

	return metrics

if __name__ == '__main__':
	from data.input_data import load_MNIST
	data = load_MNIST('models/mnistData')

	test_set_size = 5000
	test_set = data['test'].next_batch(test_set_size)

	metrics = record_test_metrics(test_set)

	test_results_dir = 'results/test_results/' + metrics['model_type']
	if not os.path.exists(test_results_dir):
		os.makedirs(test_results_dir)

	np.save(test_results_dir + '/' + str(input('Name the file to be saved in results/test_results/model_type folder: ')), metrics)


