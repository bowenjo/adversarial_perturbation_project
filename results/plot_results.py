import numpy as np 
import matplotlib.pyplot as plt 

def model_comparison(filename_list, metric, title):
	"""
	Bar chart that compares a designated metric for each model in filename_list 
	"""

	# orgnaize the data
	N = len(filename_list) # number of models being compared
	means = []; SEMs = []; models = []
	for file in filename_list:
		metrics = np.load(file).item()
		# append data
		means.append(np.mean(metrics[metric]))
		if metric != 'test_accuracy':
			SEMs.append(np.std(metrics[metric])/np.sqrt(len(metrics[metric])))
		models.append(metrics['model_type'])

	# create the bar chart
	width = 0.35
	ind = np.arange(N)  # the x locations for the groups     

	fig, ax = plt.subplots()
	if metric == 'test_accuracy':
		ax.bar(ind, means, color='g')
	else:
		ax.bar(ind, means, color='g', yerr=SEMs)
	ax.set_xticks(ind)
	ax.set_xticklabels(models)
	ax.set_ylabel(metric)
	ax.set_xlabel("model")
	fig.suptitle(title)
	

def sparsity_performance(filename_list, sparsity_range, metric, title, control_file = None):
	"""
	compares LCA model performace as a function of sparsity (activated neural units)
	can compare against a single control group for certain metrics (MLP)
	"""
	means = []; SEMs = []
	for file in filename_list:
		metrics = np.load(file).item()
		means.append(np.mean(metrics[metric]))
		if metric != 'test_accuracy':
			SEMs.append(np.std(metrics[metric])/np.sqrt(len(metrics[metric])))

	fig, ax = plt.subplots()
	if metric == 'test_accuracy':
		ax.errorbar(sparsity_range, means, fmt='o-', label = metrics['model_type'])
	else:
		ax.errorbar(sparsity_range, means, yerr = SEMs, fmt='o-', label = metrics['model_type'])
	ax.set_xlabel('sparsity coefficient')
	ax.set_ylabel(metric)
	if control_file is not None:
		control_metrics  = np.load(control_file).item()
		control_mean = np.mean(control_metrics[metric]); # control_SEM = np.std(control_metrics[metric]) / np.sqrt(len(control_metrics[metric]))
		ax.plot(sparsity_range, [control_mean]*len(sparsity_range), 'r--', label=control_metrics['model_type'])
	ax.legend(loc=1)
	fig.suptitle(title)
	

if __name__ == '__main__':
	# -------------------Compare each models MSE to generated adversarial examples ------------------------------------
	# also can test other adversarial metrics (PSNR). 
	filename_list = ['adversarial_results/MLP/mlp_30epochs.npy',
					 'adversarial_results/LCA/s_.22.npy',
					 'adversarial_results/DLCA/dlca_epochs30.npy'
					 ]
	model_comparison(filename_list, "MSE", "Adversarial Results Model Comparison")

	# -------------------Compare each models testing accuracy ---------------------------------------------------------
	# also can test other testing metrics (test_confidence).
	filename_list = ['test_results/MLP/mlp_epochs30.npy',
	 				 'test_results/LCA/s_.22.npy',
	 				 'test_results/DLCA/dlca_epochs30.npy'
	 				 ]
	model_comparison(filename_list, "test_accuracy", "Test Results Model Comparison")

	#-------------------Compare MSE to generated adversarial example as function of LCA sparsity coeffiecient----------
	# also can test other adversarial metrics (PSNR).
	sparsity_filename_list = ['adversarial_results/LCA/s_0.npy',
							  'adversarial_results/LCA/s_.11.npy',
							  'adversarial_results/LCA/s_.22.npy',
							  'adversarial_results/LCA/s_.33.npy',
							  'adversarial_results/LCA/s_.44.npy',
							  'adversarial_results/LCA/s_.55.npy',
							  'adversarial_results/LCA/s_.66.npy',
							  'adversarial_results/LCA/s_.77.npy',
							  'adversarial_results/LCA/s_.88.npy']

	control_file = 'adversarial_results/MLP/mlp_30epochs.npy' # MLP MSE line

	sparsity_range = np.linspace(0,1,10)[:9] # range of sparsity coefficients used
	sparsity_performance(sparsity_filename_list, sparsity_range, 'MSE', "Adversarial Performance", control_file)

	#-----------------Compare test accuracy as a function of LCA sparsity coeffiecient -------------------------------
	# also can test other testing metrics (test_confidence, percent_activated_units, recon_SNR).
	sparsity_filename_list = ['test_results/LCA/s_0.npy',
							  'test_results/LCA/s_.11.npy',
							  'test_results/LCA/s_.22.npy',
							  'test_results/LCA/s_.33.npy',
							  'test_results/LCA/s_.44.npy',
							  'test_results/LCA/s_.55.npy',
							  'test_results/LCA/s_.66.npy',
							  'test_results/LCA/s_.77.npy',
							  'test_results/LCA/s_.88.npy']

	control_file = 'test_results/MLP/mlp_epochs30.npy' # MLP test accuracy line (comment out for percent_activated_units and recon_SNR) 

	sparsity_range = np.linspace(0,1,10)[:9]
	sparsity_performance(sparsity_filename_list, sparsity_range, 'test_accuracy', "Test Performance", control_file)

	plt.show()







