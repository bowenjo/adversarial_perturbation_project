import numpy as np 
import tensorflow as tf 

import matplotlib.pyplot as plt 


def model_comparison_plot(filename_list, metric, title):
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
	

def sparsity_performance_plot(filename_list, sparsity_range, metric, title, control_file = None):
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
	ax.set_ylim([0,.06])
	fig.suptitle(title)

if __name__ == "__main__":
	sparsity_range = np.linspace(0,1,10)

	MLP_file = "saves_MLP/adversarial_metrics.npy"
	LCA_files = ["./saves_LCA/"+str(round(st, 3))+"/adversarial_metrics.npy" for st in sparsity_range]

	sparsity_performance_plot(LCA_files, sparsity_range, "MSE", "MSE", MLP_file)	
	plt.show()