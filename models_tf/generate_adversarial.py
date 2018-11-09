import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from MLP import MLP
from LCA import LCA_Classifier

import utils.plotFunctions as pf

from tensorflow.examples.tutorials.mnist import input_data

class Generate_Adversarial(object):
	"""
	Generates adversarial examples from a pre-trained model using the Fast Gradient Sign method (Goodfellow, Shlens, & Szegedy (2014)) 
	"""
	def __init__(self, save_folder, model_type,
				 input_units, hidden_units, output_units, phi_filename=None,
				 sparsity_tradeoff=.14, num_steps=20, tau=50,
				 epsilon=1e-1, conf_thresh=.90):

		"""
		INPUTS:
		--------
		save_folder: str

		"""

		# define hyperparameters
		# ------------------------------
		self.SAVE_FOLDER = save_folder
		self.MODEL_TYPE = model_type
		self.PHI_FILENAME = phi_filename

		self.INPUT_UNITS = input_units
		self.HIDDEN_UNITS = hidden_units
		self.OUTPUT_UNITS = output_units

		self.SPARSITY_TRADEOFF = sparsity_tradeoff
		self.NUM_STEPS = num_steps
		self.TAU = tau

		self.EPSILON = epsilon
		self.CONF_THRESH = conf_thresh
		# ------------------------------

		# initialize test examples
		self.data = input_data.read_data_sets('MNIST_data', one_hot=True)

	def record_metrics(self, num_samples = 10000):
		self.metrics = {"MSE": [], "PSNR": [], "model_type": self.MODEL_TYPE}
		for i in range(num_samples):
			x, a_x = self.generate()[:2]
			if x is None:
				continue
			print(MSE(x,a_x))
			self.metrics["MSE"].append(MSE(x,a_x))
			self.metrics["PSNR"].append(PSNR(x,a_x))
			print(i)


	def generate(self, VISUALIZE=False):
		# initialize model
		sess = tf.InteractiveSession()
		example_image = self.data.test.next_batch(1)

		if self.MODEL_TYPE == "MLP":
			model = MLP(self.INPUT_UNITS, self.HIDDEN_UNITS, self.OUTPUT_UNITS,
						TRAIN=False, init_image=example_image[0], ground_truth=example_image[1])
			loss = model.loss
			name_list = ["h/weights:0", "h/biases:0",
			      		 "y/weights:0", "y/biases:0"]

		elif self.MODEL_TYPE == "LCA":
			model = LCA_Classifier(self.INPUT_UNITS, self.HIDDEN_UNITS, self.OUTPUT_UNITS, self.SPARSITY_TRADEOFF, self.NUM_STEPS, self.TAU,
								   TRAIN_TYPE=None, phi_filename=self.PHI_FILENAME, init_image=example_image[0], ground_truth=example_image[1])
			model.LCA()
			model.Classifier()
			loss = model.class_loss()
			name_list = ["y/weights:0", "y/biases:0"]

		else:
			raise NameError(self.MODEL_TYPE+ " is not a recognized model type: ['MLP', 'LCA']")

		var_list = [v for v in tf.global_variables() if v.name in name_list]
		Saver = tf.train.Saver(var_list=var_list)

		optimizer = tf.train.GradientDescentOptimizer(1)
		grads = optimizer.compute_gradients(tf.negative(loss), [model.x])
		sign_grads = [(self.EPSILON * tf.sign(grad), var) for grad, var in grads] 
		update_image = optimizer.apply_gradients(sign_grads)

		sess.run(tf.global_variables_initializer())
		Saver.restore(sess, self.SAVE_FOLDER +"/model")

		if not self.check_correct(model.y.eval(), example_image[1]):
			return(None, None, None, None)
		if self.confidence(model.y.eval()) <= self.CONF_THRESH:
			return(None, None, None, None)

		# perturb the model
		it = 0
		adversarial_check = 1
		while adversarial_check:
			it+=1
			if it > 1000:
				return(None, None, None, None)
			_, y_step = sess.run([update_image, model.y])
			if not self.check_correct(y_step, example_image[1]) and self.confidence(y_step) >= self.CONF_THRESH:
				adversarial_check = 0
			else:
				adversarial_check = 1
		print("it:", it)

		if VISUALIZE:
			fig, ax = plt.subplots(1,2)
			ax[0].imshow(example_image[0].reshape(28,28), "gray"); ax[0].set_xlabel(str(self.classification(example_image[1]))); ax[0].set_title("Original")
			ax[1].imshow(model.x.eval().reshape(28,28), "gray"); ax[1].set_xlabel(str(self.classification(model.y.eval()))); ax[1].set_title("Adversarial")
			plt.show()

		return(example_image[0], model.clipped_x.eval(), self.classification(example_image[1]), self.classification(model.y.eval()))


	def check_correct(self, z, ground_truth):
		"""
		checks if model prediction matches ground truth
		"""
		return np.argmax(z) == np.argmax(ground_truth)

	def confidence(self, z):
		"""
		checks confidence of model's classification
		"""
		return np.max(z)

	def classification(self, z):
		return np.argmax(z)

	def show_examples(self, size):
		examples = []; classifications = []
		adversarial_examples = []; adversarial_classifications = []
		for i in range(size):
			x, a_x, c, a_c = self.generate()

			print(np.mean(np.abs(x-a_x)))
			print(np.max(a_x), np.min(a_x))
			examples.append(x); adversarial_examples.append(a_x)
			classifications.append(c); adversarial_classifications.append(a_c)

		fig_o = pf.plotDataTiled(np.asarray(examples).flatten('F').reshape(784,size), classifications, title="Original Images " + self.MODEL_TYPE)
		fig_a = pf.plotDataTiled(np.asarray(adversarial_examples).flatten('F').reshape(784,size), adversarial_classifications, title="Adversarial Images " + self.MODEL_TYPE)
		plt.show()


def MSE(image, image_adver):
	return np.mean((image - image_adver)**2)

def PSNR(image, image_adver):
	mse = MSE(image, image_adver)
	return(10 * np.log10(1 / mse))


if __name__ == "__main__":
	# =============
	# MLP==========
	# =============
	save_folder = "./saves_MLP"
	G = Generate_Adversarial(save_folder, "MLP", 784, 400, 10) 
	# G.record_metrics(30)
	# np.save(save_folder+"/adversarial_metrics", G.metrics)
	G.show_examples(4)


	# =============
	# LCA==========
	# =============
	save_folder = "./saves_LCA/"

	sparsity_range = np.linspace(0,1,10)
	for st in [sparsity_range[4]]:
		G = Generate_Adversarial(save_folder+str(round(st,3)), "LCA", 784, 400, 10, sparsity_tradeoff=st, phi_filename='lca_mnist_phi.npz')
		# G.record_metrics(30)
		# np.save(save_folder+str(round(st,3))+"/adversarial_metrics", G.metrics)
		#print(st)
		G.show_examples(4)