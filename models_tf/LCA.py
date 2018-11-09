import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

import os

class LCA_Classifier(object):
	def __init__(self, input_units, hidden_units, output_units, sparsity_tradeoff, num_steps, tau, TRAIN_TYPE=None, phi_filename=None, init_image=None, ground_truth=None):
		self.input_units = input_units
		self.hidden_units = hidden_units
		self.output_units = output_units
		self.phi_filename = phi_filename
		self.TRAIN_TYPE = TRAIN_TYPE

		self.sparsity_tradeoff = sparsity_tradeoff
		self.num_steps = num_steps
		self.tau = tau
		self.eta = 1 / tau
		
		if self.TRAIN_TYPE:
			self.x = tf.placeholder(tf.float32, [None, input_units])
			self.y_ = tf.placeholder(tf.float32, [None, output_units])
		else:
			self.x = tf.Variable(tf.constant(init_image), dtype=tf.float32)
			self.y_ = tf.constant(ground_truth)

	## =============
	## LCA =========
	## =============

	def LCA(self):
		# initialize constants
		self.u_init = tf.zeros(shape=[tf.shape(self.x)[0], self.hidden_units], dtype=tf.float32)

		# sparse dictionary
		if self.TRAIN_TYPE == "sparse_dictionary":
			#phi_init = tf.transpose(self.get_saved_phi(self.phi_filename))
			phi_init = tf.truncated_normal([self.input_units, self.hidden_units], mean=0.0, stddev=0.5, dtype=tf.float32, name="phi_init") 
			self.phi = tf.get_variable(name="phi", dtype=tf.float32, initializer=phi_init, trainable=True)
		else:
			self.phi = tf.transpose(self.get_saved_phi(self.phi_filename))

		# self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi, dim=0, name="row_l2_norm")) 
		# self.norm_weights = tf.group(self.norm_phi, name="l2_normalization")

		# actiavtions
		u_step, a_step = self.lca_sparsify()
		self.u = tf.identity(u_step[-1], name="u")
		self.a = tf.identity(a_step[-1], name="a")

	def lca_sparsify(self):
		self.clipped_x = tf.clip_by_value(self.x, 0, 1)
		b = tf.matmul(self.clipped_x, self.phi)  # Driving input
		gramian = tf.subtract(tf.matmul(tf.transpose(self.phi), self.phi), tf.constant(np.identity(self.hidden_units, dtype=np.float32))) # Explaining away matrix
		u_step = [self.u_init] # Initialize membrane potentials to 0
		a_step = [self.lca_threshold(u_step[0])] # Activity vector contains thresholded membrane potentials
		for step in range(self.num_steps):
			du = tf.subtract(tf.subtract(b, tf.matmul(a_step[step], gramian)), u_step[step])  # LCA dynamics define membrane update
			u_update = tf.add(u_step[step], tf.multiply(self.eta, du))  # Update membrane potentials using time constant
			u_step.append(u_update)
			a_step.append(self.lca_threshold(u_step[step+1]))

		return u_step, a_step

	def lca_threshold(self, u):
		a = tf.where(
				tf.greater(u, self.sparsity_tradeoff), tf.subtract(u, self.sparsity_tradeoff), 
			tf.where(
				tf.less(u, -self.sparsity_tradeoff), tf.add(u, self.sparsity_tradeoff), self.u_init)
			)

		return a

	def recon_sparse_loss(self):
		self.S = tf.matmul(self.a, tf.transpose(self.phi), name="recon") 
		self.recon_loss = tf.reduce_mean(.5 * tf.reduce_sum(tf.square(tf.subtract(self.S,self.x)), axis=[1]))
		self.sparse_loss = tf.reduce_mean(self.sparsity_tradeoff * tf.reduce_sum(tf.abs(self.a), axis=[1]))

		self.total_loss = tf.add(self.recon_loss, self.sparse_loss)

	def get_saved_phi(self, filename):
		return tf.constant(np.load(filename)["weights"], name="phi")


 	## =============
 	## Classifier ==
 	## =============

	def Classifier(self):
		self.y = self.fc_layer(self.a, self.hidden_units, self.output_units, RELU=False, name='y')

	def class_loss(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

	def accuracy_function(self):
		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def fc_layer(self, x, input_dim, output_dim, name, RELU=True):

		with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
			weights = tf.get_variable('weights', shape=[input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
			biases = tf.get_variable('biases', initializer = tf.constant(0.1, shape=[output_dim])) #tf.get_variable('biases', shape = [output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))

		out = tf.matmul(x,weights) + biases

		if RELU:
			return tf.nn.relu(out)
		else:
			return out

def train(index, save_folder, TYPE="Classifier", sparsity_tradeoff=.14):
	# define hyperparameters
	# ------------------------------
	INPUT_UNITS = 784
	HIDDEN_UNITS = 400
	OTUPUT_UNITS = 10

	SPARSITY_TRADEOFF = sparsity_tradeoff
	NUM_STEPS = 20
	TAU = 50

	LEARNING_RATE = 1e-1
	EPOCHS = 30
	DISPLAY = .1
	BATCH_SIZE = 100

	PHI_FILENAME = 'lca_mnist_phi.npz'
	# ------------------------------

	# import data
	from tensorflow.examples.tutorials.mnist import input_data
	data = input_data.read_data_sets('MNIST_data', one_hot=True)
	epoch_interval = int(data.train.labels.shape[0]/ BATCH_SIZE)

	# initialize the model
	sess = tf.InteractiveSession()

	##===============
	## LCA ==========
	##===============
	if TYPE == "LCA":
		model = LCA_Classifier(INPUT_UNITS, HIDDEN_UNITS, OTUPUT_UNITS, SPARSITY_TRADEOFF, NUM_STEPS, TAU, TRAIN_TYPE="sparse_dictionary", phi_filename=PHI_FILENAME)
		model.LCA()
		model.recon_sparse_loss()
		train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(model.total_loss)
		#train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(model.total_loss)
		# run the model
		sess.run(tf.global_variables_initializer())

		epoch = 0
		for i in range(EPOCHS * epoch_interval):
			batch = data.train.next_batch(BATCH_SIZE)
			train_step.run(feed_dict={model.x:batch[0]})
			if (i) % (epoch_interval*DISPLAY) == 0:
				loss = model.total_loss.eval(feed_dict={model.x:batch[0]}) 
				print("Epoch %r; Reconstruction loss: %r"%(round(i/epoch_interval,1), loss))
				epoch+=DISPLAY

	##==============
	## Classifier===
	##==============
	elif TYPE == "Classifier":
		model = LCA_Classifier(INPUT_UNITS, HIDDEN_UNITS, OTUPUT_UNITS, SPARSITY_TRADEOFF, NUM_STEPS, TAU, TRAIN_TYPE="classifier", phi_filename=PHI_FILENAME)
		model.LCA()
		model.Classifier()
		classifier_loss = model.class_loss()
		classifier_accuracy = model.accuracy_function()

		train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(classifier_loss)

		# run the model
		sess.run(tf.global_variables_initializer())

		epoch = 0
		for i in range(EPOCHS * epoch_interval):
			batch = data.train.next_batch(BATCH_SIZE)
			train_step.run(feed_dict={model.x:batch[0], model.y_:batch[1]})
			if (i) % (epoch_interval*DISPLAY) == 0:
				acc = classifier_accuracy.eval(feed_dict={model.x:batch[0], model.y_:batch[1]}) 
				print("Sparsity: %r; Epoch %r; Accuracy: %r"%(SPARSITY_TRADEOFF, epoch, acc))
				epoch+=DISPLAY
	else:
		raise NameError(TYPE + " is not a recgnized training type: ['LCA', 'Classifier']")
	
	Saver = tf.train.Saver()
	# save session
	if not os.path.exists(save_folder):
		os.mkdir(save_folder)
		Saver.save(sess, save_folder+"/model")


if __name__ == "__main__":
	sparsity_range = np.linspace(0,1,10)
	directory = "saves_LCA/"
	for i, st in enumerate(sparsity_range[1:]):
		train(index=i, save_folder=directory+str(round(st,3)), sparsity_tradeoff=st, TYPE="LCA")


