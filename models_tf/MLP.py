import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import os 

class MLP(object):

	def __init__(self, input_units, hidden_units, output_units, TRAIN=True, init_image=None, ground_truth=None):
		self.input_units = input_units
		self.hidden_units = hidden_units
		self.output_units = output_units

		if TRAIN:
			self.x = tf.placeholder(tf.float32, [None, input_units])
			self.y_ = tf.placeholder(tf.float32, [None, output_units])
		else:
			self.x = tf.Variable(tf.constant(init_image), dtype=tf.float32)
			self.y_ = tf.constant(ground_truth)

		self.build_graph()
		self.loss = self.loss_function()
		self.accuracy = self.accuracy_function()


	def build_graph(self):
		self.clipped_x = tf.clip_by_value(self.x, 0, 1)
		self.h = self.fc_layer(self.clipped_x, self.input_units, self.hidden_units, name="h")
		self.y = self.fc_layer(self.h, self.hidden_units, self.output_units, RELU=False, name="y")

	def loss_function(self):
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

def train(folder_name):
	# define hyperparameters
	# ------------------------------
	INPUT_UNITS = 784
	HIDDEN_UNITS = 400
	OTUPUT_UNITS = 10

	LEARNING_RATE = 1e-1
	EPOCHS = 30
	DISPLAY = 1
	BATCH_SIZE = 100
	# ------------------------------

	# import data
	from tensorflow.examples.tutorials.mnist import input_data
	data = input_data.read_data_sets('MNIST_data', one_hot=True)
	epoch_interval = int(data.train.labels.shape[0]/ BATCH_SIZE)

	# initialize the model
	sess = tf.InteractiveSession()
	model = MLP(INPUT_UNITS, HIDDEN_UNITS, OTUPUT_UNITS, TRAIN=True)
	train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(model.loss)

	# run the model
	sess.run(tf.global_variables_initializer())
	epoch = 0
	for i in range(EPOCHS * epoch_interval):
		batch = data.train.next_batch(BATCH_SIZE)
		train_step.run(feed_dict={model.x:batch[0], model.y_:batch[1]})
		if (i) % (epoch_interval*DISPLAY) == 0:
			acc = model.accuracy.eval(feed_dict={model.x:batch[0], model.y_:batch[1]}) 
			print("Epoch %r; validation accuracy: %r"%(epoch, acc))
			epoch+=DISPLAY

	# save session
	Saver = tf.train.Saver()

	if not os.path.exists(folder_name):
		os.mkdir(folder_name)
		Saver.save(sess, folder_name+"/model")


if __name__ == "__main__":
	train("MLP_saves")







