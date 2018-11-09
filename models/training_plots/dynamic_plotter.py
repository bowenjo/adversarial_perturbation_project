import numpy as np
import matplotlib.pyplot as plt

"""
Dynamic plotter used to visualize accuracy and error of models over time learned
"""

class DynamicPlotter(object):

	def __init__(self, num_epochs):

		self.training_epochs = []
		self.training_accuracy = []
		self.training_error = []

		plt.ion()

		self.figure, self.ax = plt.subplots(1,2, figsize = (8,4))

		self.ax[0].set_title("Cross Entropy")
		self.ax[0].set_xlabel("epoch")
		self.ax[0].set_ylabel("error")
		self.ax[0].set_xlim([0,num_epochs])
		self.ax[0].set_ylim([0,4])

		self.ax[1].set_title("Training Accuracy")
		self.ax[1].set_xlabel("epoch")
		self.ax[1].set_ylabel("training accuracy")
		self.ax[1].set_xlim([0,num_epochs])
		self.ax[1].set_ylim([0,1])

		self.line_error, = self.ax[0].plot(self.training_epochs, self.training_error, 'bo')
		self.line_accuracy, = self.ax[1].plot(self.training_epochs, self.training_accuracy, 'ro')

	def update_plot(self, epoch, error, accuracy):
		self.training_epochs.append(epoch)
		self.training_accuracy.append(accuracy)
		self.training_error.append(error)

		self.line_error.set_xdata(self.training_epochs)
		self.line_error.set_ydata(self.training_error)

		self.line_accuracy.set_xdata(self.training_epochs)
		self.line_accuracy.set_ydata(self.training_accuracy)

		self.figure.canvas.draw()
