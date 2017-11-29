import numpy as np
import matplotlib.pyplot as plt
from data.input_data import load_MNIST
import utils.plotFunctions as pf
import utils.model_utils as utils

## --------- Generate Adversarial Images from a Given Model Network --------------------------------------------------------------------------------------

class AdversarialGenerate(object):
    """
    Class that creates an adversarial-image generator on a given model network. First choose the model parameters, then generate adversarial images.
    """
    def __init__(self, weights_bias):
        self.WB = weights_bias
        self.adversarial_image = None

        AVAILABLE_TYPES = ['MLP', 'LCA', 'DLCA']
        if self.WB['model_type'] not in AVAILABLE_TYPES:
            raise ValueError("%s is not a supported model type. Please choose from %s"%(self.model_type, AVAILABLE_TYPES))

    def generate_adversarial_image(self, input_image, ground_truth, epsilon, confidnece_threshold):
        """
        Generates adversarial examples to an pre-trained model. Performs gradient ascent with respect to the image domain and pushes the input image x in the direction 
        of the sign of the delta x.

        Parameters:
        -------------
        input_image: numpy array with dim (numInputUnits,1)
            contains the image to be perturbed
        ground_truth: numpy array with dim (numOutputUnits,1)
            ground truth classification to input_x
        episilon: float
            perturbation size scalar for fast-gradient sign method
        confidence_threshold: float [0,1]
            adversarial confidence value that must be exceeded by the model to stop perturbing the image  
        """
        self.original_image = input_image
        self.ground_truth = ground_truth
        self.confidnece_threshold = confidnece_threshold
        self.epsilon = epsilon

        x = self.original_image.copy()
        self.iterator = 0 
        self.adversarial_example_check = 1
        while self.adversarial_example_check:
            # compute perturbation
            dx = self.perturb_image(x)
            if dx is not None:
                # perturb image in direction of dx (ascent direction) by fast-sign method
                x += epsilon * np.sign(dx)
                # Constrain the pixel value to range [0,1]
                x = np.minimum(x, np.ones_like(x))
                x = np.maximum(x, np.zeros_like(x))
            else:
                if self.iterator == 0:
                    #print("No perturbation was computed. The original model classified the image incorrectly")
                    return None
                else:
                    continue
            self.iterator += 1

        self.adversarial_image = x

    def perturb_image(self, x):
        """
        generates perturbation (dx) of image x by Fast Gradient Sign Method (Goodfellow et. al 2015)
        """
        # -----------------------------------------------------------MLP--------------------------------------------------------------------------
        if self.WB['model_type'] == 'MLP':
            numHiddenUnits, numOutputUnits  = self.WB['weightsTwo'].shape 
            uy = (self.WB['weightsOne'].T @ x) + np.mean(self.WB['biasOne'],axis=1).reshape(numHiddenUnits,1)  # must average across batch dimension in bias
            y = utils.relu(uy)  
            uz = (self.WB['weightsTwo'].T @ y) + np.mean(self.WB['biasTwo'],axis=1).reshape(numOutputUnits,1)  # must average across batch dimension in bias
            z = utils.softmax(uz) 

            if not self.check_correct(z, self.ground_truth) and self.check_confidence(z) >= self.confidnece_threshold:
                self.classification = self.check_classification(z)
                self.confidence = self.check_confidence(z)
                self.adversarial_example_check = 0 
                return None
            else:
                # image layer derivation
                modErrorZ = z - self.ground_truth  
                modErrorY = (self.WB['weightsTwo'] @ modErrorZ) * utils.reluDeriv(uy)  
                dx = self.WB['weightsOne'] @ modErrorY 

        # -----------------------------------------------------------LCA----------------------------------------------------------------------------
        elif self.WB['model_type'] == "LCA":
            from models.LCA_classifier import lcaSparsify 

            _, numOutputUnits = self.WB['weights'].shape
            sparseCode = lcaSparsify(x, self.WB['phi'], self.WB['tau'], self.WB['sparsityTradeoff'], self.WB['numSteps'])
            z = utils.softmax((self.WB['weights'].T @ sparseCode) + np.mean(self.WB['bias'],axis=1).reshape(numOutputUnits,1))

            if not self.check_correct(z, self.ground_truth) and self.check_confidence(z) >= self.confidnece_threshold:
                self.classification = self.check_classification(z)
                self.confidence = self.check_confidence(z)
                self.adversarial_example_check = 0 
                return None
            else:
                # image layer derivation
                dx = self.WB['phi'] @ (self.WB['weights'] @ (z - self.ground_truth))  

        # ----------------------------------------------------------DLCA----------------------------------------------------------------------------
        elif self.WB['model_type'] == "DLCA":
            from models.DLCA_classifier import dlcaSparsify

            _, numOutputUnits = self.WB['weights'].shape
            sparseCode = dlcaSparsify(x, self.WB['phi'], self.WB['tau'], self.WB['sparsityTradeoff'], self.WB['numSteps'], self.WB['weights'], 
                                      np.mean(self.WB['bias'],axis=1).reshape(numOutputUnits,1), self.WB['feedbackRate'])

            z = utils.softmax((self.WB['weights'].T @ sparseCode) + np.mean(self.WB['bias'],axis=1).reshape(numOutputUnits,1)) 

            if not self.check_correct(z, self.ground_truth) and self.check_confidence(z) >= self.confidnece_threshold:
                self.classification = self.check_classification(z)
                self.confidence = self.check_confidence(z)
                self.adversarial_example_check = 0 
                return None
            else:
                # image layer derivation
                dx = self.WB['phi'] @ (self.WB['weights'] @ (z - self.ground_truth))  

        return dx

    def check_correct(self, z, ground_truth):
        """
        checks if model prediction matches ground truth
        """
        return np.argmax(z) == np.argmax(ground_truth)

    def check_confidence(self, z):
        """
        checks confidence of model's classification
        """
        return np.max(z)

    def check_classification(self, z):
        return np.argmax(z)



## -------------- Record Metrics and Show Examples of Adversarial Images -----------------------------------------------------

import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def MSE(image, image_adver):
    #print(image, image_adver)
    return np.mean((image - image_adver)**2)

def PSNR(image, image_adver):
    mse = MSE(image, image_adver)
    return(10 * np.log10(1 / mse))

class AdversarialRecord(object):
    """
    Used to anlayze adversarial examples given a network model weights. Also of examples.
    
    self.record_metrics
        records meaned square error (MSE) and PSNR of adversarial perturbations to MNIST iamges

    self.show_examples
        plots examples of images and their adversarial perturbations

    """
    def __init__(self):
        self.data = load_MNIST('models/mnistData')

        Tk().withdraw() 
        weights_bias_filename = askopenfilename() 
        weights_bias = np.load(weights_bias_filename).item()

        # create the adversarial example object
        self.AE = AdversarialGenerate(weights_bias)

    def record_metrics(self, num_examples, epsilon, confidence_threshold):
        """
        Parameters:
        --------------
        num_examples: int ((0,5000])
            number of MNIST images to record adversarial metrics on 
        """ 

        self.metrics = {"MSE": [], "PSNR": [], "adversarial_confidence": [], 
                        "model_type": self.AE.WB['model_type'], "epsilon": epsilon, "confidence_threshold": confidence_threshold}

        for i in range(num_examples):
            example = self.data["test"].next_batch(1) # load in examples
            self.AE.generate_adversarial_image(example[0].T, example[1].T, epsilon, confidence_threshold) # generate examples

            if self.AE.adversarial_image is None:
                continue # check to make sure original classification was correct

            # append data
            self.metrics["MSE"].append(MSE(self.AE.original_image, self.AE.adversarial_image))
            self.metrics["PSNR"].append(PSNR(self.AE.original_image, self.AE.adversarial_image))
            self.metrics["adversarial_confidence"].append(self.AE.confidence)

        # make adversarial metrics results directory if it does not already exist
        adversarial_results_dir = 'results/adversarial_results/' + self.AE.WB['model_type']
        if not os.path.exists(adversarial_results_dir):
            os.makedirs(adversarial_results_dir)

        # choose name of file and save
        np.save(adversarial_results_dir + '/' + str(input('Name the file to be saved in results/adversarial_results/model_type folder: ')), self.metrics)

    def show_examples(self, size):
        """
        Parameters:
        ---------------
        size: int 
            number of adversarial images to show
        """
        examples = []; adversarial_examples = []; classifications = []; adversarial_classifications = []
        example_size_check = 1
        while example_size_check:
            example = self.data["test"].next_batch(1) # load in examples
            self.AE.generate_adversarial_image(example[0].T, example[1].T, epsilon, confidence_threshold) # generate examples

            if self.AE.adversarial_image is None:
                continue # check to make sure original classification was correct

            # append the data
            examples.append(self.AE.original_image)
            classifications.append(self.AE.check_classification(self.AE.ground_truth))
            adversarial_examples.append(self.AE.adversarial_image)
            adversarial_classifications.append(self.AE.classification)
           
            if len(examples) == size:
                example_size_check = 0
        # print example images with classifications
        fig_o = pf.plotDataTiled(np.asarray(examples).flatten('F').reshape(784,size), classifications, title="Original Images " + self.AE.WB["model_type"])
        fig_a = pf.plotDataTiled(np.asarray(adversarial_examples).flatten('F').reshape(784,size), adversarial_classifications, title="Adversarial Images " + self.AE.WB["model_type"])
        plt.draw()
        plt.pause(10)
        plt.close('all')

if __name__ == '__main__':
    num_examples = 100
    epsilon = 0.01
    confidence_threshold = 0.90
    num_show_examples = 9
    
    Rec = AdversarialRecord()
    Rec.record_metrics(num_examples, epsilon, confidence_threshold)
    Rec.show_examples(num_show_examples)
    print("Metrics Saved")





