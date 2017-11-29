import numpy as np

"""
helpful functions for multilayer neural models
"""

# --------------General Functions-------------------------------------------------------------------

def normalizeColumnDim(weights):
    norm = np.linalg.norm(weights, axis=0)
    return(weights / (norm[None, :] + np.finfo(float).eps))

def softmax(u):
    norm = np.sum(np.exp(u), axis=0)
    return(np.exp(u) / (norm[None, :] + np.finfo(float).eps))

def relu(u):
    return np.maximum(0, u)

def reluDeriv(u):
	u_new = np.zeros_like(u)
	u_new[np.where(u>0)] = 1
	return(u_new)

def crossEntropy(z, teacher):
	return(-np.sum(teacher * np.log(z+np.finfo(float).eps)))

def Accuracy(z, teacher):
	trainAccArray = np.array(np.argmax(z, axis=0) == np.argmax(teacher, axis=0))
	trainAccArray.astype(int)
	return(np.sum(trainAccArray))



    
