import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))
    f1 = []
    f2 = []
	# Feature 1: Measure of Symmetry
	### YOUR CODE HERE
    for image in raw_image:
        f1.append(np.abs(image - np.flip(image)).sum()/256)
    f1 = np.array(f1).reshape(-1,1)
    for image in raw_image:
        f2.append(image.sum()/256)
    
    f2 = np.array(f2).reshape(-1, 1)
	### END YOUR CODE

	# Feature 2: Measure of Intensity
	### YOUR CODE HERE
	
	### END YOUR CODE

	# Feature 3: Bias Term. Always 1.
	### YOUR CODE HERE
	# 1
    f3 = np.ones(raw_image.shape[0]).reshape(-1, 1)
	### END YOUR CODE

	# Stack features together in the following order.
	# [Feature 3, Feature 1, Feature 2]
	### YOUR CODE HERE
    X = np.dstack((f3, f2, f1))
    X = np.reshape(X, (-1, 3))
	### END YOUR CODE
    return X

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx




