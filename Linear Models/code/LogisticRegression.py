from random import randint, random
import numpy as np
import sys


"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.W = None

    def sigmoid(self, X): 
        return 1 / (1 + np.exp(-X))

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
		### YOUR CODE HERE
        self.W = np.zeros(n_features)
        for n in range(self.max_iter):
            gradient = self._gradient(X, y)
            vt = -gradient
            self.W = self.W + self.learning_rate * vt
		### END YOUR CODE
        return self
        
    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for n in range(self.max_iter):
            start = 0
            end = batch_size
            while start < X.shape[0]:
            # calc gradient here
                gradient = self._gradient(X[start:end], y[start:end])
                # print(X[start:end].shape, y[start:end].shape)
                vt = -gradient
                self.W = self.W + self.learning_rate * vt
                # update start and end
                start += batch_size
                end += batch_size
                if start > X.shape[0]:
                    break
                if(end > X.shape[0]):
                    end = X.shape[0]
            # print(self.W, "                  -> ", batch_size, n)

        # for n in range(self.max_iter):
        #     while 
        #         
		# ### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        rnd_point = randint(1, X.shape[0] - 1)
        # rnd_point = 10

        for n in range(self.max_iter):
            sigmoidVar = np.exp(y[rnd_point] * (self.W.T * X[rnd_point]))
            mulyiplyer = y[rnd_point] * X[rnd_point] * (self.learning_rate /(1 +  sigmoidVar))
            self.W = self.W.T + mulyiplyer
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        samples, features = _x.shape
        sum = np.zeros(features)
        for i in range(samples):
            numerator = _y[i] * _x[i]
            transposeMulX = _y[i] * (np.dot(_x[i], self.W.T))
            denomenator = 1 + np.exp(transposeMulX)
            sum += numerator/denomenator
        gradient = -(sum/samples)
		### END YOUR CODE
        return gradient

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        # print(self)
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        return self.sigmoid(np.dot(X, self.W))
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        # sum = 0
        # y_hat = []
        # for i in range(X.shape[0]):
        #     y_hat.append(np.round(self.sigmoid(np.dot(self.W.T, X[i]))))
        # tot = self.sigmoid(sum)
        return np.array([self.sigmoid(np.dot(self.W.T, i)) for i in X])
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        y_hat = np.round(self.predict(X))
        y_hat[y_hat == 0] = -1
        return np.sum(np.equal(y_hat, y)) / len(y_hat)
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

