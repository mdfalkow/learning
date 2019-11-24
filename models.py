import math
import random

import numpy as np


class LinearModel(object):
    """
    Implementation of the linear model for classification.
    Uses perceptron learning algorithm with pocket for training.

    Attributes:
    -----------------
    w : numpy.ndarray
        weight vector of the model
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 r: float = 0,
                 n_iter: int = 2000,
                 debug: bool = False):
        """
        Create a new LinearModel

        Parameters
        ----------
        X : numpy.ndarray (2 dimensional)
            Matrix containing the training data. 
            Requires: 
                Points should be expressed as rows.
                Does contain dummy/bias feature (model does this).

        y : numpy.ndarray (1 dimensional)
            Labels for training data. 
            Requires:
                Length should match X

        r : float, optional
            Regularization coefficient (default: 0).

        n_iter : int, optional
            Number of training iterations (default: 2000).

        debug : bool, optional
            Prints out training information if True (default: False).

        Raises
        ------
        Value error
            - If argument dimensions do not match.
            - If X is not 2 dimensional.
            - If y is not 1 dimensional.
            - If the first column of X is 
        """
        y = np.asfarray(y)
        X = np.asfarray(np.column_stack((np.ones_like(len(X)), np.array(X))))

        if len(X.shape) != 2:
            raise ValueError(
                f'{X.shape} is invalid shape for X. Should be 2-dimensional.')
        if len(y.shape) != 1:
            raise ValueError(
                f'{y.shape} is invalid shape for y. Should be 1-dimensional.')
        if len(X) == len(y):
            raise ValueError(
                f'Number of points does not match. len(X) ({len(X)}) != len(y) ({len(y)})')

        self.w = self.__train_model(X, y, r)

    def __train_model(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      r: float):
        """
        Train model on provided data using Linear regression then pocket 
        perceptron learning algorithm for improvement. 
        """
        w_ = self.__calc_w_lin(X, y, r)
        E_ = self.__calc_E_in(w_, r)
        w = np.copy(self.w_)
        for t in range(self.n_iter):
            while True:
                i = random.randint(0, len(X) - 1)  # pick random point
                pred = self.__classify(X[i], w)
                if pred != y:
                    # Update w and E_in
                    w += y[i] * X[i]
                    E_in = self.__calc_E_in(w, r)
                    if E_in < E_:
                        E_ = E_in
                        np.copyto(w_, w)
                    break
        return w_

    def __calc_w_lin(self, X: np.ndarray, y: np.ndarray, r: float):
        """
        Get w_lin, the weight vector from linear regression
        """
        n = X.shape[1]
        return (np.linalg.inv((X.T @ X) + r * np.eye(n)) @ X.T) @ y

    def __calc_E_in(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    w: np.ndarray,
                    r: float):
        """
        Calculate in-sample error for specified weight vector
        """
        pred = np.sign(X @ w)
        pred[pred == 0] = 1
        return np.sum(np.abs(pred - y)) + r * (w @ w)

    def __classify(self, x, w):
        """
        Returns the classification of x using provided the vector w

        Returns 1 if sign(w @ x) >= 0 else -1
        """
        return 1 if np.sign(np.dot(x, w)) >= 0 else -1

    @staticmethod
    def classify(x):
        """
        Returns the classification of x using the weight vector w of this model. 

        Returns 1 if sign(w @ x) >= 0 else -1
        """
        return self.__classify(x, self.w)
