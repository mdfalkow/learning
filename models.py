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
                 reg_const: float = 0,
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

        reg_const : float, optional

            Regularization constant (default: 0).

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
        self.y = np.array(y, dtype='float')
        self.X = np.column_stack(
            (np.ones_like(len(X), dtype='float'), np.array(X, dtype='float')))

        if len(X.shape) != 2:
            raise ValueError(
                f'{X.shape} is invalid shape for X. Should be 2-dimensional.')
        if len(y.shape) != 1:
            raise ValueError(
                f'{y.shape} is invalid shape for y. Should be 1-dimensional.')
        if len(X) == len(y):
            raise ValueError(
                f'Number of points does not match. len(X) ({len(X)}) != len(y) ({len(y)})')

        self.reg_const = reg_const

        self.n_iter = n_iter
        self.debug = debug

        self.w = self.__calc_w_lin()

    def __train_model(self):
        """
        Train model on provided data using Linear regression then pocket 
        perceptron learning algorithm for improvement. 
        """
        E_ = self.__calc_E_in(self.w)
        w_, w = np.copy(self.w), np.copy(self.w)
        for t in range(self.n_iter):
            while True:
                i = random.randint(0, len(X) - 1)  # pick random point
                pred = self.classify[w, self]
                if pred != y:
                    # Update w and E_in
                    w += y * x
                    if E_in < E_:
                        E_ = E_in
                        np.copyto(w_, w)
                    break
        return w_

    def __calc_w_lin(self):
        """
        Get w_lin, the weight vector from linear regression
        """
        n = self.X.shape[1]
        return (np.linalg.inv((self.X.T @ self.X) +
                              reg_const * np.eye(n)) @ self.X.T) @ self.y
        # try:
        #     Xdag = np.linalg.inv(XtX + reg_const * np.identity(len(XtX))) @ Xt
        # except np.linalg.LinAlgError:
        #     Xdag = np.linalg.pinv(XtX + reg_const * np.identity(len(XtX))) @ Xt
        # return Xdag @ labels

    def __calc_E_in(self, w):
        """
        Calculate in-sample error for specified weight vector
        """
        return np.sum(np.abs(np.sign(self.X @ w) - self.y))

    def classify(self, x):
        """
        Returns the classification of x using the weight vector w associated 
        with this model. 

        Returns 1 if sign(w @ x) >= 0 else -1
        """
        return self.classify(x, self.w)

    def classify(self, x, w):
        """
        Returns the classification of x using provided the vector w

        Returns 1 if sign(w @ x) >= 0 else -1
        """
        return 1 if np.sign(np.dot(x, w)) >= 0 else -1
