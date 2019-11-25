import math
import random

import numpy as np
from time import time


class BaseModel(object):
    """
    BaseModel class definition. All models inherit from this class.

    Attributes
    ----------
    X : numpy.ndarray (2 dimensional)
        Matrix containing data used to build the model
    y : numpy.ndarray (1 dimensional)
        Vector containing labels for data points in X. 
    """

    def __init__(self, X, y):
        """
        Create a new BaseModel.        

        Parameters
        ----------
        X : numpy.ndarray (2 dimensional)
            Matrix containing the training data. 
        y : numpy.ndarray (1 dimensional)
            Labels for training data. 

        Raises
        ------
        ValueError
            - If argument dimensions do not match.
            - If X is not 2 dimensional.
            - If y is not 1 dimensional.
            - If the first column of X is a dummy feature *WIP*
        """
        X = np.asfarray(X)
        y = np.asfarray(X)

        if len(X.shape) != 2:
            raise ValueError(
                f'{X.shape} is invalid shape for X. Should be 2-dimensional.')
        if len(y.shape) != 1:
            raise ValueError(
                f'{y.shape} is invalid shape for y. Should be 1-dimensional.')
        if len(X) == len(y):
            raise ValueError(
                f'Unequal dimensions. len(X) ({len(X)}) != len(y) ({len(y)})')

        self.X = X
        self.y = y


class LinearModel(BaseModel):
    """
    Implementation of the linear model for classification.

    Attributes
    ----------
    X : numpy.ndarray (2 dimensional)
        Matrix containing data used to build the model.
    y : numpy.ndarray (1 dimensional)
        Vector containing labels for data points in X. 
    w : numpy.ndarray
        Weight vector of the model.
    r : float
        Regularization coefficient for the model.
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 r: float = 0,
                 n_iter: int = 2000,
                 debug: bool = False):
        """
        Create a new LinearModel.

        Parameters
        ----------
        X : numpy.ndarray (2 dimensional)
            Matrix containing the training data. 
        y : numpy.ndarray (1 dimensional)
            Labels for training data. 
        r : float, optional
            Regularization coefficient (default: 0).
        n_iter : int, optional
            Number of training iterations (default: 2000).
        debug : bool, optional
            Prints out training information if True (default: False).

        Raises
        ------
        ValueError
            If conditions for BaseModel are not met.
        """
        super().__init__(X, y)

        self.X = np.asfarray(np.column_stack(
            (np.ones_like(self.X.shape[0]), self.X)))
        self.r = r
        self.w = self.__train_model(n_iter, debug)

    def __train_model(self,
                      n_iter: int,
                      debug: bool):
        """
        Train model on provided data using Linear regression then pocket 
        perceptron learning algorithm for improvement. 
        """
        w_ = self.__calc_w_lin(r)
        E_ = self.__calc_E_in(w_, r)
        if debug:
            print(f'{time()} -- t: 0 -- (w_, E_): {(w_, E_)}')
        w = np.copy(self.w_)
        for t in range(n_iter):
            if debug:
                print(f'{time()} -- t: {t + 1} -- (w_, E_): {(w_, E_)}')
            while True:
                i = random.randint(0, len(self.X) - 1)  # pick random point
                pred = self.__classify(self.X[i], w)
                if pred != self.y[i]:
                    # Update w and E_in
                    w += self.y[i] * self.X[i]
                    E_in = self.__calc_E_in(w, r)
                    if E_in < E_:
                        E_ = E_in
                        np.copyto(w_, w)
                    break
        if debug:
            print(f'{time()} -- t: {n_iter} -- Best Results: (w_, E_): {(w_, E_)}')

        return w_

    def __calc_w_lin(self):
        """
        Get w_lin, the weight vector from linear regression

        Parameters
        ----------
        r : float
            Regularization coefficient

        """
        n = self.X.shape[1]
        return (np.linalg.inv((self.X.T @ self.X) +
                              self.r * np.eye(n)) @ self.X.T) @ self.y

    def __calc_E_in(self, w: np.ndarray, r: float):
        """
        Calculate in-sample error for specified weight vector

        Parameters
        ----------
        w : numpy.ndarray

        """
        pred = np.sign(self.X @ w)
        pred[pred == 0] = 1
        return np.sum(np.abs(pred - self.y)) + self.r * (w @ w)

    def __classify(self, x, w):
        """
        Returns the classification of x using provided the vector w

        Returns
        -------
        int 
             1 if `np.dot(x, w)` >= 0
            -1 otherwise
        """
        return 1 if np.sign(np.dot(x, w)) >= 0 else -1

    @staticmethod
    def classify(x):
        """
        Returns the classification of x using the weight vector w of this model. 

        Parameters
        ----------
        x : numpy.ndarray
            Point to classify

        Returns
        -------
        int 
             1 if `np.dot(x, w)` >= 0
            -1 otherwise
        """
        return self.__classify(x, self.w)


class kNNModel(BaseModel):
    """
    Implementation of the k-Nearest Neighbors model for classification.

    Attributes
    ----------
    X : numpy.ndarray (2 dimensional)
        Matrix containing data used to build the model.
    y : numpy.ndarray (1 dimensional)
        Vector containing labels for data points in X. 
    k : int
        Number of nearest neighbors to check
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, k: int = 1):
        """
        Create a new kNNModel

        Parameters
        ----------
        X : numpy.ndarray (2 dimensional)
            Matrix containing data used to build the model
        y : numpy.ndarray (1 dimensional)
            Vector containing labels for data points in X. 
        k : int, optional
            Number of nearest neighbors to check (Default : 1)

        Raises
        ------
        ValueError
            If conditions for BaseModel are not met.
        """
        super().__init__(X, y)
        self.k = k

    @staticmethod
    def __euclidean_distance(x1, x2) -> float:
        """
        Returns distance between two points

        Parameters
        ----------
            x1 : numpy.ndarray
                First point
            x2 : numpy.ndarray
                Second point

        Returns
        -------
        float
            Euclidean distance between x1 and x2.
        """
        return np.linalg.norm(x1 - x2)

    def classify(self, x: np.ndarray, k: int = None) -> int:
        """
        Classify a point based on the k-nearest neighbors.

        Parameters
        ----------
        x : numpy.ndarray
            Point to classify
        k : int, optional
            Number of nearest neighbors to check 
            (Default: None, model uses attribute k).

        Returns
        -------
        int 
             1 if the k-nearest neighbors have labels >= 0
            -1 otherwise
        """
        # use provided, otherwise use default
        k = k if k != None else self.k

        # get distance to each point in X, then sort
        distances = sorted((distance(x0, X[i]), y[i]) for i in range(len(X)))

        # get k nearest neighbors, then take the sum of the labels
        nearest_neighbors = distances[:k]
        return sum(label for distance, label in nearest_neighbors)
