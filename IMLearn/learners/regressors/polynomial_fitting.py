from __future__ import annotations
from typing import NoReturn
from . import LinearRegression
from ...base import BaseEstimator
import numpy as np


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.deg = k
        self.linReg = LinearRegression()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        transpose = X.reshape(X.shape[0], ) # turn into col, assuming colnum=1
        vanDerMatrix = self.__transform(transpose) # get vanDerMatrix
        self.linReg.fit(vanDerMatrix, y)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        transpose = X.reshape(X.shape[0], )  # turn into col, assuming colnum=1
        vanDerMatrix = self.__transform(transpose)  # get vanDerMatrix
        return self.linReg.predict(vanDerMatrix)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True sorted_labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        transpose = X.reshape(X.shape[0], )  # turn into col
        vanDerMatrix = self.__transform(transpose)  # get vanDerMatrix
        return self.linReg.loss(vanDerMatrix, y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(X, N=self.deg + 1, increasing=True)
