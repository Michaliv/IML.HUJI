from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error



class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} sorted_labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_loss = np.inf
        # for all features, calc threshold using -1 and 1
        for sign, j in product([-1,1], range(X.shape[1])):
            cur_tresh, cur_loss = self._find_threshold(X[:, j], y, sign)
            if cur_loss < min_loss:
                min_loss = cur_loss
                self.threshold_ , self.sign_ , self.j_ = cur_tresh, sign, j


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.sign_ * ((X[:, self.j_] >= self.threshold_) * 2 - 1)


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        sorted_labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        indexes_of_sorted = np.argsort(values)
        sorted_values = values[indexes_of_sorted]
        sorted_labels = labels[indexes_of_sorted] # reindex sorted_labels to be same as values

        y_true = np.sign(sorted_labels)  # get real y
        d = np.abs(sorted_labels)  # get d (weights)

        min_err = np.inf
        tresh = np.inf
        # at first iter, because values are sorted, all value are larger or equal
        # to first so they all get sign
        pred = np.repeat(sign, len(sorted_labels))
        for i in range(len(sorted_values)):
            # if sorted_values[i-1] == sorted_values[i] no need to calc again:
            if (i==0 or (i !=0 and sorted_values[i-1] < sorted_values[i])):
                cur_err = np.sum(np.where(y_true != pred, d,
                                          np.zeros(len(sorted_values)))) / len(sorted_labels)
                if (cur_err < min_err):
                    min_err = cur_err
                    tresh = sorted_values[i]
            pred[i] = -sign
        pred = np.repeat(-sign, len(sorted_labels))
        cur_err = np.sum(np.where(y_true != pred, d,
                                          np.zeros(len(sorted_values)))) / len(sorted_labels)
        if (cur_err < min_err):
            min_err = cur_err
            tresh = np.inf
        return tresh , min_err


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True sorted_labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        if (self.fitted_):
            y_pred = self.predict(X)
            return misclassification_error(y, y_pred, True)
