import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.fitted_ = True
        # num of classes (=k) is the number of unique labels in y:
        self.classes_ = np.unique(y)
        n_k = [] # count num of yi==k for each k in classes
        for k in self.classes_:
            n_k.append(np.count_nonzero(y==k))
        n_k = np.array(n_k)
        self.pi_ = n_k * (1/X.shape[0])

        outer_mu = []
        for k in range(self.classes_.shape[0]):
            sum = np.zeros(X.shape[1])
            for i in range(X.shape[0]):
                if (y[i] == self.classes_[k]):
                    sum += X[i]
            sum *= 1 / (n_k[k])
            outer_mu.append(sum)
        self.mu_ = np.array(outer_mu)

        outer_sigma = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            sub = np.array(X[i] - self.mu_[y[i]]).reshape(2,1)
            transpose = np.array(sub.T)
            outer_sigma += sub @ transpose

        outer_sigma *= 1 / (X.shape[0] - self.classes_.shape[0]) #unbiased
        self.cov_ = np.array(outer_sigma)
        self._cov_inv = inv(self.cov_)

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
        a_k_of_all = []
        b_k_of_all = []
        for k in range(self.classes_.shape[0]):
            m_k_trans = self.mu_[k].T
            a_k_of_all.append(self._cov_inv @ m_k_trans)
            b_k_of_all.append(np.log(self.pi_[k])-0.5 * self.mu_[k] @ self._cov_inv @  self.mu_[k])

        y_pred = []
        for i in range(X.shape[0]):
            max_val = -math.inf
            max_k = 0
            for k in range(self.classes_.shape[0]):
                cur_val = a_k_of_all[k].T.reshape(1,-1) @ X[i].reshape(2,1) + b_k_of_all[k]

                if (cur_val > max_val):
                    max_val = cur_val
                    max_k = k
            y_pred.append(self.classes_[max_k])
        return np.array(y_pred)


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m = X.shape[0]
        d = X.shape[1]
        scalar = 1 / np.sqrt(np.power(2 * np.pi, d)) * np.linalg.det(self.cov_)
        likelihood = []
        for i in range(m):
            single_row = []
            for k in range(len(self.classes_)):
                one_side = X[i] - self.mu_[k]
                exp = -0.5 * one_side.T @ self._cov_inv @ one_side
                res = scalar * np.exp(exp) * self.pi_[k]
                single_row.append(res)
            likelihood.append(single_row)
        return np.array(likelihood)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        if (self.fitted_):
            y_pred = self.predict(X)
            return misclassification_error(y, y_pred)


