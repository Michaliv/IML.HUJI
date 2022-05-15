from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different sorted_labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.fitted_ = True
        # num of classes (=k) is the number of unique sorted_labels in y:
        self.classes_ = np.unique(y)
        n_k = []  # count num of yi==k for each k in classes
        for k in self.classes_:
            n_k.append(np.count_nonzero(y == k))
        n_k = np.array(n_k)
        self.pi_ = n_k * (1 / X.shape[0])

        outer_mu = []
        for k in range(self.classes_.shape[0]):
            sum = np.zeros(X.shape[1])
            for i in range(X.shape[0]):
                if (y[i] == self.classes_[k]):
                    sum += X[i]
            sum *= 1 / (n_k[k])
            outer_mu.append(sum)
        self.mu_ = np.array(outer_mu)

        # calc variance
        outer_var = []
        for k in range(self.classes_.shape[0]):
            single_row = []
            for i in range(X.shape[1]):
                sum = 0
                for j in range(X.shape[0]):
                    if y[j] == self.classes_[k]:
                        calc = X[j][i] - self.mu_[k][i]
                        calc_twice = (1 / (n_k[k] - 1)) * calc * calc
                        sum += calc_twice
                single_row.append(sum)
            outer_var.append(single_row)
        self.vars_ = np.array(outer_var)

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
        if self.fitted_:
            y_pred = [self.predict_helper(sample) for sample in X]
            return np.array(y_pred)

    def predict_helper(self, sample):
        """
        helper function for predict- calculates the needed value for a single
        sample
        """
        to_maximaize = []
        # calc the probability for each class:
        for i in range(self.classes_.shape[0]):
            log = np.log(self.pi_[i])
            single = np.sum(np.log(self.normal_dist_pdf(i, sample)))
            single = log + single
            to_maximaize.append(single)

        # return class with highest probability calculated
        return self.classes_[np.argmax(to_maximaize)]

    def normal_dist_pdf(self, class_ind, sample):
        """
        helper function to predict- calculates the normal distribution pdf of
        the sample and the class index
        """
        mean = self.mu_[class_ind]
        var = self.vars_[class_ind]
        inside_exp = -((sample - mean) ** 2) / (2 * var)
        exp_calc = np.exp(inside_exp)
        sqrt_calc = np.sqrt(2 * np.pi * var)
        return exp_calc / sqrt_calc

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

        likelihood = []
        for sample in X:
            class_likelihood = []
            for k in self.classes_:
                class_likelihood.append(self.normal_dist_pdf(k, sample))
            likelihood.append(class_likelihood)
        return np.array(likelihood)

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
            return misclassification_error(y, y_pred)
