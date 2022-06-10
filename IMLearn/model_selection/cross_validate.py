from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = 0
    validation_score = 0
    # separate train set into cv equally sized sets:
    indexes = np.arange(X.shape[0])
    folds = np.array_split(indexes, cv)
    for fold in folds:
        # train set:
        boolean_arr = np.ones(X.shape[0], bool)
        boolean_arr[fold] = False # put false in index matching the test set loc (to be seperated)
        train_X_without_i = X[boolean_arr] # gets only indexes matching True
        train_Y_without_i = y[boolean_arr]

        # test set:
        testX = X[fold]
        testY = y[fold]

        estimator.fit(train_X_without_i, train_Y_without_i)
        # estimate error on train set:
        train_score += scoring(train_Y_without_i, estimator.predict(train_X_without_i))
        # estimate error on test set:
        validation_score += scoring(testY, estimator.predict(testX))
    return float(train_score) / cv, float(validation_score) / cv
