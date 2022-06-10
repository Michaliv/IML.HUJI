from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error, loss_functions
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    response = lambda x: (x+3) * (x+2) * (x+1) * (x-1) * (x-2)

    X = np.linspace(-1.2,2,n_samples)
    epsilon = np.random.normal(0, noise, n_samples)
    noiseless_y = response(X)
    f_x = noiseless_y + epsilon
    trainX, trainY, testX, testY = split_train_test(pd.DataFrame(X), pd.Series(f_x), 2.0/3)
    trainX = np.array(trainX[0])
    trainY = np.array(trainY)
    testX = np.array(testX[0])
    testY = np.array(testY)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=noiseless_y,
                             name="True Labels", mode="markers+lines", marker=dict(color="black")))
    fig.add_trace(go.Scatter(x=testX, y=testY,
                             name="Test Data", mode="markers", marker=dict(color="blue")))
    fig.add_trace(go.Scatter(x=trainX, y=trainY,
                             name="Train Data", mode="markers", marker=dict(color="red")))
    fig.update_layout(title= "True Noiseless Data and Test and Train Containing Noise")
    fig.show()



    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    training_errors = []
    validation_errors = []
    k_values = np.array([i for i in range(11)])
    min_k = 0
    min_validtion_error = math.inf
    for k in range(11):
        poly_fit = PolynomialFitting(k)
        train, validation = cross_validate(poly_fit, trainX, trainY, mean_square_error, 5)
        training_errors.append(train)
        validation_errors.append(validation)
        if (validation < min_validtion_error):
            min_validtion_error = validation
            min_k = k
    training_errors = np.array(training_errors)
    validation_errors = np.array(validation_errors)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=k_values, y=training_errors,
                             name="Training Errors", mode="markers+lines", marker=dict(color="black")))
    fig2.add_trace(go.Scatter(x=k_values, y=validation_errors,
                             name="Validation Errors", mode="markers+lines", marker=dict(color="blue")))
    fig2.update_layout(title="Training and Validation errors as a function of k",
                       xaxis_title="Polynomial Degree",
                       yaxis_title="Error")
    fig2.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_poly_fit = PolynomialFitting(min_k)
    best_poly_fit.fit(trainX, trainY)
    test_error = round(best_poly_fit.loss(testX, testY),2)
    print("best k is: " + str(min_k))
    print("test error is: " + str(test_error))
    print("original validation error is: " + str(round(min_validtion_error,2)))



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X,y = datasets.load_diabetes(return_X_y=True)
    trainX , trainY, testX, testY = X[:50, :] , y[:50], X[50:, :], y[50:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    num_evaluations = 500
    lambda_range = np.linspace(0.001, 3, num=num_evaluations)

    ridge_training_errors = []
    ridge_validation_errors = []
    lasso_training_errors = []
    lasso_validation_errors = []
    min_ridge_validation_err = math.inf
    best_lambda_ridge = 0
    min_lasso_validation_err = math.inf
    best_lambda_lasso = 0
    for lam in lambda_range:
        ridge = RidgeRegression(lam)
        lasso = Lasso(alpha=lam)

        trainRidge, validationRidge = cross_validate(ridge, trainX, trainY, mean_square_error, 5)
        ridge_training_errors.append(trainRidge)
        ridge_validation_errors.append(validationRidge)
        trainLasso, validationLasso = cross_validate(lasso, trainX, trainY, mean_square_error, 5)
        lasso_training_errors.append(trainLasso)
        lasso_validation_errors.append(validationLasso)
        if validationRidge < min_ridge_validation_err:
            min_ridge_validation_err = validationRidge
            best_lambda_ridge = lam
        if validationLasso < min_lasso_validation_err:
            min_lasso_validation_err = validationLasso
            best_lambda_lasso = lam
    ridge_validation_errors = np.array(ridge_validation_errors)
    ridge_training_errors = np.array(ridge_training_errors)
    lasso_validation_errors = np.array(lasso_validation_errors)
    lasso_training_errors = np.array(lasso_training_errors)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambda_range, y=ridge_training_errors,
                             name="Ridge Training Errors", mode="markers+lines", marker=dict(color="red")))
    fig.add_trace(go.Scatter(x=lambda_range, y=ridge_validation_errors,
                             name="Ridge Validation Errors", mode="markers+lines", marker=dict(color="blue")))
    fig.add_trace(go.Scatter(x=lambda_range, y=lasso_training_errors,
                             name="Lasso Training Errors", mode="markers+lines", marker=dict(color="green")))
    fig.add_trace(go.Scatter(x=lambda_range, y=lasso_validation_errors,
                             name="Lasso Validation Errors", mode="markers+lines", marker=dict(color="yellow")))
    fig.update_layout(title="Train and Validation Errors as Function of Lambda",
                      xaxis_title="Lambda value range",
                      yaxis_title= "Error")
    fig.show()



    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    print("Best regularization value for Ridge: " + str(best_lambda_ridge))
    print("Best regularization value for Lasso: " + str(best_lambda_lasso))

    best_ridge = RidgeRegression(best_lambda_ridge)
    best_ridge.fit(trainX, trainY)

    best_lasso = Lasso(best_lambda_lasso)
    best_lasso.fit(trainX, trainY)

    lin = LinearRegression()
    lin.fit(trainX, trainY)

    print("Test error of Ridge: " + str(best_ridge.loss(testX, testY)))
    y_predict = best_lasso.predict(testX)  # predict responses of given samples
    print("Test error of Lasso: " + str(loss_functions.mean_square_error(testY, y_predict)))
    print("Test error of Least Squares: " + str(lin.loss(testX, testY)))



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100,5)
    select_polynomial_degree(100,0)
    select_polynomial_degree(1500,10)
    select_regularization_parameter()
