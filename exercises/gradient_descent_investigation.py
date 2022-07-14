import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from plotly.subplots import make_subplots

import IMLearn
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import mean_square_error, misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))



def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights = []
    vals = []

    def callback(**kwargs):
        weights.append(kwargs["weights"])
        vals.append(kwargs["vals"])
    return callback, vals, weights




def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    best_gd1 = 2
    best_etha_l1 = 0
    best_gd2 = 2
    best_etha_l2 = 0

    for etha in etas:
        L1_module = L1(init)
        L2_module = L2(init)

        l1_callback, l1_vals, l1_weights = get_gd_state_recorder_callback()
        l2_callback, l2_vals, l2_weights = get_gd_state_recorder_callback()

        gd1 = GradientDescent(learning_rate=FixedLR(etha), callback=l1_callback)
        gd1.fit(L1_module, np.zeros((10,10)), np.zeros((10,1)), ) # fit using L1

        if (l1_vals[-1] < best_gd1): # check if last value in values is a better loss for l1
            best_gd1 = l1_vals[-1]
            best_etha_l1 = etha

        gd2 = GradientDescent(learning_rate=FixedLR(etha), callback=l2_callback)
        gd2.fit(L2_module, np.zeros((10, 10)), np.zeros((10, 1)))  # fit using L2

        if (l2_vals[-1] < best_gd2): # check if last value in values is a better loss for l1
            best_gd2 = l2_vals[-1]
            best_etha_l2 = etha

        fig1 = plot_descent_path(IMLearn.desent_methods.modules.L1, np.array(l1_weights))
        fig1.show()

        fig2 = plot_descent_path(IMLearn.desent_methods.modules.L2, np.array(l2_weights))
        fig2.show()

        # Q2
        iters = np.arange(1000)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=iters, y=l1_vals,
                                 mode="markers", marker=dict(color="blue")))
        fig3.update_layout(title=f"L1 Norm as Function Of Iteration Number for etha: {etha}",
                          xaxis_title="Number of Iteration",
                          yaxis_title="L1 norm")
        fig3.show()

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=iters, y=l2_vals,
                                  mode="markers", marker=dict(color="blue")))
        fig4.update_layout(title=f"L2 Norm as Function Of Iteration Number for etha: {etha}",
                           xaxis_title="Number of Iteration",
                           yaxis_title="L2 norm")
        fig4.show()

    print(f"best etha for l1: {best_etha_l1}, loss achieved: {best_gd1}")
    print(f"best etha for l2: {best_etha_l2}, loss achieved: {best_gd2}")





def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = make_subplots(rows=2, cols=2)
    iter_x = 1
    iter_y = 1
    counter = 1

    best_gamma = 0
    best_loss = 2

    colors = ["blue", "red", "yellow", "green"]

    for gamma in gammas:
        L1_module = L1(init)
        l1_callback, l1_vals, l1_weights = get_gd_state_recorder_callback()
        iters = np.arange(1000)

        gd1 = GradientDescent(learning_rate=ExponentialLR(eta, gamma), callback=l1_callback)
        gd1.fit(L1_module, np.zeros((10, 10)), np.zeros((10, 1)), )  # fit using L1
        fig.add_traces([go.Scatter(x=iters, y=l1_vals,
                                  mode="markers", marker=dict(color=colors[counter-1], size=4),
                                   name=f"gamma: {gamma}")],
                      rows=iter_x, cols=iter_y)
        fig.update_layout(title=f"L1 Norm as Function Of Iteration Number for etha: {eta}")
        if counter == 1:
            iter_x = 1
            iter_y = 2
        elif counter == 2:
            iter_x = 2
            iter_y = 1
        elif counter == 3:
            iter_x = 2
            iter_y = 2

        counter += 1

        if (l1_vals[-1] < best_loss):  # check if last value in values is a better loss for l1
            best_loss = l1_vals[-1]
            best_gamma = gamma

    fig.show()

    # Q6:
    print(f"best gamma for l1: {best_gamma}, loss achieved: {best_loss}")

    # Plot descent path for gamma=0.95
    L1_module = L1(init)
    L2_module = L2(init)

    exp_LR = ExponentialLR(base_lr=eta, decay_rate=0.95)

    l1_callback, l1_values, l1_weights = get_gd_state_recorder_callback()

    gd1 = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=l1_callback)
    gd1.fit(L1_module, np.zeros((10, 10)), np.zeros((10, 1)), )  # fit using L1
    fig2 = plot_descent_path(L1, np.concatenate(l1_weights, axis=0).reshape(len(l1_weights), len(init)),
                                          title=f"L1 Module trajectory: eta={eta}, gamma = 0.95")
    fig2.show()

    l2_callback, l2_values, l2_weights = get_gd_state_recorder_callback()

    gd2 = GradientDescent(learning_rate=ExponentialLR(eta, 0.95), callback=l2_callback)
    gd2.fit(L2_module, np.zeros((10, 10)), np.zeros((10, 1)), )  # fit using L2
    fig3 = plot_descent_path(L2, np.concatenate(l2_weights, axis=0).reshape(len(l2_weights), len(init)),
                                          title=f"L2 Module trajectory: eta={eta}, gamma = 0.95")
    fig3.show()



def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data
    alphas = np.arange(0, 1.01, 0.01)
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)

    custom = [[0.0, "rgb(165,0,38)"],
              [0.1111111111111111, "rgb(215,48,39)"],
              [0.2222222222222222, "rgb(244,109,67)"],
              [0.3333333333333333, "rgb(253,174,97)"],
              [0.4444444444444444, "rgb(254,224,144)"],
              [0.5555555555555556, "rgb(224,243,248)"],
              [0.6666666666666666, "rgb(171,217,233)"],
              [0.7777777777777778, "rgb(116,173,209)"],
              [0.8888888888888888, "rgb(69,117,180)"],
              [1.0, "rgb(49,54,149)"]]

    c = [custom[0], custom[-1]]

    y_prob = logistic_reg.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=alphas, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

    best_alpha = round(thresholds[np.argmax(tpr - fpr)], 2)
    print(f"Best alpha: {best_alpha}")
    best_log_reg_for_alpha = LogisticRegression(alpha=best_alpha)
    y_pred = best_log_reg_for_alpha.fit(X_train, y_train).predict(X_test)
    print(f"Test error for alpha = {best_alpha}: {misclassification_error(y_test, y_pred)}")



    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambda_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    penalties = ["l1", "l2"]
    for penalty in penalties:
        training_results, validation_results = [], []
        for lam in lambda_range:
            logistic_reg = LogisticRegression(penalty=penalty, alpha=0.5, lam=lam)
            training_score, validation_score = cross_validate(logistic_reg,
                                                              X_train, y_train,
                                                              misclassification_error)
            training_results.append(training_score)
            validation_results.append(validation_score)
        validation_results = np.array(validation_results)
        best_lam = lambda_range[np.argmin(validation_results)]
        print(f"Best lambda for {penalty}: {best_lam}")
        best_log_reg = LogisticRegression(penalty=penalty, alpha=0.5, lam=best_lam)
        y_pred = best_log_reg.fit(X_train, y_train).predict(X_test)
        print(f"Test error for {penalty}: {misclassification_error(y_test, y_pred)}")





if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
