import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of sorted_labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and sorted_labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = \
        generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    num_of_leaners = [i for i in range(1,n_learners)]
    training_errors = []
    test_errors = []
    for learner in num_of_leaners:
        training_errors.append(adaboost.partial_loss(train_X, train_y, learner))
        test_errors.append(adaboost.partial_loss(test_X, test_y, learner))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=num_of_leaners, y= training_errors, name="training errors"))
    fig.add_trace(go.Scatter(x=num_of_leaners, y=test_errors, name="test errors"))

    fig.update_layout(
        title="Training and Test errors as function of Number Of Learners",
    )
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if (noise == 0):
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t}}}$" for t in T],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        for i, t in enumerate(T):
            fig2.add_traces([decision_surface(lambda X: adaboost.partial_predict(X,t),
                                              lims[0], lims[1], showscale=False),
                            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                       showlegend=False,
                                       marker=dict(color=test_y,
                                                   colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1)))
                             ],
                           rows=(i // 2) + 1, cols=(i % 2) + 1)

        fig2.update_layout(
            title="Decision boundaries of Adaboost Ensemble",
        )
        fig2.show()

        # Question 3: Decision surface of best performing ensemble
        best_test_error = np.min(test_errors)
        matching_test_number = np.where(test_errors == best_test_error)[0] + 1
        matching_test_number = matching_test_number[0]
        accuracy = 1 - best_test_error

        fig3 = go.Figure()
        fig3.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, matching_test_number),
                                          lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                    showlegend=False,
                                    marker=dict(color=test_y.astype(int),
                                                colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))])
        fig3.update_layout(
            title=f"Decision Surface of Best Test Error <br>Ensemble Size: "
                  f"{matching_test_number} <br>Accuracy: {accuracy}",
        )
        fig3.show()


    # Question 4: Decision surface with weighted samples
    fig4 = go.Figure()
    D = adaboost.D_
    D = D / np.max(D) * 15
    fig4.add_traces([decision_surface(lambda X: adaboost.predict(X),
                                      lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                                showlegend=False,
                                marker_size = D,
                                marker=dict(color=train_y.astype(int),
                                            colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))])
    fig4.update_layout(
        title="Train Set with Point Size Proportional to it's Weight",
    )
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
