from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
     File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent
     features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
     linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable",'../datasets/linearly_separable.npy'),
                 ("Linearly Inseparable", '../datasets/linearly_inseparable.npy')]:
        # Load dataset
        X , y_true = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def callback_func(fit: Perceptron, x: np.ndarray, y: int):
            losses.append(fit.loss(X,y_true))

        percept_object = Perceptron(callback=callback_func)
        percept_object.fit(X,y_true) # uses callback_func

        # Plot figure of loss as function of fitting iteration
        iterations = [i for i in range (len(losses))]

        fig = px.line(x=iterations, y=losses, title=n + ": Loss Values as a function of Iteration Number")

        fig.update_layout(
            xaxis_title="Iteration Number",
            yaxis_title="Loss after each iteration"
        )


        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else \
        (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")



def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ['../datasets/gaussian1.npy', '../datasets/gaussian2.npy']:
        # Load dataset
        X , y_true = load_dataset(f)

        # Fit models and predict over training set
        lda_model = LDA()
        lda_model.fit(X,y_true)
        lda_y_pred = lda_model.predict(X)

        naive_bayes_model = GaussianNaiveBayes()
        naive_bayes_model.fit(X,y_true)
        naive_bayes_y_pred = naive_bayes_model.predict(X)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lda_model_acc = accuracy(y_true, lda_y_pred)
        naive_bayes_acc = accuracy(y_true, naive_bayes_y_pred)

        # Add traces for data-points setting symbols and colors
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Naive Bayes accuracy: {acc}".format(acc= naive_bayes_acc),
                            "LDA accuracy: {acc}.".format(acc= lda_model_acc)))

        fig.add_trace(go.Scatter(x=(X.T)[0], y=(X.T)[1], mode="markers",
                                 showlegend=False,
                                 marker=dict(color= lda_y_pred, symbol=y_true,
                                             colorscale=[custom[1], custom[2]],
                                 line=dict(color="black", width=1))), row=1, col=2)

        fig.add_trace(go.Scatter(x=(X.T)[0], y=(X.T)[1], mode="markers",
                                 showlegend=False,
                                 marker=dict(color=naive_bayes_y_pred, symbol=y_true,
                                             colorscale=[custom[1], custom[2]],
                                             line=dict(color="black", width=1))), row=1, col=1)

        fig.update_layout(height=550, width=950, title_text= "Gaussian Dataset ")


        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=np.array(lda_model.mu_)[:, 0],
                                 y=np.array(lda_model.mu_)[:, 1],
                                 mode="markers",
                                 showlegend=False,
                                 marker=dict(color="black", symbol="x",
                                                               line=dict(color="black", width=1), size=10)),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=np.array(naive_bayes_model.mu_)[:, 0],
                                 y=np.array(naive_bayes_model.mu_)[:, 1],
                                 mode="markers",
                                 showlegend=False,
                                 marker=dict(color="black", symbol="x",
                                                               line=dict(color="black", width=1), size=10)),
                      row=1, col=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(lda_model.mu_.shape[0]):
            fig.add_trace(get_ellipse(np.array(lda_model.mu_)[i, :], lda_model.cov_),
                          row=1, col=2)

        for i in range(naive_bayes_model.mu_.shape[0]):
            fig.add_trace(get_ellipse(np.array(naive_bayes_model.mu_)[i, :],
                                      np.diag(naive_bayes_model.vars_[i, :])),
                          row=1, col=1)

        fig.update_layout(showlegend=False)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
