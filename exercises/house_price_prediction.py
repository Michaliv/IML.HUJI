import math

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    houseDataFrame = pd.read_csv(filename) # read data set

    remove_invalid_values(houseDataFrame) # remove invalid data

    # replace each none value with the mean of this feature:
    for feature in houseDataFrame:
        meanOfFeature = houseDataFrame[feature].mean()
        houseDataFrame[feature].replace(np.nan, meanOfFeature, inplace=True)

    # add more features:
    add_recently_renewed_feature(houseDataFrame)
    add_number_of_people_feature(houseDataFrame)

    # generate dummy data for zipcode
    houseDataFrame = pd.get_dummies(houseDataFrame, prefix='zipcode',
                                    columns=['zipcode'])

    houseDataFrame.insert(0, 'intercept', 1, True)

    # the fitted model
    # lr = LinearRegression()
    # lr.fit(np.array(houseDataFrame), houseDataFrame['price'])
    # print(lr.predict(np.array(houseDataFrame)))
    # print(houseDataFrame['price'])

    # print(houseDataFrame.head(50).to_string())

    return houseDataFrame.drop("price", 1), houseDataFrame.price

def remove_invalid_values(houseDataFrame):
    """
    removes the invalid values from the data frame
    """
    # remove id, date, lat, long (entire feature)
    houseDataFrame.drop(["id", "long", "date", "lat"], axis=1, inplace=True)

    # remove prices that have to be really positive: (larger than 0)
    removeNegOrZero = ['price', 'sqft_living', 'sqft_living15', 'floors',
                       'sqft_above', 'yr_built']
    for feature in removeNegOrZero:
        houseDataFrame.drop(houseDataFrame[houseDataFrame[feature] <= 0].index,
                            inplace=True)

    # remove prices that have to be non-negative: (could be 0)
    removeNeg = ['bathrooms', 'bedrooms', 'sqft_lot', 'sqft_lot15']
    for feature in removeNeg:
        houseDataFrame.drop(houseDataFrame[houseDataFrame[feature] < 0].index,
                            inplace=True)

    houseDataFrame.drop(houseDataFrame[houseDataFrame["bedrooms"] >= 15].index,
                            inplace=True)
    houseDataFrame.drop(houseDataFrame[houseDataFrame["sqft_living"] >= 12000].index,
                                         inplace=True)
    houseDataFrame.drop(houseDataFrame[houseDataFrame["sqft_lot"] >= 1300000].index,
                                         inplace=True)
    houseDataFrame.drop(houseDataFrame[houseDataFrame["sqft_lot15"] >= 500000].index,
                                         inplace=True)

def add_recently_renewed_feature(houseDataFrame):
    """
    adds to the data set a binary feature of recently renewed-
    if the house was built in the last 10 years or renewed in the last 10 years
    the index will have 1, else- 0
    """
    conditions = [(houseDataFrame['yr_built'] > 2005) |
                  (houseDataFrame['yr_renovated'] > 2005),
                  (houseDataFrame['yr_built'] <= 2005) &
                  (houseDataFrame['yr_renovated'] <= 2005)]
    values = [1, 0]

    houseDataFrame['recently_renewed'] = np.select(conditions, values)

def add_number_of_people_feature(houseDataFrame):
    """
    adds to the data set a feature of number of people who can live in the
    house- we assume that num of bedrooms = num of people who can live in the
    house, but- if the sqft of the house is bigger than the average, the
    house can accomadte more- so we add to the original number the dis
    between the actual sqft of house and the average sqft (meaning the value
    of how bigger the house is than usual), divided by the squareFitPerPerson.
    If the house is equal in size to normal/smaller than average, we don't
    change the value
    """
    # calculate the mean value of sqft above and the mean value of bedrooms:
    meanOfSqftAbove = math.ceil(houseDataFrame['sqft_above'].mean())
    meanOfNumBedrooms = houseDataFrame['bedrooms'].mean()

    # calculate the sf per person:
    squareFitPerPerson = meanOfSqftAbove // meanOfNumBedrooms

    conditions = [(houseDataFrame['sqft_above'] > meanOfSqftAbove),
                               (houseDataFrame['sqft_above'] <= meanOfSqftAbove)]

    calc = np.ceil((1 / squareFitPerPerson) * (houseDataFrame['sqft_above'] -
                                               meanOfSqftAbove))
    values = [houseDataFrame['bedrooms'] + calc, houseDataFrame['bedrooms']]

    houseDataFrame['num_of_people'] = np.select(conditions, values)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.iloc[:, 1:] # disclude "intercept" feature
    for feature in X:
        # cov returns a matrix which the diagonal is var of x and ver of y,
        # and the cov is the non diagonals- so we take [0][1] index:
        cov = np.cov(X[feature], y)[0][1]
        stdX = np.std(X[feature])
        stdY = np.std(y)

        pearCorr = cov / (stdX * stdY)

        # fig = go.Figure()
        df = pd.DataFrame({"Feature Values": X[feature], "Response Values": y})
        fig = px.scatter(df, x= "Feature Values", y="Response Values",
                         trendline="ols", trendline_color_override='darkblue')

        # fig.add_trace(go.Scatter(
        #     x=X[feature],
        #     y= y,
        #     mode="markers"
        # ))

        fig.update_layout(
            title=f"Feature: {feature} <br>Pearson Correlation: {pearCorr}",
            xaxis_title="Feature Values",
            yaxis_title="Response Values"
        )
        # fig.show()
        fig.write_image(output_path + feature + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data('C:/Users/Michal/Desktop/school/cs/year2/semesterB/IML/'
                     'CourseGit/IML.HUJI/datasets/house_prices.csv')
    #
    # # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)
    #
    # # Question 3 - Split samples into training- and testing sets.
    trainX, trainY, testX, testY = split_train_test(X,y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    lin = LinearRegression(True)  # include intercept in lin regression
    trainX.insert(0, 'response', np.array(trainY), True)
    meanLossOfAllSamples = []
    stdLossOfAllSamples = []
    percentages = np.arange(10,101)
    for p in range(10,101):
        lossMseOf10Samples = []
        numOfSamples = (p * trainX.shape[0]) // 100
        for i in range(10):
            sampleSizedP = trainX.sample(n = numOfSamples)
            sampleX, sampleY = sampleSizedP.drop('response', 1), sampleSizedP.response
            lin.fit(sampleX, sampleY)
            lossMseOf10Samples.append(lin.loss(sampleX, sampleY))
        lossMseOf10Samples = np.array(lossMseOf10Samples) # create nparray
        meanLossOfAllSamples.append(np.mean(lossMseOf10Samples)) # add to array
        stdLossOfAllSamples.append(np.std(lossMseOf10Samples)) # array of std

    # meanLossOfAllSamples = np.array(meanLossOfAllSamples)
    # stdLossOfAllSamples = np.array(stdLossOfAllSamples)
    # meanLossPred, stdLossPred = np.mean(meanLossOfAllSamples, axis=0), np.std(meanLossOfAllSamples, axis=0)

    # mult = [i*2 for i in range(len(stdLossOfAllSamples))]
    # subtracted = [element1 - element2 for (element1, element2) in zip(meanLossOfAllSamples, mult)]
    # added = [element1 + element2 for (element1, element2) in zip(meanLossOfAllSamples, mult)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentages, y=meanLossOfAllSamples, mode="markers+lines", name="Mean Loss", line=dict(dash="dash"),
                marker=dict(color="green", opacity=.7)))
    fig.add_trace(go.Scatter(x=percentages, y= np.array(meanLossOfAllSamples) - np.array(stdLossOfAllSamples)*2, fill=None, mode="lines", line=dict(color="lightgrey"),
                               showlegend=False))
    fig.add_trace(go.Scatter(x=percentages, y= np.array(meanLossOfAllSamples) + np.array(stdLossOfAllSamples)*2, fill='tonexty', mode="lines",
                             line=dict(color="lightgrey"),
                             showlegend=False))

    fig.show()





