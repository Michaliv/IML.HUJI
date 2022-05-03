import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    temperatureDataFrame = pd.read_csv(filename, parse_dates=['Date'])
    # add DayOfYear col:
    temperatureDataFrame['DayOfYear'] = temperatureDataFrame['Date'].dt.dayofyear

    # remove invalid negative values (under -20 degrees in Israel-> invalid)
    temperatureDataFrame.drop(temperatureDataFrame[temperatureDataFrame["Temp"] < -20].index,
                            inplace=True)

    return temperatureDataFrame

def explore_country_temp_by_days_of_year(X: pd.DataFrame):
    """
    plots a graph of temp in Israel by days of the year
    """
    X["Year"] = X["Year"].astype(str) # convert year to str for discrete graph
    fig = px.scatter(X, x="DayOfYear", y="Temp", color="Year",
                     title="Temperature in Israel by days of the year")

    fig.show()

def explore_country_std_by_month(X: pd.DataFrame):
    """
    creates a graph for the std of temp in Israel by each month
    """
    xMonthsStd = X.groupby('Month').Temp.agg([np.std])
    months = [str(month) for month in range(1,13)]
    fig = px.bar(xMonthsStd, x=months, y="std")
    fig.update_layout(
        title="The STD of Temperature in Israel by month",
        xaxis_title="Months",
        yaxis_title="STD Values"
    )
    fig.show()

def all_countries_std_and_average_by_month(X: pd.DataFrame):
    """
    creates a graph for the mean temp by month with std error bars, for each
    country in the data set
    """
    xMonthAndCountryStdMean = X.groupby(["Country", "Month"]).Temp.agg([np.mean, np.std])
    fig = px.line(xMonthAndCountryStdMean, x=xMonthAndCountryStdMean.index.get_level_values("Month"),
                  y="mean", error_y="std",
                  color=xMonthAndCountryStdMean.index.get_level_values("Country"))
    fig.update_layout(
        title="Mean Temperature by month with STD error bars",
        xaxis_title="Months",
        yaxis_title="Mean Values"
    )
    fig.show()

def fit_over_k_degree(X: pd.DataFrame, y: pd.Series):
    """
    fits the model over the data set of israel with different degree of
    polynomials and plots a graph
    """
    trainX, trainY, testX, testY = split_train_test(X,y)
    lossesArr = []
    kArr = [str(k) for k in range(1,11)]
    for k in range (1,11):
        poly = PolynomialFitting(k)
        poly.fit(np.array(trainX), np.array(trainY))
        currLoss = round(poly.loss(np.array(testX), np.array(testY)),2)
        print("Error for polynomial of {deg} degree: {mse}".format(deg=k, mse=currLoss))
        lossesArr.append(currLoss)

    # plot the bar graph:
    fig = px.bar(x=kArr, y=lossesArr)
    fig.update_layout(
        title="MSE loss by degree k",
        xaxis_title="Degree of polynomial fit",
        yaxis_title="MSE Values",
    )
    fig.show()

def israel_model_fitting(xIsrael: pd.DataFrame, yIsrael: pd.Series, deg: int, X: pd.DataFrame):
    """
    creates a bar graph of the Fitted model of Israel tested on other countries
    """
    #fit over Israel:
    poly = PolynomialFitting(deg)
    poly.fit(np.array(xIsrael), np.array(yIsrael))

    #test other countries:
    southAfricaMSE = test_other_countries(X, poly, "South Africa")
    print("South Africa: " + str(southAfricaMSE))
    jordanMSE = test_other_countries(X, poly, "Jordan")
    print("Jordan: " + str(jordanMSE))
    theNetherlandsMSE = test_other_countries(X, poly, "The Netherlands")
    print("The Netherlands: " + str(theNetherlandsMSE))
    allMSE = [southAfricaMSE, jordanMSE, theNetherlandsMSE]
    countryNames = ["South Africa", "Jordan", "The Netherlands"]

    fig = px.bar(x=countryNames, y=allMSE)
    fig.update_layout(
        title="Fitted model of Israel tested on other countries",
        xaxis_title="Countries",
        yaxis_title="MSE Values"
    )
    fig.show()

def test_other_countries(X: pd.DataFrame, polyModel: PolynomialFitting, countryName: str):
    """
    returns the loss of the poly model fitted on Israel and tested on the
    given countryName
    """
    countryDataFrame = X[X['Country'] == countryName]
    xCountry = countryDataFrame['DayOfYear'].to_frame()
    yCountry = countryDataFrame['Temp']

    return round(polyModel.loss(np.array(xCountry), np.array(yCountry)),2)





if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data('City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israelDataFrame = X[X['Country'] == "Israel"]
    explore_country_temp_by_days_of_year(israelDataFrame)
    explore_country_std_by_month(israelDataFrame)

    # Question 3 - Exploring differences between countries
    all_countries_std_and_average_by_month(X)

    # Question 4 - Fitting model for different values of `k`
    dayOfYearIsrael = israelDataFrame['DayOfYear'].to_frame()
    yIsrael = israelDataFrame['Temp']

    dayOfIsraelCopy = dayOfYearIsrael.copy(deep = True)
    fit_over_k_degree(dayOfIsraelCopy, yIsrael)


    # Question 5 - Evaluating fitted model on different countries
    israel_model_fitting(dayOfYearIsrael, yIsrael, 5, X)