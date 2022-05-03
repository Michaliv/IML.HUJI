from typing import Tuple
from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd


def _preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Cast dates columns to be real dates
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])
    if 'cancellation_datetime' in df.columns:
        df['cancellation_datetime'] = pd.to_datetime(df['cancellation_datetime'])

    # Remove unreasonable data
    df = df[df['booking_datetime'] <= df['checkin_date']]
    df = df[df['checkout_date'] > df['checkin_date']]
    if 'cancellation_datetime' in df.columns:
        df.drop(df[~(df['booking_datetime'] < df['cancellation_datetime']) &
                (df['cancellation_datetime'] < df['checkout_date'])].index)

    # Add column indicate whether the origin country booking differ from hotel country.
    df['foreign_booking'] = df['origin_country_code'] == df[
        'hotel_country_code']

    # Cast bool features to 0/1
    df['foreign_booking'] = df['foreign_booking'].astype(int)
    df['is_user_logged_in'] = df['is_user_logged_in'].astype(int)

    # Add dummies instead of hotel id and accommodation_type_name, hotel_city_code,
    # hotel_area_code
    # dummies = pd.get_dummies(df['hotel_id'])
    # df = pd.concat([df, dummies], axis=1)
    # dummies = pd.get_dummies(df['accommadation_type_name'])
    # df = pd.concat([df, dummies], axis=1)
    # dummies = pd.get_dummies(df['hotel_city_code'])
    # df = pd.concat([df, dummies], axis=1)
    # dummies = pd.get_dummies(df['hotel_area_code'])
    # df = pd.concat([df, dummies], axis=1)



    # Cast charge_option feature
    df['charge_option'] = df['charge_option'].replace(
        {'Pay Now': 15, 'Pay Later': 25,
         'Pay at Check-in': 35})



    accomodation_types = df['accommadation_type_name'].unique()
    map_of_types = {'Hotel': 10, 'UNKNOWN': 0, 'Boat / Cruise':0, 'Resort' : 10, 'Serviced Apartment' : 8, 'Guest House / Bed & Breakfast': 8,
                    'Hostel': 10, 'Capsule Hotel': 6, 'Apartment': 7, 'Bungalow': 5, 'Motel':10,
                    'Ryokan': 3, 'Tent': 3, 'Resort Villa':10, 'Home':0, 'Love Hotel':4, 'Holiday Park / Caravan Park':5,
                    'Private Villa': 10, 'Inn':4, 'Lodge':3, 'Homestay':4, 'Chalet':5 }
    df['accommadation_type_name'] = df['accommadation_type_name'].replace(map_of_types)

    # hotel_types = df['hotel_id'].unique()
    # map_of_hotels = {}
    # for ac in range(len(hotel_types)):
    #     map_of_hotels[hotel_types[ac]] = ac % 134
    # df['hotel_id'] = df['hotel_id'].replace(map_of_hotels)

    # Handle dates:

    if 'cancellation_datetime' in df.columns:
        df['cancellation_year'] = pd.DatetimeIndex(
            df['cancellation_datetime']).year
        df['cancellation_day_of_year'] = df['cancellation_datetime'].dt.day_of_year

    df['booking_year'] = pd.DatetimeIndex(df['booking_datetime']).year
    df['booking_day_of_year'] = df['booking_datetime'].dt.day_of_year

    df['checkin_day_of_year'] = df['checkin_date'].dt.day_of_year

    df['checkout_day_of_year'] = df['checkout_date'].dt.day_of_year

    counts = pd.value_counts(df['cancellation_policy_code'])
    mask = df['cancellation_policy_code'].isin(
        counts[counts > counts[10]].index)
    # dummies = pd.get_dummies(df['cancellation_policy_code'][mask])
    df['cancellation_policy_code'].iloc[~mask] = '-'
    dummies = pd.get_dummies(df['cancellation_policy_code'])

    df = pd.concat([df, dummies], axis=1)



    if 'cancellation_datetime' in df.columns:
        df["cancellation_datetime"] = df["cancellation_datetime"].fillna(0)
        df[df["cancellation_datetime"] != 0] = 1
        df['cancellation_day_of_year'] =  df['cancellation_day_of_year'].fillna(0)
        df['cancellation_year'] = df['cancellation_year'].fillna(0)
        df['cancellation_year'] = df['cancellation_year'].replace(
        {2017: 0, 2018: 2, 2019: 1})

    df['booking_year'] = df['booking_year'].replace({2017: 0, 2018: 1})

    response = df['cancellation_datetime'] if 'cancellation_datetime' in df.columns else pd.DataFrame([0] * df.shape[0])
    # response = response.astype('int')

    accomodation_types = df['accommadation_type_name'].unique()

    # remove those features:
    to_drop = ["h_booking_id", "hotel_id",
               'hotel_chain_code', 'hotel_brand_code', 'hotel_live_date',
               'h_customer_id', 'customer_nationality',
               'guest_nationality_country_name', 'no_of_adults',
               'no_of_children',
               'no_of_extra_bed',
               'original_payment_method', 'original_payment_type',
                'hotel_city_code', 'hotel_area_code',
               'cancellation_policy_code', 'hotel_country_code',
               'origin_country_code', 'original_payment_currency', 'request_nonesmoke',
               'request_latecheckin', 'request_highfloor', 'request_earlycheckin',
               'request_largebed', 'request_twinbeds', 'request_airport',
               'language']

    to_drop.extend(['booking_datetime', 'checkin_date', 'checkout_date'])
    if 'cancellation_datetime' in df.columns:
        to_drop.extend(['cancellation_datetime'])

    df = df.drop(to_drop, axis=1)
    return df, response


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    df = pd.read_csv(filename).drop_duplicates()
    df, y = _preprocess(df)
    return df, y


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(
        filename,
        index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data(
        '../datasets/agoda_cancellation_train.csv')

    train_X, train_y, ignore_X1, ignore_y1 = split_train_test(df,
                                                        cancellation_labels, train_proportion=1)

    week2_test, _ = load_data('../datasets/test_set_week_3.csv')
    week2_test = week2_test.reindex(columns=train_X.columns, fill_value=0)
    week2_labels = pd.read_csv('../datasets/test_set_week_3_labels.csv')
    week2_labels = week2_labels['h_booking_id|label'].str.split(pat='|')
    week2_labels = np.array(week2_labels.to_list())[:, 1]
    week2_labels = pd.DataFrame(week2_labels).astype(int)

    # week2_test = week2_test.loc[:,
    #              ~week2_test.columns.duplicated(keep='first')]
    # week2_test.reset_index(inplace=True)
    # week2_test = week2_test.reindex(columns=df.columns, fill_value=0)



    train_X = train_X.fillna(0).to_numpy()
    train_y = train_y.fillna(0).to_numpy()

    # Fit model over data
    estimator = AgodaCancellationEstimator()
    estimator.fit(train_X, train_y)


    # Store model predictions over test set
    df_test_set, cancellation_labels_test_set = load_data("test_set_week_3.csv")

    df_test_set = df_test_set.reindex(columns= df.columns, fill_value=0)
    test_X, test_y, ignore_X2, ignore_y2 = split_train_test(df_test_set, cancellation_labels_test_set, train_proportion=1)
    test_X = test_X.fillna(0).to_numpy()

    print(estimator.loss(week2_test.to_numpy(), week2_labels.to_numpy()))


    evaluate_and_export(estimator, test_X, "207182452_208586537_318929742.csv")
