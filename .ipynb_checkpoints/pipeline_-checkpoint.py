import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import roc_curve
from datetime import timedelta
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV


def create_target_data(df, current_date):
    '''
    Create target feature

    Parameters:
    ------------------
    df           : dataframe with datetime column 'last_trip_date'
    current_date : the current date to predict churn from

    Return:
    ------------------
    np.array of whether the user has "churned" or not with 1 being churned
    and 0 being not churning
    '''
    one_month_back = current_date - DateOffset(months=1)
    return (df['last_trip_date'] < one_month_back).astype(int)


class CreateDummies(BaseEstimator, TransformerMixin):
    '''
    Creates dummy variables from two columns: ['city', 'phone']
    '''

    def fit(self, X, y):
        self.phone_names = [x for x in X['phone'].unique() if not pd.isnull(x)]
        self.city_names = [x for x in X['city'].unique() if not pd.isnull(x)]

        return self

    def transform(self, X):
        for phone in self.phone_names:
            X[phone] = (X['phone'] == phone).astype(int)
        for city in self.city_names:
            X[city] = (X['city'] == city).astype(int)
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    '''
    Drop given unneeded columns

    Parameters:
    ------------------------
    columns : list strings of column names
    '''

    def __init__(self, drop_columns=[]):
        self.drop_columns = drop_columns

    def get_params(self, **kwargs):
        return {'drop_columns': self.drop_columns}

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.drop(self.drop_columns, axis=1)


class BooleanField(BaseEstimator, TransformerMixin):
    '''
    Set boolean fields to 1 and 0 ('luxury_car_user')
    '''

    def fit(self, X, y):
        return self

    def transform(self, X):
        X['luxury_car_user'] = X['luxury_car_user'].astype(int)
        return X


class DriverDummies(BaseEstimator, TransformerMixin):
    '''
    Create dummy variables for binned values of ['avg_rating_of_driver', 'avg_rating_by_driver']
    Column bins are (rating < 4.3), (rating >= 4.3 and rating < 4.8), and (rating >= 4.8)
    '''

    def fit(self, X, y):
        return self

    def transform(self, X):
        for base, col in [('of_driver_', 'avg_rating_of_driver'), ('by_driver_', 'avg_rating_by_driver')]:
            X[base+'low'] = (X[col] < 4.3).astype(int)
            X[base+'med'] = ((X[col] >= 4.3) & (X[col] <= 4.8)).astype(int)
            X[base+'high'] = (X[col] > 4.8).astype(int)
        return X


def pipeline_grid_search(X_train, y_train, pipeline, params, scoring):
    '''
    Runs a grid search on the given pipeline with the given params

    Parameters:
    --------------------------
    X_train  : 2 dimensional array-like
    y_train  : 1 dimensional array-like
    pipeline : Sklearn pipeline object
    params   : dictionary of pipeline parameters

    Returns:
    --------------------------
    the resulting GridSearchCV object
    '''
    grid = GridSearchCV(pipeline, params, scoring=scoring, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)

    return grid


class CreateFeatures(BaseEstimator, TransformerMixin):
    '''
    creates new column estimating users who only rode one time
    creates new column stating if avg_rating_of_driver is NaN
    '''

    def fit(self, X, y):
        return self

    def transform(self, X):
        X['one_time_user'] = (X['avg_rating_by_driver'] % 1 == 0) & (X['avg_rating_of_driver'] % 1 == 0) & (
            X['trips_in_first_30_days'] < 2) & (X['weekday_pct'] % 100 == 0) & (X['surge_pct'] % 100 == 0)
        X['one_time_user'] = X['one_time_user'].astype(int)
        X['no_driver_rating'] = X['avg_rating_of_driver'].isnull().astype(int)
        return X


class ImputeValues(BaseEstimator, TransformerMixin):
    '''
    Imputes avg_rating_of_driver with the mean if the value is NaN
    Imputes avg_rating_by_driver with the mean if the value is NaN
    '''

    def fit(self, X, y):
        self.driver_avg = X['avg_rating_of_driver'].mean()
        self.rider_avg = X['avg_rating_by_driver'].mean()
        return self

    def transform(self, X):
        X['avg_rating_of_driver'] = X['avg_rating_of_driver'].apply(
            lambda x: self.driver_avg if np.isnan(x) else x)
        X['avg_rating_by_driver'] = X['avg_rating_by_driver'].apply(
            lambda x: self.rider_avg if np.isnan(x) else x)
        return X


def plot_roc(y_true, y_pred, label=''):
    '''
    Plots the ROC curve of the given data

    Parameters:
    ------------------
    y_true : 1d array like of length n
    y_pred : 1d array like of length n
    label  : string of what to label the plotted curve
    '''

    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=None,
                                    sample_weight=None, drop_intermediate=True)

    plt.plot(fpr, tpr, label=label)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend()
