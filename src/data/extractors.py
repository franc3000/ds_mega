import logging

import numpy as np
import pandas as pd
from dict import experian_mosaic_dict, mosaic_missing_val, experian_income_dict
from sklearn.base import TransformerMixin, BaseEstimator

"""

Feature Extractors

Don't use pandas inplace=True for fillna or other functions.
http://stackoverflow.com/questions/21463589/pandas-chained-assignments/21463854#21463854

"""


class AgeExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, column='combined_age'):
        self.column = column

    def get_feature_names(self):
        return 'combined_age'

    def transform(self, data):
        """
        Income column as float, and cleaned
        Any row with incom = NULL set to median
        """
        logging.getLogger('AgeExtractor').info(self.column)

        # get col
        col = data[self.column]

        # expand: bool, default False
        # If True, return DataFrame.
        # If False, return Series / Index / DataFrame
        age = col.str.extract('(\d+)', expand=False)

        # convert to float64
        age = age.astype(float)

        # boundaries
        m = age.median()
        if not np.isnan(m):
            age = age.fillna(m)

        # single feature, must reshape to ndarray
        return age.values.reshape(-1, 1)

    def fit(self, *_):
        return self


class IncomeExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, column='est_household_income_v5'):
        self.column = column

    def get_feature_names(self):
        return 'est_household_income_v5'

    def transform(self, data):
        """
        Income column as float, and cleaned
        Any row with incom = NULL set to median
        """
        logging.getLogger('IncomeExtractor').info(self.column)

        # get col
        col = data[self.column].replace([c for c in experian_income_dict.keys()],
                                        [c for c in experian_income_dict.values()])

        # convert to float64
        col = col.astype(float)

        # boundaries
        m = col.median()
        if not np.isnan(m):
            col = col.fillna(m)

        # single feature, must reshape to ndarray
        return col.values.reshape(-1, 1)

    def fit(self, *_):
        return self


class YearBuiltExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, column='YearBuilt'):
        self.column = column

    def get_feature_names(self):
        return 'year_built'

    def transform(self, data):
        """
        YearBuilt column as float, and cleaned
        Any row with YearBuilt < 1900 set to median
        """
        logging.getLogger('YearBuiltExtractor').info(self.column)

        # get col as float
        col = data[self.column].astype(float)

        # boundaries, but only if valid median
        m = col[col > 1850].median()
        if not np.isnan(m):
            col[col < 1900] = m
            col = col.fillna(m)

        # single feature, must reshape to ndarray
        return col.values.reshape(-1, 1)

    def fit(self, *_):
        return self


class ShortLOOExtractor(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.cols = ['TransferPrice', 'LengthOfOwnership', 'SquareFootage']

    def get_feature_names(self):
        return 'short_loo'

    def transform(self, data):
        """
        special condition
        """
        logging.getLogger('ShortLOOExtractor').info('ShortLOOExtractor')

        for c in self.cols:
            if c not in data.columns.values:
                raise ValueError('{} not in df'.format(c))

        # get col
        a = data['TransferPrice'] < 1000
        b = data['LengthOfOwnership'] < 2
        c = data['SquareFootage'] > 0
        col = a & b & c

        # convert to int
        col = col.astype(int)

        # single feature, must reshape to ndarray
        return col.values.reshape(-1, 1)

    def fit(self, *_):
        return self


class LOOExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, column='LengthOfOwnership'):
        self.column = column

    def get_feature_names(self):
        return 'loo'

    def transform(self, data):
        """
        YearBuilt column as float, and cleaned
        Any row with YearBuilt < 1900 set to median
        """
        logging.getLogger('LOOExtractor').info(self.column)

        # get col
        col = data[self.column]

        # convert to float64
        col = col.astype(float)

        # boundaries
        col[col == 0] = 20
        col = col.fillna(20)

        # single feature, must reshape to ndarray
        return col.values.reshape(-1, 1)

    def fit(self, *_):
        return self


class MosaicExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, column='mosaic_household'):
        self.column = column

    def get_feature_names(self):
        return 'mosaic_household'

    def transform(self, data):
        """
        YearBuilt column as float, and cleaned
        Any row with YearBuilt < 1900 set to median
        """
        logging.getLogger('MosaicExtractor').info(self.column)

        # mosaic key-value translation to int values
        col = data[self.column].replace([c for c in experian_mosaic_dict.keys()],
                                        [c for c in experian_mosaic_dict.values()])

        # NA --> mosaic missing value
        col = col.fillna(mosaic_missing_val)

        # single feature, must reshape to ndarray
        return col.values.reshape(-1, 1)

    def fit(self, *_):
        return self


class ConvertToDataFrame(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def transform(self, data):
        return pd.DataFrame(data)

    def fit(self, *_):
        return self


class ColumnExtractor(TransformerMixin, BaseEstimator):
    """
    returns a numpy.ndarray, given a DataFrame
    """

    def __init__(self, column):
        self.column = column

    def transform(self, df):
        logging.getLogger('ColumnExtractor').info(self.column)

        col = df[self.column]
        # if all values are NaN, then replace with 0
        if col.isnull().all():
            col = col.fillna(0)

        return col.values.reshape(-1, 1)

    def fit(self, *_):
        return self


class DataFrameColumnExtractor(TransformerMixin, BaseEstimator):
    """
    returns a DataFrame, given a DataFrame
    """

    def __init__(self, column):
        self.column = column

    def transform(self, df):
        logging.getLogger('DFCE').info(self.column)

        df_col = df[[self.column]]

        # if all values are NaN, then replace with 0
        for c in df_col.columns:
            if df_col[c].isnull().all():
                df_col[c] = df_col[c].fillna(0)

        return df_col

    def fit(self, *_):
        return self


class DataFrameImputer(TransformerMixin, BaseEstimator):
    """
    Impute missing values.
    Columns of dtype object are imputed with the most frequent val in col.
    Columns of other types are imputed with mean of column.
    """

    def __init__(self):
        self.fill = 0
        pass

    def fit(self, df, y=None):
        # if not df and not series, error
        if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
            raise ValueError('var `df` type is not a DataFrame or Series, it is a {}'.format(type(df)))
        self.fill = pd.Series([df[c].median(skipna=True) for c in df], index=df.columns)
        return self

    def transform(self, df, y=None):
        # logging.getLogger('DataFrameColumnExtractor').info('transform')
        return df.fillna(self.fill)


class StandardScalerLimitTransformer(TransformerMixin, BaseEstimator):
    """
    Replaces extreme values with the min and max allowed values
    """

    def __init__(self, min_value=-3, max_value=3):
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # logging.getLogger('SSLimitTransformer').info('transform')
        X[X < self.min_value] = self.min_value
        X[X > self.max_value] = self.max_value
        return X
