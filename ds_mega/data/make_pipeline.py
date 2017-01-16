import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ds_mega.data.extractors import *
from ds_mega.data.dict import experian_mosaic_dict, mosaic_missing_val, experian_income_dict
from ds_mega.data.dataframe_feature_union import DataFrameFeatureUnion


def make_pipeline(df):
    pipeline_yb = Pipeline([
        ('YearBuilt', YearBuiltExtractor()),
        ('scaler', StandardScaler())
    ])

    pipeline_sqft = Pipeline([
        ('Sqft', DataFrameColumnExtractor('SquareFootage')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    pipeline_lot = Pipeline([
        ('LotSize', DataFrameColumnExtractor('LotSize')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    pipeline_est_val = Pipeline([
        ('EstimatedValue', DataFrameColumnExtractor('EstimatedValue')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    pipeline_tav = Pipeline([
        ('TaxAssessedValue', DataFrameColumnExtractor('TaxAssessedValue')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    pipeline_tiv = Pipeline([
        ('TaxImprovementValue', DataFrameColumnExtractor('TaxImprovementValue')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    pipeline_tlv = Pipeline([
        ('TaxLandValue', DataFrameColumnExtractor('TaxLandValue')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    pipeline_tip = Pipeline([
        ('TaxImprovementPercent', DataFrameColumnExtractor('TaxImprovementPercent')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    pipeline_tax = Pipeline([
        ('TaxAmount', DataFrameColumnExtractor('TaxAmount')),
        ('df', DataFrameImputer()),
        ('scaler', StandardScaler()),
        ('minmaxlimit', StandardScalerLimitTransformer())
    ])

    feature_union = DataFrameFeatureUnion([
        ('year_built', pipeline_yb),
        ('sqft', pipeline_sqft),
        ('lot', pipeline_lot),

        ('tax', pipeline_tax),
        ('tav', pipeline_tav),
        ('tiv', pipeline_tiv),
        ('tlv', pipeline_tlv),
        ('tip', pipeline_tip),

        ('est', pipeline_est_val),
    ])

    X = feature_union.fit_transform(df)

    return X
