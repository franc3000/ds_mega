import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler

from src.data.extractors import YearBuiltExtractor


dtype = {'fips': str}

df = pd.read_csv('../data/raw/tmp_ds_cluster.csv.gz', dtype=dtype)

df.head()

df.describe().transpose().round(1)

df_01073 = df.loc[df['fips']=='01073']

print len(df)
print len(df_01073)

df['YearBuilt'].hist()

pipeline_yb = Pipeline([
    ('year_built', YearBuiltExtractor())
])

