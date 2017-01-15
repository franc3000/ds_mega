import pandas as pd
from sklearn.pipeline import FeatureUnion, _name_estimators


class DataFrameFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        out = super(DataFrameFeatureUnion, self).fit_transform(X, y, **fit_params)
        cols = [name for name, trans in self.transformer_list]
        return pd.DataFrame(out, columns=cols)


def make_dataframeunion(*transformers):
    """
    DO NOT USE, drops the names of the columns

    Construct a DataFrameFeatureUnion from the given transformers.

    This is a shorthand for the FeatureUnion constructor; it does not require,
    and does not permit, naming the transformers. Instead, they will be given
    names automatically based on their types. It also does not allow weighting.
    """
    return DataFrameFeatureUnion(_name_estimators(transformers))
