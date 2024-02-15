import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_model(model, dev, oos, feats, target):
    """Build and evaluate a model.

    Args:
        model: The model object.
        dev (DataFrame): Development dataset.
        oos (DataFrame): Out-of-sample dataset.
        feats (list): List of feature names.
        target (str): Name of the target variable.

    Returns:
        model: The trained model object.
    """
    model.fit(dev[feats], dev[target])
    for ds in [dev, oos]:
        ds['score'] = model.predict(ds[feats])
        print(roc_auc_score(ds[target], ds['score']))
    return model