import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.base import BaseEstimator
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


def tune_model_random(model: BaseEstimator, X: np.ndarray, y: np.ndarray, param_grid: dict) -> RandomizedSearchCV:
    """Tune hyperparameters of a model using random search.

    This function performs hyperparameter tuning on a given model using random search
    and evaluates the performance using cross-validation with ROC AUC score.

    Args:
        model (BaseEstimator): The model to be tuned.
        X (ndarray): The feature matrix.
        y (ndarray): The target array.
        param_grid (dict): The parameter grid to search over.

    Returns:
        RandomizedSearchCV: A RandomizedSearchCV object containing the tuned model.
    """
    cv = KFold(n_splits=4, shuffle=True, random_state=0)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        return_train_score=True,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2)
    search.fit(X, y)
    return search

# param_grid ={'learning_rate':[0.05, 0.1], 
#              'num_leaves': [40, 50, 60], 
#              'reg_alpha': [0, 0.1, 1],
#              'reg_lambda': [0, 0.1, 1]}

# clf = lgb.LGBMClassifier(random_state=100, silent=True, metric='None', n_jobs=4, 
#                          max_depth=-1, n_estimators=100)

# result = tune_model_random(clf, dev[feats], dev['loan_status'], param_grid)
# result.best_params_, result.best_score_
