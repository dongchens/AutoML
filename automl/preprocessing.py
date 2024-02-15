import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def split_sample(df, frac=0.7, random_state=0):
    """Split a DataFrame into two subsets.

    Args:
        df (DataFrame): The DataFrame to be split.
        frac (float): Fraction of rows to include in the first subset.
        random_state (int): Seed for random number generation.

    Returns:
        tuple: Two DataFrames, the first one containing a fraction `frac` of the original data and the second one containing the remaining data.
    """
    np.random.seed(random_state)
    mask = np.random.rand(len(df)) < frac
    dev = df[mask].copy()
    oos = df[~mask].copy()
    return dev, oos


class Imputation:
    """Handle missing value imputation for numerical and categorical variables."""
    
    def __init__(self, num_vars, cat_vars, num_strategy='median', cat_strategy='UNK'):
        """Initialize an Imputation object.

        Args:
            num_vars (list): List of numerical variable names.
            cat_vars (list): List of categorical variable names.
            num_strategy (str): Strategy for imputing missing values in numerical variables.
            cat_strategy (str): Strategy for imputing missing values in categorical variables.
        """
        self.num_vars = num_vars
        self.cat_vars = cat_vars
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.imputer = dict()

    def fit(self, df):
        """Fit the imputer on the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame.
        """
        for x in self.num_vars:
            if self.num_strategy == 'median':
                self.imputer[x] = df[x].median()
            elif self.num_strategy == 'mean':
                self.imputer[x] = df[x].median()  # Should this be df[x].mean() instead?
            else:
                self.imputer[x] = self.num_strategy
        for x in self.cat_vars:
            if self.cat_strategy == 'UNK':
                self.imputer[x] = self.cat_strategy

    def transform(self, df):
        """Impute missing values in the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame to be transformed.
        """
        for k, v in self.imputer.items():
            df[k] = df[k].fillna(v)


class LabelEncoder:
    """Encode categorical variables into ordinal labels."""
    
    def __init__(self, ordinal=True, unknown='UNK', default=999999):
        """Initialize a LabelEncoder object.

        Args:
            ordinal (bool): Whether to encode categorical variables as ordinal labels.
            unknown (str): Label for unknown values.
            default (int): Default label for missing values.
        """
        self.ordinal = ordinal
        self.default = default
        self.unknown = unknown
        self.labels = dict()
        
    def fit(self, df, cat_vars, target):
        """Fit the encoder on the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame.
            cat_vars (list): List of categorical variable names.
            target (str): Name of the target variable.
        """
        for x in cat_vars:
            agg = df.groupby([x])[target].mean()
            if self.ordinal:
                agg = agg.sort_index()
            values = agg.index
            self.labels[x] = dict(zip(values, range(len(values))))
            self.labels[x][self.unknown] = self.labels[x].get(self.unknown, self.default)

    def transform(self, df):
        """Encode categorical variables in the input DataFrame.

        Args:
            df (DataFrame): The input DataFrame to be transformed.
        """
        for k, v in self.labels.items():
            df[k + '_label'] = df[k].map(v)
            df[k + '_label'] = df[k + '_label'].fillna(self.default)