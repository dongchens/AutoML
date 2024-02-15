import numpy as np
import pandas as pd


def bucket_value(a, func=np.sum, n=10):
    """Compute summary statistics in equal-size buckets.

    Args:
        a (array_like): Input array or DataFrame.
        func (callable): The function to compute summary statistics.
        n (int): Number of buckets.

    Returns:
        list: List of summary statistics computed for each bucket.
    """
    s = round(len(a) / float(n))
    out = [func(a[i - s:i], axis=0) for i in range(s, s * n, s)]
    out.append(func(a[s * (n - 1):], axis=0))
    return out


def capture_value(a, func=np.sum, n=10):
    """Compute cumulative value across buckets.

    Args:
        a (array_like): Input array or DataFrame.
        func (callable): The function to compute cumulative value.
        n (int): Number of buckets.

    Returns:
        ndarray: Cumulative value across buckets.
    """
    a_sum = bucket_value(a, func=func, n=n)
    a_cum = np.cumsum(a_sum)
    return a_cum


def capture_frac(a, func=np.sum, n=10):
    """Compute cumulative fraction across buckets.

    Args:
        a (array_like): Input array or DataFrame.
        func (callable): The function to compute cumulative fraction.
        n (int): Number of buckets.

    Returns:
        ndarray: Cumulative fraction across buckets.
    """
    a_sum = bucket_value(a, func=func, n=n)
    a_cum = np.cumsum(a_sum)
    frac = a_cum / float(sum(a))
    return frac


def gini(y_true, y_score, n_buckets=10):
    """Compute Gini score.

    Args:
        y_true (array_like): True target values.
        y_score (array_like): Predicted scores.
        n_buckets (int): Number of buckets.

    Returns:
        float: Gini score.
    """
    unit = [t[0] for t in sorted(zip(y_true, y_score), key=lambda t: t[1], reverse=True)]
    best = sorted(y_true, reverse=True)
    unit_cap = capture_frac(unit, n=n_buckets)
    best_cap = capture_frac(best, n=n_buckets)
    base = np.linspace(1. / n_buckets, 1., n_buckets)
    return sum(unit_cap - base) / sum(best_cap - base)


def gini_value(y_true, y_score, weight=None, n_buckets=10):
    """Compute weighted Gini score.

    Args:
        y_true (array_like): True target values.
        y_score (array_like): Predicted scores.
        weight (array_like): Sample weights.
        n_buckets (int): Number of buckets.

    Returns:
        tuple: Weighted Gini score and value Gini score.
    """
    if weight is None:
        y_value = np.array(y_true)
    else:
        y_value = np.array(weight) * np.array(y_true)
    sort_by_score = sorted(zip(y_true, y_score, y_value), key=lambda t: t[1], reverse=True)
    unit_cap = capture_frac([t[0] for t in sort_by_score], n=n_buckets)
    value_cap = capture_frac([t[2] for t in sort_by_score], n=n_buckets)
    best_unit_cap = capture_frac(sorted(y_true, reverse=True), n=n_buckets)
    best_value_cap = capture_frac(sorted(y_value, reverse=True), n=n_buckets)
    base = np.linspace(1. / n_buckets, 1., n_buckets)
    return (sum(unit_cap - base) / sum(best_unit_cap - base),
            sum(value_cap - base) / sum(best_value_cap - base))


def lift_table(y_true, y_score, weight=None, n_buckets=10):
    """Generate a lifting table.

    Args:
        y_true (array_like): True target values.
        y_score (array_like): Predicted scores.
        weight (array_like): Sample weights.
        n_buckets (int): Number of buckets.

    Returns:
        DataFrame: Lifting table.
    """
    if weight is None:
        y_value = np.array(y_true)
    else:
        y_value = np.array(weight) * np.array(y_true)
    sort_by_score = sorted(zip(y_true, y_score, y_value), key=lambda t: t[1], reverse=True)
    unit_cap = capture_value([t[0] for t in sort_by_score], n=n_buckets)
    value_cap = capture_value([t[2] for t in sort_by_score], n=n_buckets)
    counter = capture_value(np.ones(len(y_true)), n=n_buckets)
    base = np.linspace(1. / n_buckets, 1., n_buckets)
    lift = np.c_[base, counter, unit_cap, unit_cap / counter, unit_cap / float(sum(unit_cap)),
                 value_cap / float(sum(value_cap))]
    return pd.DataFrame(lift, columns=['Quantile', 'Count', 'Units', 'Rate', 'Unit %', 'Value %'])



def fpr_table(y_true, y_score, weight=None, start=1, end=10):
    """Generate a false positive rate (FPR) table.

    Args:
        y_true (array_like): True target values.
        y_score (array_like): Predicted scores.
        weight (array_like): Sample weights.
        start (int): Start quantile.
        end (int): End quantile.

    Returns:
        list: FPR table.
    """
    if weight is None:
        y_value = y_true
    else:
        y_value = np.array(y_true) * np.array(weight)
    sort_by_score = sorted(zip(y_true, y_score, y_value), key=lambda t: t[1], reverse=True)
    out = []
    unit, value = 0, 0
    unit_sum, value_sum = float(sum(y_true)), float(sum(y_value))
    for i in range(len(y_true)):
        unit += sort_by_score[i][0]
        value += sort_by_score[i][2]
        if ((i + 1) % unit != 0) or ((i + 1 - unit) / unit < start):
            continue
        fpr = (i + 1 - unit) / float(unit)
        if fpr > end:
            break
        else:
            out.append([i + 1, unit, fpr, unit / unit_sum, value / value_sum])
    return out


def _bucket(x, cuts):
    """Helper function to assign values to buckets."""
    for i in range(1, len(cuts)):
        if cuts[i - 1] < x <= cuts[i]:
            return i
    return 'NAN'


def rankplot(df, var, target, bins=10, show=False):
    """Generate a rank plot.

    Args:
        df (DataFrame): The input DataFrame.
        var (str): Name of the variable.
        target (str): Name of the target variable.
        bins (int): Number of bins for grouping the data.
        show (bool): Whether to display the plot.

    Returns:
        DataFrame: DataFrame containing aggregated values for each bin.
    """
    cuts = df[var].quantile(np.linspace(0, 1, bins + 1)).values
    cuts[0] = float('-inf')
    cuts[-1] = float('inf')
    df['bucket'] = df[var].map(lambda x: _bucket(x, cuts))
    cols = [var, target]
    rp = df.groupby(['bucket'])[cols].mean().reset_index().sort_values([var])
    if show:
        plt.plot(rp[var], rp[target])
        plt.xlabel(var)
        plt.ylabel(target)
    return rp


def logloss(y_true, y_score, eps=1e-15):
    """Compute logloss.

    Args:
        y_true (array_like): True target values.
        y_score (array_like): Predicted scores.
        eps (float): Small value to avoid log(0).

    Returns:
        float: Logloss value.
    """
    loss = 0.
    for i in range(len(y_true)):
        loss += y_true[i] * np.log(y_score[i] + eps) + (1 - y_true[i]) * np.log(1 - y_score[i] + eps)
    return -loss / len(y_true)


import numpy as np

def get_bucket(v, cutoff):
    """Return the index of the bucket for the given value.

    Args:
        v (float): The value to find the bucket for.
        cutoff (array_like): The cutoff points defining the buckets.

    Returns:
        int: The index of the bucket where the value falls.
    """
    cutoff[0], cutoff[-1] = -np.inf, np.inf
    for i in range(1, len(cutoff)):
        if cutoff[i] < v <= cutoff[i+1]:
            return i
    return len(cutoff)


def get_dist(alist, cutoff, func=get_bucket, n=10):
    """Compute the distribution of a numerical list.

    Args:
        alist (array_like): The list of numerical values.
        cutoff (array_like): The cutoff points defining the buckets.
        func (callable): The function to get the bucket index.
        n (int): The number of buckets.

    Returns:
        ndarray: The distribution of the numerical list.
    """
    dist = np.zeros(n+1)
    for v in alist:
        if v != v:
            dist[-1] += 1
        else:
            dist[get_bucket(v, cutoff)] += 1
    return dist / len(alist)


def get_psi(benchmark_dist, test_dist, eps=1e-15):
    """Calculate the Population Stability Index (PSI) based on the distribution.

    Args:
        benchmark_dist (array_like): The distribution of the benchmark population.
        test_dist (array_like): The distribution of the test population.
        eps (float): Small value to avoid division by zero.

    Returns:
        ndarray: The PSI values for each bucket.
    """
    benchmark_dist = np.array(benchmark_dist) + eps
    test_dist = np.array(test_dist) + eps
    return (benchmark_dist - test_dist) * np.log(benchmark_dist / test_dist)


def compute_psi(benchmark, test, n=10):
    """Calculate the Population Stability Index (PSI).

    Args:
        benchmark (array_like): The benchmark population data.
        test (array_like): The test population data.
        n (int): The number of buckets.

    Returns:
        ndarray: The PSI values for each bucket.
    """
    cutoff = np.quantile(benchmark, np.linspace(0,1,n+1))
    benchmark_dist = get_dist(benchmark, cutoff, get_bucket, n)
    test_dist = get_dist(test, cutoff, get_bucket, n)
    return get_psi(benchmark_dist, test_dist)


def get_cat_dist(alist):
    """Get the distribution for categorical variables.

    Args:
        alist (array_like): The list of categorical values.

    Returns:
        dict: The distribution of categorical values.
    """
    d = dict()
    for x in alist:
        d[x] = d.get(x, 0) + 1
    tot = float(len(alist))
    return {k: v / tot for k, v in d.items()}


def compute_cat_psi(benchmark, test, eps=1e-15):
    """Calculate the Population Stability Index (PSI) for categorical variables.

    Args:
        benchmark (array_like): The benchmark population data.
        test (array_like): The test population data.
        eps (float): Small value to avoid division by zero.

    Returns:
        float: The PSI value for categorical variables.
    """
    benchmark_dist = get_cat_dist(benchmark)
    test_dist = get_cat_dist(test)
    out = 0.
    for k in benchmark_dist.keys():
        out += (benchmark_dist[k] - test_dist[k]) * np.log((benchmark_dist[k] + eps) / (test_dist[k] + eps))
    return out
