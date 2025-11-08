from aif360.sklearn.metrics import statistical_parity_difference
import pandas as pd
import numpy as np
from fairlearn.metrics import selection_rate, MetricFrame
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import make_scorer
from functools import partial
import math


def convert_to_numpy(array):
    # convert to numpy array
    if type(array) == list:
        array = np.array(array)
    if isinstance(array, pd.DataFrame):
        array = array.to_numpy()
    if isinstance(array, pd.Series):
        array = array.to_numpy()
    return array

def check_y_array(array):
    # convert to numpy array
    array = convert_to_numpy(array)

    # get max value
    if len(array.shape) > 1:
        array = np.max(array, axis=1)

    return array

def check_s_array(array):
    # convert to numpy array
    array = convert_to_numpy(array)

    # if array is one-hot encoded, de-encode it
    if len(array.shape) > 1:
        array = array[:, 0]

    return array

def discrimination(y, y_pred, S):
    # check input arrays
    y_pred = check_y_array(y_pred)
    y = check_y_array(y)
    s = check_s_array(S)

    # we set a hard th here
    y_pred = np.array([1 if i > 0.5 else 0 for i in y_pred])

    mf = MetricFrame(metrics={'selection_rate': selection_rate}, y_true=y, y_pred=y_pred,
                     sensitive_features=s)
    discr = abs(mf.by_group.iloc[0, 0] - mf.by_group.iloc[1, 0])
    return discr

def delta(y, y_pred, S):
    # check input arrays
    y_pred = check_y_array(y_pred)
    s = check_s_array(S)

    acc = roc_auc_score(y, y_pred)
    discr = discrimination(y, y_pred, s)
    return acc - discr

def group_pairwise_accuracy(y, y_pred, S):
    """Returns the group-dependent pairwise accuracies.

    Returns the group-dependent pairwise accuracies Acc_{G_i > G_j} for each pair
    of groups G_i \in {0, 1} and G_j \in {0, 1}.

    Args:
      prediction_diffs: NumPy array of shape (#num_pairs,) containing the
                        differences in scores for each ordered pair of examples.
      paired_groups: NumPy array of shape (#num_pairs, 2) containing the protected
                     groups for the better and worse example in each pair.

    Returns:
      A NumPy array of shape (2, 2) containing the pairwise accuracies, where the
      ij-th entry contains Acc_{G_i > G_j}.
    """

    # check input arrays
    y_pred = check_y_array(y_pred)
    s = check_s_array(S)

    scores = np.reshape(y_pred, len(y_pred))
    df = pd.DataFrame()
    df = df.assign(scores=scores, labels=y, groups=s, merge_key=0)
    df = df.merge(df.copy(), on="merge_key", how="outer", suffixes=("_high", "_low"))
    df = df[df.labels_high > df.labels_low]

    paired_scores = np.stack([df.scores_high.values, df.scores_low.values], axis=1)
    paired_groups = np.stack([df.groups_high.values, df.groups_low.values], axis=1)

    # Baseline
    prediction_diffs = paired_scores[:, 0] - paired_scores[:, 1]

    accuracy_matrix = np.zeros((2, 2))
    for group_high in [0, 1]:
        for group_low in [0, 1]:
            # Predicate for pairs where the better example is from group_high
            # and the worse example is from group_low.
            predicate = ((paired_groups[:, 0] == group_high) &
                         (paired_groups[:, 1] == group_low))
            # Parwise accuracy Acc_{group_high > group_low}.
            accuracy_matrix[group_high][group_low] = (
                    np.mean(prediction_diffs[predicate] > 0) +
                    0.5 * np.mean(prediction_diffs[predicate] == 0))
    return abs(accuracy_matrix[0][1] - accuracy_matrix[1][0])

def rND(y, y_pred, S, step=10, start=10, protected_group_idx=1, non_protected_group_idx=0):
    # check input arrays
    prediction = check_y_array(y_pred)
    s = check_s_array(S)

    # we don't want to have uniqual size
    if len(prediction) != len(s):
        raise AssertionError(
            'len of prediction ' + str(len(prediction)) + ' and s ' + str(len(s)) + ' are uniqual'
        )
    unique, counts = np.unique(s, return_counts=True)
    count_dict_all = dict(zip(unique, counts))
    if len(unique) != 2:
        raise AssertionError(
            'array s contains more than 2 classes.'
        )
    keys = [protected_group_idx, non_protected_group_idx]
    for key in keys:
        if key not in count_dict_all:
            count_dict_all[key] = 0

    sorted_idx = np.argsort(np.array(prediction))[::-1]
    sorted_s = np.array(s[sorted_idx])

    # a fake sorted list of s which gives the worst possible result, used for regularization purposes
    # it is maximally discriminative, having all non-protected individuals first and then the others.
    fake_horrible_s = np.hstack(([non_protected_group_idx for i in range(count_dict_all[non_protected_group_idx])],
                                 [protected_group_idx for i in range(count_dict_all[protected_group_idx])]))

    fake_horrible_s_2 = np.hstack(([protected_group_idx for i in range(count_dict_all[protected_group_idx])],
                                 [non_protected_group_idx for i in range(count_dict_all[non_protected_group_idx])]))

    rnd = 0
    max_rnd = 0
    max_rnd_2 = 0

    for i in range(start, len(s), step):
        unique, counts = np.unique(sorted_s[:i], return_counts=True)
        count_dict_top_i = dict(zip(unique, counts))

        unique, counts = np.unique(fake_horrible_s[:i], return_counts=True)
        count_dict_reg = dict(zip(unique, counts))

        unique_2, counts_2 = np.unique(fake_horrible_s_2[:i], return_counts=True)
        count_dict_reg_2 = dict(zip(unique_2, counts_2))

        keys = [protected_group_idx, non_protected_group_idx]
        for key in keys:
            if key not in count_dict_reg:
                count_dict_reg[key] = 0
            if key not in count_dict_top_i:
                count_dict_top_i[key] = 0
        rnd += 1/np.log2(i) * np.abs(count_dict_top_i[protected_group_idx]/i - count_dict_all[protected_group_idx]/len(s))
        max_rnd += 1/np.log2(i) * np.abs(count_dict_reg[protected_group_idx]/i - count_dict_all[protected_group_idx]/len(s))
        max_rnd_2 += 1/np.log2(i) * np.abs(count_dict_reg_2[protected_group_idx]/i - count_dict_all[protected_group_idx]/len(s))

    if max_rnd_2 > max_rnd:
        max_rnd = max_rnd_2

    return rnd/max_rnd

def discrimination_score(pred, s, th):
    # check input arrays
    pred = check_y_array(pred)
    s = check_s_array(s)

    y_pred = np.array([1 if i > th else 0 for i in pred])
    if len(y_pred[s==0]) == 0 or len(y_pred[s==1]) == 0:
        return "NaN"
    return np.abs(sum(y_pred[s==1])/len(y_pred[s==1]) - sum(y_pred[s==0])/len(y_pred[s==0]))

def auc_discrimination(y, y_pred, S):
    # check input arrays
    prediction = check_y_array(y_pred)
    s = check_s_array(S)

    fpr, tpr, thresholds = roc_curve(y, y_pred)

    # in case we have an inf threshold or devision is inf we skip this one
    y_filtered = []
    filtered_thresholds = []
    for th in thresholds:
        if math.isinf(th): continue
        value = discrimination_score(y_pred, s, th)
        if value == "NaN": continue
        y_filtered.append(value)
        filtered_thresholds.append(th)
    return np.abs(np.trapz(y_filtered, filtered_thresholds))

def statistical_parity_diff(y, y_pred, S):
    # check input arrays
    y_pred = check_y_array(y_pred)
    y = check_y_array(y)
    # we set a hard th here
    y_pred = np.array([1 if i > 0.5 else 0 for i in y_pred])

    return statistical_parity_difference(pd.DataFrame(y), pd.DataFrame(y_pred))

def auc(y, y_pred, S):
    return roc_auc_score(y, y_pred)

def sklearn_score(estimator, X_test, y_test, func, greater_is_better, is_acc_score, is_binary):
    X = X_test.copy()
    S = X["S"]
    del X["S"]

    if is_binary:
        y_pred = estimator.predict(X)
    if not is_binary:
        y_pred = estimator.predict_proba(X)

    y_pred = check_y_array(y_pred)

    if is_acc_score and greater_is_better:
        return func(y_test, y_pred)
    if is_acc_score and not greater_is_better:
        return 1 - func(y_test, y_pred)
    if not is_acc_score and greater_is_better:
        return func(y_test, y_pred, S)
    if not is_acc_score and not greater_is_better:
        return 1 - func(y_test, y_pred, S)

# Define shortcut for all fair_metrics
fair_metrics_dict = {
    "rND": rND,
    #"GPA": group_pairwise_accuracy,
    "discrimination": discrimination,
    "auc_discrimination": auc_discrimination,
    "statistical_parity_difference": statistical_parity_diff,
    "delta": delta
}

# Define shortcut for all acc_metrics
acc_metrics_dict = {
    "ACC": accuracy_score,
    "AUC": roc_auc_score
}

# Define sklearn metric
sklearn_dict = {
    "rND": partial(sklearn_score, func=rND, greater_is_better=False, is_acc_score=False, is_binary=False),
    "discrimination": partial(sklearn_score, func=discrimination, greater_is_better=False, is_acc_score=False, is_binary=True),
    "auc_discrimination": partial(sklearn_score, func=auc_discrimination, greater_is_better=False, is_acc_score=False, is_binary=False),
    "statistical_parity_difference": partial(sklearn_score, func=statistical_parity_diff, greater_is_better=False, is_acc_score=False, is_binary=True),
    "delta": partial(sklearn_score, func=delta, greater_is_better=True, is_acc_score=False, is_binary=False),
    "acc": partial(sklearn_score, func=accuracy_score, greater_is_better=True, is_acc_score=True, is_binary=True),
    "auc": partial(sklearn_score, func=roc_auc_score, greater_is_better=True, is_acc_score=True, is_binary=False),
}
