import numpy as np
import pandas as pd

def entropy_scores(scores,rng):

    M = 10

    n_bins_list = np.unique(np.round(np.linspace(5, 150, 40)).astype(int))
    cv_results = cv_loglik_equalwidth(scores, n_bins_list, k_folds=5, alpha=1.0, rng=rng)
    best = max(cv_results, key=lambda d: d['mean_loglik'])

    bins = best['n_bins']
    counts, edges = np.histogram(scores, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    entropies = np.zeros(M)

    for m in range(M):
        samples_unc, p_draw = sample_from_hist_with_uncertainty(counts, edges, size=5000, alpha=1.0, rng=rng)
        counts_unc, edges_unc = np.histogram(samples_unc, bins=edges, density=False)
        prob_unc = counts_unc/5000

        for i in range(bins):
            summand = prob_unc[i] * np.log2(prob_unc[i])
            summand = np.nan_to_num(summand,copy=False, nan=0.0)
            entropies[m] -= summand

    return entropies, entropies/np.log2(bins), bins


def cv_loglik_equalwidth(x, n_bins_list, k_folds=5, alpha=1.0, rng=None):

    # variables
    N = len(x)
    x_min = np.min(x)
    x_max = np.max(x)
    # Slightly pad the range to avoid edge floating errors
    padding = 1e-9 * max(1.0, (x_max - x_min))
    x_min -= padding
    x_max += padding

    # Shuffle indices and split into folds
    idx = rng.permutation(N)
    folds = np.array_split(idx, k_folds)

    results = []
    for B in n_bins_list:
        fold_scores = []
        width = (x_max - x_min) / B
        edges = x_min + np.arange(B+1) * width

        for i in range(k_folds):
            val_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k_folds) if j != i])

            x_train = x[train_idx]
            x_val = x[val_idx]

            # Train: counts per bin (equal-width, fixed edges)
            counts, _ = np.histogram(x_train, bins=edges)
            N_train = counts.sum()

            # Smoothed bin probabilities
            p_bins = (counts + alpha) / (N_train + alpha * B)

            # Validation log-likelihood
            # For each val sample, find its bin index
            # np.searchsorted gives the index of the right edge; subtract 1 to get bin
            bin_indices = np.searchsorted(edges, x_val, side='right') - 1
            # Clip to [0, B-1] just in case of numerical extremes
            bin_indices = np.clip(bin_indices, 0, B-1)

            loglik = np.log(p_bins[bin_indices]) - np.log(width)
            fold_scores.append(np.mean(loglik))

        results.append({
            'n_bins': B,
            'mean_loglik': float(np.mean(fold_scores)),
            'std_loglik': float(np.std(fold_scores, ddof=1)),
            'per_fold': fold_scores,
        })
    return results


def sample_from_hist_with_uncertainty(counts, edges, size, alpha=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        raise ValueError('All histogram counts are zero.')
    # Dirichlet draw for probabilities
    p = rng.dirichlet(counts + alpha)
    # Sample bin indices and then sample uniformly within chosen bins
    bin_idx = rng.choice(len(p), size=size, p=p)
    left = edges[bin_idx]
    right = edges[bin_idx + 1]
    return rng.uniform(left, right), p

if __name__=="__main__":
    # Synthetic data with skew + outliers to highlight differences
    rng = np.random.default_rng(1234)

    # Build a toy dataset with multiple scales & skew
    n = 6000
    x = np.concatenate([
        rng.normal(-1.2, 0.5, int(0.45*n)),
        rng.normal( 1.8, 0.6, int(0.35*n)),
        rng.lognormal(0.0, 0.5, int(0.18*n)) + 1.0,
        rng.normal(-4.5, 0.25, int(0.02*n)),
    ])
    entr, entr_normalized = entropy_scores(x,rng)
    print(entr)
    print(entr_normalized)
    print(np.mean(entr_normalized))
    print(np.std(entr_normalized))