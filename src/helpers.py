import matplotlib.pyplot as plt
import numpy as np

def vfae_check(X, model, path, gamma, n_passes=100):
    """
    Repeatedly query:
      1) model.get_representations(X)
         - focusing on the FIRST sample's representation across multiple passes.
      2) model.predict_proba(X)
         - for 20 different samples, each across multiple passes, then plot a grid
           of histograms for the (class=1) probability.

    Assumptions:
    1) X has at least 20 data samples; we only use X[0] for representation checks,
       and X[0..19] for probability histograms.
    2) model.get_representations(X) -> array of shape (num_samples, num_features)
    3) model.predict_proba(X)       -> array of shape (num_samples,),
       each entry is p(class=1) for that sample. p(class=0) = 1 - p(class=1).

    Parameters
    ----------
    X : np.ndarray
        Input data (>= 20 samples). We'll do different things with the first sample
        vs. the first 20 samples.
    model : object
        A model with methods:
            - get_representations(X)
            - predict_proba(X)
    path : str
        Directory path where plots are saved.
    gamma : float
        A parameter (or tag) used in filenames for saving the plot.
    n_passes : int
        Number of times we repeatedly sample (default 10).
    """

    # Ensure n_passes is an integer
    n_passes = int(n_passes)

    # =========================================================================
    #  1) LATENT REPRESENTATIONS FOR THE FIRST SAMPLE
    # =========================================================================
    repr_list = []
    for _ in range(n_passes):
        # shape: (num_samples, num_features)
        repr_test = model.get_representations(X)  
        # Focus on the first sample only
        repr_list.append(repr_test[0])  # shape: (num_features,)

    # multi_repr: (n_passes, num_features)
    multi_repr = np.array(repr_list)

    # Plot a grid of histograms for up to the first 10 neurons
    num_features = multi_repr.shape[1]
    num_neurons  = min(10, num_features)

    fig_repr, axes_repr = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows x 5 columns
    axes_repr = axes_repr.flatten()

    for i in range(num_neurons):
        ax = axes_repr[i]
        data = multi_repr[:, i]  # shape: (n_passes,)
        ax.hist(data, bins=5, alpha=0.7)
        ax.set_title(f"Neuron {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    # Hide any extra subplots if model has < 10 features
    for j in range(num_neurons, 10):
        fig_repr.delaxes(axes_repr[j])

    plt.tight_layout()
    fig_repr.suptitle(
        f"Latent Representations (Sample 0) - First {num_neurons} Neurons (gamma={gamma})",
        y=1.02
    )
    fig_repr.savefig(f"{path}/repr_histograms_{round(gamma, 1)}.png")
    plt.close(fig_repr)

    # Print variance across n_passes for each neuron
    column_variances = np.var(multi_repr, axis=0)
    print(f"\n[Representation Variances for the First {num_neurons} Neurons]")
    for i in range(num_neurons):
        print(f"  Neuron {i}: var = {column_variances[i]:.6f}")

    # =========================================================================
    #  2) CLASS PROBABILITY HISTOGRAMS FOR 20 DIFFERENT SAMPLES
    # =========================================================================
    # We now look at predict_proba for 20 different samples across n_passes.
    sample_count = 20
    if X.shape[0] < sample_count:
        raise ValueError(
            f"X has only {X.shape[0]} samples, but we need at least {sample_count}."
        )

    # We'll store the predicted probability of class=1 for each of the 20 samples
    # across all n_passes.  multi_proba: (sample_count, n_passes)
    multi_proba = np.zeros((sample_count, n_passes), dtype=np.float32)

    for pass_idx in range(n_passes):
        # shape: (num_samples,) -> p(class=1) for each sample
        probas = model.predict_proba(X)
        # Collect the first 20 samples' class-1 probabilities
        for sample_idx in range(sample_count):
            multi_proba[sample_idx, pass_idx] = probas[sample_idx]

    # We'll build a grid of histograms: 4 rows x 5 columns = 20 subplots
    fig_proba, axes_proba = plt.subplots(4, 5, figsize=(20, 12))
    axes_proba = axes_proba.flatten()

    for sample_idx in range(sample_count):
        ax = axes_proba[sample_idx]
        # Probability of class=1 across n_passes
        data = multi_proba[sample_idx, :]  # shape: (n_passes,)
        ax.hist(data, bins=20, alpha=0.7)
        ax.set_title(f"Sample {sample_idx}")
        ax.set_xlabel("p(class=1)")
        ax.set_ylabel("Frequency")
        ax.set_xlim((0, 1))

    plt.tight_layout()
    fig_proba.suptitle(
        f"Predicted Probability Histograms (20 Samples) - p(class=1), gamma={gamma}",
        y=1.02
    )
    fig_proba.savefig(f"{path}/proba_20samples_{round(gamma, 1)}.png")
    plt.close(fig_proba)

    # Print variance across passes for each of the 20 samples
    prob_variances = np.var(multi_proba, axis=1)
    print(f"\n[Probability Variances for 20 Samples (class=1)]")
    for sample_idx in range(sample_count):
        print(f"  Sample {sample_idx}: var = {prob_variances[sample_idx]:.6f}")
