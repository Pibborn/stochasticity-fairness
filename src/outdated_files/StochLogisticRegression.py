from sklearn.linear_model import LogisticRegression
import numpy as np

class StochasticLogisticRegression(LogisticRegression):
    def __init__(self,random_state=None, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return self.rng.binomial(n=1, p=prob)

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from load_data import DATALOADER
    import matplotlib.pyplot as plt
    import argparse
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--dataset", type=str, default="adult")
    args = argparser.parse_args()
    dataset = args.dataset
    SEED = args.seed
    
    def entropy(p):
        if p == 0.0 or p == 1.0:
            return 0
        return -p*np.log2(p) - (1-p)*np.log2(1-p)
    
    X_train, X_test, y_train, y_test, S_train, S_test = DATALOADER[dataset](SEED)
    model = StochasticLogisticRegression(random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    
    entropy_vect = np.vectorize(entropy)
    entr_y = entropy_vect(model.predict_proba(X_test)[:, 1])
    
    plt.hist(entr_y, bins=20,color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Entropy')
    plt.ylabel('Number')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0,1)
    plt.title(f"Entropy of predictions (Acc: {acc})")
    mean = np.mean(entr_y)
    std = np.std(entr_y)
    plt.axvline(mean, color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(mean + std, color='grey', linestyle='dashed', linewidth=1)
    plt.axvline(mean - std, color='grey', linestyle='dashed', linewidth=1)
    
    plt.savefig(f"entropy_hist_{dataset}_stochlogreg.pdf")
    