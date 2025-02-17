#!/usr/bin/env python3
"""
Demonstrate 12 ML models using the Wine dataset from scikit-learn,
including a deeper MLP and an ensemble (VotingClassifier).

Data usage:
-----------
- Linear Regression: 'alcohol' (feature 0) as target, other 12 features as predictors.
- Classification (Logistic Regression, Naive Bayes, Decision Tree,
  Gradient Boosting, Random Forest, KNN, MLP, Deep MLP, Ensemble):
  uses the 3-class Wine labels, but reduces data to 2D via PCA for plotting
  decision boundaries. (In real practice, you'd use the full 13D data.)

- PCA demonstration: 13D -> 2D scatter with true class labels.
- K-Means: clustering on the same 2D data.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier


def plot_decision_boundary(model, X_2d, y, ax, title, hyperparams, formula=""):
    """
    Train on X_2d, y (both 2D) and plot the decision boundary plus scatter.
    Also display a text box showing only the formula.
    """
    # Train the model
    model.fit(X_2d, y)

    # If the model is Logistic Regression, build a formula string with learned parameters.
    learned_formula = formula
    if isinstance(model, LogisticRegression):
        intercept = model.intercept_[0]
        coefs = model.coef_[0]

        # Build the logistic function formula with actual learned intercept/coefs.
        learned_formula = r"$P(y=1|x) = \frac{1}{1 + e^{-("
        learned_formula += f"{intercept:.2f}"
        for i, c in enumerate(coefs):
            learned_formula += f" + {c:.2f}x_{{{i + 1}}}"
        learned_formula += ")}}$"

    # Determine grid boundaries.
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    # Create a mesh grid.
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict for each point in the grid and reshape.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot the decision boundary.
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(title, fontsize=9)

    # Display only the formula.
    textstr = f"{learned_formula}"
    ax.text(0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.spines[['top', 'right']].set_visible(False)


def plot_linear_regression(ax, X_reg, y_reg, formula=""):
    """
    Linear Regression: Predict 'alcohol' from the first column of X_reg for 1D demonstration.
    The displayed formula will show the learned intercept and coefficient.
    """
    X_1d = X_reg[:, [0]]
    model = LinearRegression(fit_intercept=True)
    model.fit(X_1d, y_reg)
    intercept = model.intercept_
    coef = model.coef_[0]

    # Build the displayed formula.
    learned_formula = f"$y = {intercept:.2f} + {coef:.2f}x$"

    ax.scatter(X_1d, y_reg, c='steelblue', edgecolors='k', alpha=0.9)
    x_vals = np.linspace(X_1d.min(), X_1d.max(), 100).reshape(-1, 1)
    ax.plot(x_vals, model.predict(x_vals), color='darkred', linewidth=2)

    ax.set_title("Linear Regression\n(Predict 'alcohol')", fontsize=9)
    ax.set_xlabel("Feature 1", fontsize=8)
    ax.set_ylabel("Alcohol", fontsize=8)

    # Display only the formula.
    textstr = f"{learned_formula}"
    ax.text(0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.spines[['top', 'right']].set_visible(False)


def plot_pca_scatter(ax, X_full, y, formula=""):
    """
    PCA scatter: Reduce 13D -> 2D and plot with true class labels.
    """
    pca_full = PCA(n_components=2)
    X_2d_full = pca_full.fit_transform(X_full)
    ax.scatter(X_2d_full[:, 0], X_2d_full[:, 1], c=y, cmap='viridis', edgecolors='k')
    ax.set_title("PCA \n(13D -> 2D)", fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)

    # Display only the formula.
    textstr = f"{formula}"
    ax.text(0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.spines[['top', 'right']].set_visible(False)


def plot_kmeans(ax, X_2d, n_clusters=3, formula=""):
    """
    K-Means clustering on the 2D Wine data.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10,
                    max_iter=300, random_state=42)
    kmeans.fit(X_2d)
    y_km = kmeans.predict(X_2d)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_km, cmap='viridis', edgecolors='k')

    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], s=150, marker='X', c='red', edgecolors='k')
    ax.set_title("K-Means (Wine, 2D)", fontsize=9)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)

    # Display only the formula.
    textstr = f"{formula}"
    ax.text(0.05, 0.95, textstr,
            transform=ax.transAxes, fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.spines[['top', 'right']].set_visible(False)


def main():
    # Load Wine dataset.
    wine = load_wine()
    X_full = wine.data
    y_class = wine.target

    # For regression: predict 'alcohol' (column 0) from columns 1..12.
    y_reg = X_full[:, 0]
    X_reg = X_full[:, 1:]  # shape (n_samples, 12)

    # For classification boundaries, reduce to 2D with PCA.
    pca_2d = PCA(n_components=2, random_state=42)
    X_class_2d = pca_2d.fit_transform(X_full)

    # Create a 3x4 grid for 12 subplots (3 rows, 4 columns).
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("12 ML Models on on Wine dataset from scikit-learn", fontsize=12, y=0.98)

    # Assign models in order (row-major):
    # Row 0
    plot_linear_regression(axes[0, 0], X_reg, y_reg, formula=r"$y = \beta_0 + \beta_1 x$")

    log_default_formula = r"$P(y=1|x)=\frac{1}{1+e^{-(\beta_0+\beta_1 x_1+\beta_2 x_2)}}$"
    log_reg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=200)
    log_params = {"pen": "l2", "C": 1.0, "solver": "lbfgs"}
    plot_decision_boundary(log_reg, X_class_2d, y_class, axes[0, 1],
                           "Logistic Regression", log_params, formula=log_default_formula)

    nb_formula = r"$P(C|x) \propto P(C)\prod_i P(x_i|C)$"
    nb = GaussianNB(var_smoothing=1e-9)
    nb_params = {"var_smooth": 1e-9}
    plot_decision_boundary(nb, X_class_2d, y_class, axes[0, 2],
                           "Naive Bayes", nb_params, formula=nb_formula)

    dt_formula = r"$\text{Decision Tree: recursive splits}$"
    dt = DecisionTreeClassifier(criterion='gini', max_depth=5)
    dt_params = {"criterion": "gini", "max_depth": 5}
    plot_decision_boundary(dt, X_class_2d, y_class, axes[0, 3],
                           "Decision Tree", dt_params, formula=dt_formula)

    # Row 1
    gb_formula = r"$F(x)=\sum_{m=1}^{M} \gamma_m h_m(x)$"
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0)
    gb_params = {"n_est": 100, "lr": 0.1, "max_depth": 3}
    plot_decision_boundary(gb, X_class_2d, y_class, axes[1, 0],
                           "Gradient Boosting", gb_params, formula=gb_formula)

    rf_formula = r"$\mathrm{RF}(x)=\mathrm{majority\ vote}(T_1(x),\ldots,T_N(x))$"
    rf = RandomForestClassifier(n_estimators=100, max_depth=5)
    rf_params = {"n_est": 100, "max_depth": 5}
    plot_decision_boundary(rf, X_class_2d, y_class, axes[1, 1],
                           "Random Forest", rf_params, formula=rf_formula)

    pca_formula = r"$Z = W^T (X-\mu)$"
    plot_pca_scatter(axes[1, 2], X_full, y_class, formula=pca_formula)

    knn_formula = r"$y = \mathrm{mode}\{y_i : x_i \in \mathcal{N}(x)\}$"
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_params = {"n_neighbors": 5}
    plot_decision_boundary(knn, X_class_2d, y_class, axes[1, 3],
                           "K-Nearest Neighbors", knn_params, formula=knn_formula)

    # Row 2
    kmeans_formula = r"$\min_{C} \sum_{k=1}^{K} \sum_{x \in C_k} \|x-\mu_k\|^2$"
    plot_kmeans(axes[2, 0], X_class_2d, n_clusters=3, formula=kmeans_formula)

    mlp_formula = r"$y=\mathrm{softmax}(W^{(2)} \sigma(W^{(1)}x+b^{(1)})+b^{(2)})$"
    mlp = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu',
                        solver='adam', learning_rate_init=0.001,
                        max_iter=300, random_state=42)
    mlp_params = {"layers": "(50,25)", "act": "relu", "solver": "adam"}
    plot_decision_boundary(mlp, X_class_2d, y_class, axes[2, 1],
                           "MLP Classifier", mlp_params, formula=mlp_formula)

    deep_mlp_formula = r"$y=\mathrm{softmax}(W^{(3)} \sigma(W^{(2)} \sigma(W^{(1)}x+b^{(1)})+b^{(2)})+b^{(3)})$"
    deep_mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 50),
                             activation='relu', solver='adam',
                             learning_rate_init=0.001, max_iter=400,
                             random_state=42)
    deep_mlp_params = {"layers": "(100,100,50)", "act": "relu", "solver": "adam"}
    plot_decision_boundary(deep_mlp, X_class_2d, y_class, axes[2, 2],
                           "Deep MLP", deep_mlp_params, formula=deep_mlp_formula)

    ens_formula = r"$y = \mathrm{mode}(f_1(x),f_2(x),f_3(x))$"
    log_clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=200, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam',
                            learning_rate_init=0.001, max_iter=300, random_state=42)
    ensemble = VotingClassifier(estimators=[('lr', log_clf), ('rf', rf_clf), ('mlp', mlp_clf)],
                                voting='hard')
    ensemble_params = {"estimators": "LR,RF,MLP", "voting": "hard"}
    plot_decision_boundary(ensemble, X_class_2d, y_class, axes[2, 3],
                           "Ensemble (Voting)", ensemble_params, formula=ens_formula)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
