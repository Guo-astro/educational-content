import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.stats as st

# ------------- Ratio Estimator and Delta-Method Variance -------------
def ratio_estimator(x, y):
    """Compute the ratio r = sum(y)/sum(x)."""
    return np.sum(y) / np.sum(x)

def ratio_variance_approx(x, y):
    """
    Returns the approximate variance of the ratio estimator via the Delta Method:
    Var(r_hat) ~ (1/n) * [ (s_y^2 - 2*r_hat*s_xy + r_hat^2 * s_x^2 ) / (xbar^2 ) ].
    """
    n = len(x)
    xbar = np.mean(x)
    ybar = np.mean(y)
    r_hat = ybar / xbar

    s_x2 = np.var(x, ddof=1)
    s_y2 = np.var(y, ddof=1)
    s_xy = np.cov(x, y, ddof=1)[0, 1]

    var_r = (1.0 / n) * ((s_y2 - 2*r_hat*s_xy + (r_hat**2)*s_x2) / (xbar**2))
    return var_r

def bootstrap_ratio(x, y, B=1000):
    """
    Bootstrap resampling to get distribution of the ratio estimator.
    """
    n = len(x)
    estimates = []
    for _ in range(B):
        idx = np.random.randint(0, n, size=n)
        x_b = x[idx]
        y_b = y[idx]
        estimates.append(ratio_estimator(x_b, y_b))
    return np.array(estimates)

# ------------- Generate Synthetic Data (Given n) -------------
def generate_data_and_estimates(n, B=1000):
    """
    1) Generate synthetic (x, y) data of size n
    2) Compute ratio estimator and Delta-Method variance
    3) Bootstrap the ratio
    """
    np.random.seed(42)  # fixed seed for reproducibility

    # X ~ lognormal, Y ~ theta*X + noise
    theta_true = 2.0
    x = np.random.lognormal(mean=0.5, sigma=0.6, size=n)
    noise = np.random.normal(loc=0, scale=0.3, size=n)
    y = theta_true * x + noise

    # Ratio + approximate variance
    r_hat = ratio_estimator(x, y)
    var_r_approx = ratio_variance_approx(x, y)

    # Bootstrap
    r_boot = bootstrap_ratio(x, y, B=B)

    return r_hat, var_r_approx, r_boot

# ------------- Set up the Animation -------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.3)

# We'll vary n from small to large so we can see how the difference evolves
n_values = [20, 50, 100, 200, 500, 1000]

def animate(frame):
    n = n_values[frame]
    # Generate data & estimates
    r_hat, var_r_approx, r_boot = generate_data_and_estimates(n, B=2000)

    # Clear subplots
    for ax in axes:
        ax.clear()

    # -- Left subplot: Bootstrap distribution --
    axes[0].hist(r_boot, bins=30, density=True, alpha=0.6, color='C0', edgecolor='k')
    axes[0].axvline(np.mean(r_boot), color='r', linestyle='--', label='Bootstrap Mean')
    axes[0].set_title(f'Bootstrap Dist.\nn={n}')
    axes[0].set_xlabel('Ratio')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Annotate ratio formula in the top-left corner of the left plot
    # (LaTeX in raw strings with single '$' for math mode)
    axes[0].text(
        0.05, 0.90,
        r"$\hat{r} \;=\; \frac{\sum_{i=1}^n Y_i}{\sum_{i=1}^n X_i}$",
        transform=axes[0].transAxes,
        fontsize=11,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # -- Right subplot: Compare normal approximations --
    x_vals = np.linspace(min(r_boot), max(r_boot), 200)
    mean_boot = np.mean(r_boot)
    std_boot = np.std(r_boot, ddof=1)
    pdf_boot = st.norm.pdf(x_vals, loc=mean_boot, scale=std_boot)

    std_delta = np.sqrt(var_r_approx)
    pdf_delta = st.norm.pdf(x_vals, loc=r_hat, scale=std_delta)

    axes[1].hist(r_boot, bins=30, density=True, alpha=0.6, color='C0', edgecolor='k', label='Bootstrap Hist')
    axes[1].plot(x_vals, pdf_boot, 'r--', lw=2, label='Normal from Bootstrap')
    axes[1].plot(x_vals, pdf_delta, 'g-',  lw=2, label='Delta-Method Normal')

    # Add vertical lines for the two means
    axes[1].axvline(mean_boot, color='r', linestyle='--')
    axes[1].axvline(r_hat, color='g', linestyle='-.')

    axes[1].set_title(f'Compare Normal Approximations\nn={n}')
    axes[1].set_xlabel('Ratio')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    # Annotate Delta-Method variance formula in the top-left corner of the right plot
    axes[1].text(
        0.05, 0.90,
        (
            r"$\mathrm{Var}(\hat{r}) \;\approx\;"
            r"\frac{1}{n}\,\frac{s_{y}^{2} \;-\; 2\,\hat{r}\,s_{xy} \;+\; \hat{r}^{2}\,s_{x}^{2}}"
            r"{\bar{X}^{2}}$"
        ),
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

def init():
    # Initial draw (blank)
    return []

ani = FuncAnimation(fig, animate, frames=len(n_values), interval=2000, init_func=init, repeat=False)
plt.show()
