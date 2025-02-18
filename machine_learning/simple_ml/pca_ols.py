import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation

def generate_data(n=30, correlation='low'):
    """
    Generate synthetic data (x1, x2, y) for two scenarios:
    - correlation='low': x1, x2 ~ independent
    - correlation='high': x2 ~ x1 + small noise
    True model: y = 3 + 2*x1 - 1*x2 + noise
    """
    np.random.seed(42)
    if correlation == 'low':
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)  # independent of x1
    else:  # 'high'
        x1 = np.random.randn(n)
        x2 = x1 + 0.01 * np.random.randn(n)  # nearly collinear with x1

    true_intercept = 3.0
    true_coefs = np.array([2.0, -1.0])
    noise = 0.3 * np.random.randn(n)

    y = true_intercept + x1*true_coefs[0] + x2*true_coefs[1] + noise
    X = np.column_stack((x1, x2))
    return X, y

def fit_and_plot(ax, X, y, alpha_ridge=5.0):
    """
    Fit OLS, Ridge, and PCR(1PC) to (X,y), then plot:
    - 3D scatter of (x1, x2, y)
    - 3 planes (OLS, Ridge, PCR)
    - Text annotation of each plane's formula
    """
    # Fit OLS
    ols = LinearRegression()
    ols.fit(X, y)
    b0_ols = ols.intercept_
    b1_ols, b2_ols = ols.coef_

    # Fit Ridge
    ridge = Ridge(alpha=alpha_ridge)
    ridge.fit(X, y)
    b0_ridge = ridge.intercept_
    b1_ridge, b2_ridge = ridge.coef_

    # Fit PCR (1 PC)
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)
    ols_on_pca = LinearRegression().fit(X_pca, y)
    theta_1_pca = ols_on_pca.coef_[0]
    intercept_pca_model = ols_on_pca.intercept_
    beta_pcr = theta_1_pca * pca.components_[0]  # shape: (2,)
    intercept_pcr = intercept_pca_model - beta_pcr @ pca.mean_
    b0_pcr = intercept_pcr
    b1_pcr, b2_pcr = beta_pcr

    # 3D scatter
    ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')

    # Create a meshgrid for plotting surfaces
    x_surf = np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 20)
    y_surf = np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 20)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)

    # Evaluate each plane
    z_ols = b0_ols + b1_ols*x_surf + b2_ols*y_surf
    z_ridge = b0_ridge + b1_ridge*x_surf + b2_ridge*y_surf
    z_pcr = b0_pcr + b1_pcr*x_surf + b2_pcr*y_surf

    # Plot surfaces
    ax.plot_surface(x_surf, y_surf, z_ols, alpha=0.3, color='blue')
    ax.plot_surface(x_surf, y_surf, z_ridge, alpha=0.3, color='green')
    ax.plot_surface(x_surf, y_surf, z_pcr, alpha=0.3, color='orange')

    # Annotate plane equations
    # We'll place text in 2D coords in the top-left corner of each subplot
    ax.text2D(0.05, 0.95,
              (f"OLS:\n"
               fr"$\hat{{y}} = {b0_ols:.2f} + {b1_ols:.2f}\,x_1 + {b2_ols:.2f}\,x_2$"),
              transform=ax.transAxes, color='blue', fontsize=9)

    ax.text2D(0.05, 0.80,
              (f"Ridge (α={alpha_ridge}):\n"
               fr"$\hat{{y}} = {b0_ridge:.2f} + {b1_ridge:.2f}\,x_1 + {b2_ridge:.2f}\,x_2$"),
              transform=ax.transAxes, color='green', fontsize=9)

    ax.text2D(0.05, 0.65,
              (f"PCR (1 PC):\n"
               fr"$\hat{{y}} = {b0_pcr:.2f} + {b1_pcr:.2f}\,x_1 + {b2_pcr:.2f}\,x_2$"),
              transform=ax.transAxes, color='orange', fontsize=9)

    # Labels
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")

# -------------------------------
# Create figure with 2 subplots
# -------------------------------
fig = plt.figure(figsize=(14, 6))

# Left panel: low correlation
ax1 = fig.add_subplot(121, projection='3d')
X_low, y_low = generate_data(n=30, correlation='low')
fit_and_plot(ax1, X_low, y_low, alpha_ridge=5.0)
ax1.set_title("Low Correlation")

# Right panel: high correlation
ax2 = fig.add_subplot(122, projection='3d')
X_high, y_high = generate_data(n=30, correlation='high')
fit_and_plot(ax2, X_high, y_high, alpha_ridge=5.0)
ax2.set_title("High Correlation")

plt.tight_layout()

# -------------------------------
#  Animation: rotate both subplots
# -------------------------------
def update(frame):
    # frame goes from 0..(num_frames-1)
    angle = frame
    ax1.view_init(elev=30, azim=angle)
    ax2.view_init(elev=30, azim=angle)
    return []

num_frames = 90  # rotate from 0 to 90 degrees for demonstration
ani = FuncAnimation(fig, update, frames=np.arange(0, num_frames, 2), interval=100)

plt.show()
