import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Only needed for older Matplotlib versions.
from matplotlib import animation
from scipy.stats import chi2
from sklearn.datasets import load_iris

# Enable full LaTeX rendering (requires a working LaTeX installation)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

# =============================================================================
# 1. Load Real Data: Use the Iris dataset (first three features)
# =============================================================================
iris = load_iris()
# Use only the first three features (e.g., sepal length, sepal width, petal length)
data = iris.data[:, :3]

# Compute the sample mean and covariance matrix
mean = np.mean(data, axis=0)
cov = np.cov(data, rowvar=False)

# =============================================================================
# 2. Compute the Confidence Ellipsoid Parameters
# =============================================================================
# Eigen decomposition of the covariance matrix
eigvals, eigvecs = np.linalg.eigh(cov)

# Confidence level: 95%
conf = 0.95
# For a 3D normal, the quadratic form follows a chi-square distribution with 3 degrees of freedom.
scale = np.sqrt(chi2.ppf(conf, df=3))

# =============================================================================
# 3. Create the Ellipsoid Surface (Based on a Unit Sphere)
# =============================================================================
# Create a grid over the unit sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

# Scale the unit sphere by the square roots of the eigenvalues and the chi-square factor.
# The semi-axes of the ellipsoid will be: scale * sqrt(eigenvalue)
ellipsoid = np.stack((x, y, z), axis=-1) * (scale * np.sqrt(eigvals))
# Rotate the ellipsoid to align with the covariance matrix
ellipsoid_rotated = ellipsoid @ eigvecs.T
# Translate the ellipsoid so its center is at the sample mean
X = ellipsoid_rotated[..., 0] + mean[0]
Y = ellipsoid_rotated[..., 1] + mean[1]
Z = ellipsoid_rotated[..., 2] + mean[2]

# =============================================================================
# 4. Create the 3D Plot with Auto-Rotation Animation
# =============================================================================
fig = plt.figure(figsize=(8, 6), dpi=80)
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Confidence Ellipsoid for Iris Data (First 3 Features)")

# Plot the real data points
ax.scatter(data[:, 0], data[:, 1], data[:, 2],
           color='blue', alpha=0.5, label="Iris Data")

# Plot the 95% confidence ellipsoid surface
ax.plot_surface(X, Y, Z, color='red', alpha=0.3, edgecolor='none')

# =============================================================================
# 5. Add Detailed Derivation as a Block of Text
# =============================================================================
# We build a multiline string with explicit newline characters.
derivation_text = (
    r"\textbf{Explain:}" "\n\n"
    r"For a 3-dimensional random vector" "\n"
    r"\["
    r"\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma}),"
    r"\]"
    "\n\n"
    r"its probability density function is given by" "\n"
    r"\["
    r"f(\mathbf{x}) = \frac{1}{(2\pi)^{3/2} |\mathbf{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \right),"
    r"\]"
    "\n\n"
    r"and the 95\% confidence ellipsoid is defined as the set" "\n"
    r"\["
    r"\left\{ \mathbf{x} : (\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \le \chi^2_{0.95,3} \right\}."
    r"\]"
    "\n\n"
    r"Here, the mean vector and covariance matrix are:"
    "\n"
    r"\["
    r"\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \\ \mu_3 \end{bmatrix}, \quad"
    r"\mathbf{\Sigma} = \begin{bmatrix} "
    r"\sigma_{11} & \sigma_{12} & \sigma_{13} \\[6pt]"
    r"\sigma_{12} & \sigma_{22} & \sigma_{23} \\[6pt]"
    r"\sigma_{13} & \sigma_{23} & \sigma_{33}"
    r"\end{bmatrix}."
    r"\]"
)

# Place the derivation text in a text box at the bottom of the figure.
fig.text(0.05, 0.01, derivation_text, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')


# =============================================================================
# 6. Auto-Rotation Animation Function
# =============================================================================
def update_view(angle):
    ax.view_init(elev=30, azim=angle)
    return ax,


ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_zlabel("Petal Length (cm)")
ani = animation.FuncAnimation(fig, update_view, frames=np.arange(0, 360, 2), interval=100)
ani.save('ellipsoid_animation.gif', writer='pillow', fps=30)

plt.show()
