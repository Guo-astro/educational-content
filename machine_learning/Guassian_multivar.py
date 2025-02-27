import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from scipy.stats import multivariate_normal
import matplotlib as mpl
import japanize_matplotlib_modern

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Convert cm to inches: 1 cm = 0.393701 inches
fig_width = 19 * 0.393701  # ≈ 5.31 inches
fig_height = 15 * 0.393701  # ≈ 6.30 inches

# --- Prepare the grid and constant parameters ---
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
mean = np.array([0, 0])
num_frames = 200  # Total number of animation frames

# --- Create the figure with two subplots ---
fig = plt.figure(figsize=(fig_width, fig_height))
# Left subplot: 3D plot
ax3d = fig.add_subplot(121, projection='3d')
# Right subplot: 2D plot
ax2d = fig.add_subplot(122)

# --- Add a fixed text annotation (suptitle) with the multivariate Gaussian formula ---
equation = r'$p(x,y) = \frac{1}{2\pi\sqrt{1-\rho^2}} \exp\!\left(-\frac{1}{2}\, \begin{pmatrix} x & y \end{pmatrix} \frac{1}{1-\rho^2}\, \begin{pmatrix} 1 & -\rho \\ -\rho & 1 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}\right)$'
fig.suptitle(equation, fontsize=12)

# --- Animation update function ---
def update(frame):
    # Clear both axes for fresh drawing
    ax3d.clear()
    ax2d.clear()

    # Vary the correlation parameter over time (oscillates between -0.8 and 0.8)
    corr = 0.8 * np.sin(2 * np.pi * frame / num_frames)
    cov = np.array([[1, corr],
                    [corr, 1]])

    # Create a new multivariate normal with the updated covariance
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)

    # --- Update the 3D plot ---
    # Plot the 3D Gaussian surface
    ax3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    # Project contours onto the "floor" for a clear view of the elliptical level sets
    offset = np.min(Z) - 0.005
    ax3d.contour(X, Y, Z, zdir='z', offset=offset, cmap='viridis')

    ax3d.set_title(f'3D Gaussian Surface\nCorrelation ($\\rho$): {corr:.2f}', fontsize=14)
    ax3d.set_xlabel('X-axis', fontsize=14)
    ax3d.set_ylabel('Y-axis', fontsize=14)
    ax3d.set_zlabel('Density', fontsize=14)
    ax3d.set_zlim(offset, 0.5)
    ax3d.view_init(elev=30, azim=frame * 360 / num_frames)

    # --- Update the 2D contour plot ---
    cont = ax2d.contour(X, Y, Z, levels=10, cmap='viridis')
    ax2d.set_title(f'2D Gaussian Contour\nCorrelation ($\\rho$): {corr:.2f}', fontsize=14)
    ax2d.set_xlabel('X-axis', fontsize=14)
    ax2d.set_ylabel('Y-axis', fontsize=14)
    ax2d.clabel(cont, inline=True, fontsize=8)

    # --- Compute PCA (eigen decomposition of the covariance matrix) ---
    eigvals, eigvecs = np.linalg.eigh(cov)
    scale_factor = 2.0  # Adjust this factor for visualization
    for i in range(len(eigvals)):
        # Scale each eigenvector by the square root of its eigenvalue
        vector = eigvecs[:, i] * np.sqrt(eigvals[i]) * scale_factor
        ax2d.arrow(mean[0], mean[1], vector[0], vector[1],
                   head_width=0.1, head_length=0.1, fc='r', ec='r', linewidth=2)
        # Add text labels for eigenvectors
        ax2d.text(mean[0] + vector[0], mean[1] + vector[1]+0.5, f'$v_{{{i+1}}}$',
                  color='r', fontsize=12)

# --- Create the animation ---
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, repeat=True)

# Adjust layout so that the suptitle does not overlap the subplots
plt.tight_layout(rect=[0, 0, 1, 0.82])
plt.show()
