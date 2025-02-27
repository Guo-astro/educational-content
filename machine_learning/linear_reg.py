import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.animation as animation
from sklearn.datasets import load_diabetes

# -------------------------------
# Define a refined color palette
# -------------------------------
scatter_color = 'navy'       # For data points
plane_color = 'skyblue'      # For the regression plane
surface_color = 'lightgreen' # For ±3σ surfaces
arc_color = 'darkorange'     # For Gaussian arcs (3D) and overlays
mean_line_color = 'firebrick'# For the mean line in the 2D slice

# --------------------------------
# 1) Load the real Diabetes dataset
# --------------------------------
data = load_diabetes()
X_full = data.data[:5]
t_full = data.target[:5]
feature_names = data.feature_names

# We select two features:
#   BMI (feature index 2)
#   S5  (feature index 8)
x_idx = 2  # BMI
y_idx = 8  # S5
x_data = X_full[:, x_idx]
y_data = X_full[:, y_idx]

# We'll rename them for clarity in the plot:
x_label = "BMI (feature idx=2)"
y_label = "S5 (feature idx=8)"
t_label = "Disease Progression"

# -------------------------------------------------
# 2) Fit a Linear Regression via MLE (Gaussian = OLS)
# -------------------------------------------------
N = len(t_full)
X_design = np.column_stack((np.ones(N), x_data, y_data))
w = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ t_full)
w0, w1, w2 = w

# Estimate sigma from residuals (assumed homoscedastic)
residuals = t_full - (w0 + w1 * x_data + w2 * y_data)
sigma = np.sqrt(np.sum(residuals ** 2) / N)

# ---------------------------------------
# 3) Prepare data for 3D plane & surfaces
# ---------------------------------------
x_range = np.linspace(x_data.min(), x_data.max(), 30)
y_range = np.linspace(y_data.min(), y_data.max(), 30)
X_plane, Y_plane = np.meshgrid(x_range, y_range)
# Regression plane: t = w0 + w1*x + w2*y
T_plane = w0 + w1 * X_plane + w2 * Y_plane
# ±3σ surfaces (for visualization)
T_plane_plus = T_plane + 3 * sigma
T_plane_minus = T_plane - 3 * sigma

# -----------------------------------------
# 4) Create a figure with three subplots (3 columns)
# -----------------------------------------
fig = plt.figure(figsize=(20, 6))
fig.suptitle("Linear Regression via MLE (Gaussian = OLS)\non Diabetes Data (subset)", fontsize=14)

# Left subplot: 3D scatter with regression plane & Gaussian arcs
ax3d = fig.add_subplot(1, 3, 1, projection='3d')
# Middle subplot: 2D cross-section (slice at median S5)
ax2d = fig.add_subplot(1, 3, 2)
# Right subplot: 3D projection of Gaussian uncertainty onto regression plane
ax3d_proj = fig.add_subplot(1, 3, 3, projection='3d')

# -------------------------------------------------
# 5) Left Subplot: 3D Scatter, Plane, ±3σ surfaces & Wireframe Gaussian arcs
# -------------------------------------------------
ax3d.set_title("3D Scatter + MLE/OLS Plane (±3σ)")
ax3d.scatter(x_data, y_data, t_full, c=scatter_color, marker='o', alpha=0.8, label="Data")

ax3d.plot_surface(
    X_plane, Y_plane, T_plane,
    alpha=0.5, color=plane_color, edgecolor='none'
)
ax3d.plot_surface(
    X_plane, Y_plane, T_plane_plus,
    alpha=0.1, color=surface_color, edgecolor='none'
)
ax3d.plot_surface(
    X_plane, Y_plane, T_plane_minus,
    alpha=0.1, color=surface_color, edgecolor='none'
)

# Draw Gaussian arcs (wireframes) for each datapoint:
phi_vals = np.linspace(0, 2 * np.pi, 8)
scale = 0.1  # Adjust this scale as needed

def gaussian_pdf(t, mean, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((t - mean) / sigma) ** 2)

# Loop over each datapoint instead of a grid
for xi, yi in zip(x_data, y_data):
    mean_i = w0 + w1 * xi + w2 * yi
    t_range = np.linspace(mean_i - 3 * sigma, mean_i + 3 * sigma, 50)
    for phi in phi_vals:
        pdf_vals = gaussian_pdf(t_range, mean_i, sigma)
        r = scale * pdf_vals
        X_line = xi + r * np.cos(phi)
        Y_line = yi + r * np.sin(phi)
        Z_line = t_range
        # Optionally compare with the regression plane for a split style:
        plane_line = w0 + w1 * X_line + w2 * Y_line
        below_mask = (Z_line < plane_line)
        ax3d.plot(X_line[below_mask], Y_line[below_mask], Z_line[below_mask],
                  color=arc_color, linestyle='--', lw=1)
        ax3d.plot(X_line[~below_mask], Y_line[~below_mask], Z_line[~below_mask],
                  color=arc_color, lw=1)

ax3d.set_xlabel(x_label)
ax3d.set_ylabel(y_label)
ax3d.set_zlabel(t_label)
ax3d.legend()

# Add MLE (OLS) formula annotation in 3D plot
mle_formula = (
    r"$\text{MLE = OLS:}\quad"
    r"\hat{w} = (X^T X)^{-1} X^T t,\quad"
    r"\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (t_i - \hat{t}_i)^2$"
)
ax3d.text2D(0.03, 0.97, mle_formula, transform=ax3d.transAxes,
            fontsize=9, verticalalignment='top', color='black',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))

# --------------------------------------
# 6) Middle Subplot: 2D Cross-Section (slice at median S5)
# --------------------------------------
ax2d.set_title("2D Cross-Section at Median S5\n(Mean Line ±3σ)")
y_median = np.median(y_data)
x_line = np.linspace(x_data.min(), x_data.max(), 200)
t_line = w0 + w1 * x_line + w2 * y_median
ax2d.plot(x_line, t_line, '-', color=mean_line_color,
          label=f"Mean line (S5={y_median:.2f})")

t_line_plus = t_line + 3 * sigma
t_line_minus = t_line - 3 * sigma
ax2d.plot(x_line, t_line_plus, 'k--', label="+3σ")
ax2d.plot(x_line, t_line_minus, 'k--', label="−3σ")

threshold = 0.03
mask_slice = (np.abs(y_data - y_median) < threshold)
ax2d.scatter(x_data[mask_slice], t_full[mask_slice],
             color='k', marker='o', alpha=0.7, label='Data near slice')

# Overlay a small Gaussian shape for a few x positions
t_dense = np.linspace(t_line_minus.min(), t_line_plus.max(), 200)
for x_i in np.linspace(x_data.min(), x_data.max(), 6):
    mean_i = w0 + w1 * x_i + w2 * y_median
    pdf_vals = gaussian_pdf(t_dense, mean_i, sigma)
    scale_2d = 0.03
    x_curve = x_i + scale_2d * pdf_vals
    ax2d.plot(x_curve, t_dense, '-', color=arc_color, alpha=0.8)

ax2d.set_xlabel(x_label)
ax2d.set_ylabel(t_label)
ax2d.legend(loc='best')

# ---------------------------------------------
# 7) Right Subplot: 3D Projection of Gaussian Uncertainty onto Regression Plane
# ---------------------------------------------
ax3d_proj.set_title("3D Projection of Gaussian Uncertainty\nonto Regression Plane (y = median S5)")
# Plot the regression plane (using the same grid as before)
ax3d_proj.plot_surface(
    X_plane, Y_plane, T_plane,
    alpha=0.3, color=plane_color, edgecolor='none'
)

# Also plot the mean regression line for the slice y = median
ax3d_proj.plot(x_line, np.full_like(x_line, y_median), t_line,
               '-', color=mean_line_color, lw=2, label=f"Mean line (S5={y_median:.2f})")

# Define horizontal uncertainty for x (as a fraction of the x range)
horizontal_sigma = 0.02 * (x_data.max() - x_data.min())

# Compute tangent directions in the regression plane:
# u1: along the horizontal (x) direction on the regression plane given by [1, 0, w1]
u1 = np.array([1, 0, w1])
u1 = u1 / np.linalg.norm(u1)

# u2: the projection of the vertical vector [0, 0, 1] onto the regression plane.
# The regression plane is t = w0 + w1*x + w2*y so its implicit form has normal n = [-w1, -w2, 1]
n = np.array([-w1, -w2, 1])
v = np.array([0, 0, 1])
v_proj = v - (np.dot(v, n) / np.dot(n, n)) * n
u2 = v_proj / np.linalg.norm(v_proj)

# Define ellipse half-widths:
a = 3 * horizontal_sigma  # horizontal uncertainty
b = 3 * sigma * np.linalg.norm(v_proj)  # vertical uncertainty projected onto the plane

theta_vals = np.linspace(0, 2 * np.pi, 100)
for x_i in np.linspace(x_data.min(), x_data.max(), 6):
    # Center on the regression plane for y = median:
    center = np.array([x_i, y_median, w0 + w1 * x_i + w2 * y_median])
    # Compute ellipse points in the plane using the two tangent directions:
    ellipse_points = np.array([
        center + a * np.cos(theta) * u1 + b * np.sin(theta) * u2
        for theta in theta_vals
    ])
    # Plot the ellipse as a closed 3D line
    ax3d_proj.plot(ellipse_points[:, 0], ellipse_points[:, 1], ellipse_points[:, 2],
                   color=arc_color, lw=2, alpha=0.8)

# Optionally, scatter the data points (those near y = median) for reference:
ax3d_proj.scatter(x_data[mask_slice],
                  np.full_like(x_data[mask_slice], y_median),
                  w0 + w1 * x_data[mask_slice] + w2 * y_median,
                  color='k', marker='o', alpha=0.7, label='Data (projected)')

ax3d_proj.set_xlabel(x_label)
ax3d_proj.set_ylabel(y_label)
ax3d_proj.set_zlabel(t_label)
ax3d_proj.legend(loc='best')

# -----------------------------------
# 8) Automatic rotation of the 3D plot (for the left subplot)
# -----------------------------------
def rotate(angle):
    ax3d.view_init(elev=20, azim=angle)

rot_animation = animation.FuncAnimation(
    fig, rotate, frames=np.arange(0, 360, 2), interval=100
)

plt.tight_layout()
plt.show()
