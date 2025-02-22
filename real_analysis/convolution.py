import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # for 3D projection

print(plt.style.available)

# Use a pre-defined style for a modern look
plt.style.use('seaborn-v0_8-dark-palette')

# --------------------------
# 1. Fallbacks for SciPy Integration Functions
# --------------------------
try:
    from scipy.integrate import trapezoid
except ImportError:
    # For SciPy versions < 1.7, use numpy.trapz as a fallback
    def trapezoid(y, x):
        return np.trapz(y, x)

try:
    from scipy.integrate import cumtrapz
except ImportError:
    # Define cumtrapz if not available
    def cumtrapz(y, x, initial=0):
        y = np.asarray(y)
        x = np.asarray(x)
        dx = np.diff(x)
        trapz_areas = dx * (y[:-1] + y[1:]) / 2
        cum_int = np.concatenate(([initial], np.cumsum(trapz_areas)))
        return cum_int


# --------------------------
# 2. Define PDFs
# --------------------------
def pdf_triangular(x):
    """
    Triangular PDF on [-1, 1]:
      f(x) = 1 - |x| for |x| < 1, and 0 otherwise.
    Normalized to 1.
    """
    return np.where(np.abs(x) < 1, 1 - np.abs(x), 0)


def pdf_hat(x, L=-2, a=-0.2, b=1.7, R=2):
    """
    Asymmetric, non-uniform hat (trapezoidal) PDF on [L, R].

    It is defined as:
      - Linear from L to a,
      - Constant on [a, b],
      - Linear from b to R.

    The plateau and slopes are chosen such that the total area is 1.
    Normalization:
      h = 2 / ((R - L) + (b - a))
    """
    h = 2.0 / ((R - L) + (b - a))
    result = np.zeros_like(x)

    # Left slope: linear increase from L to a
    left_mask = (x >= L) & (x < a)
    result[left_mask] = h * (x[left_mask] - L) / (a - L)

    # Plateau: constant on [a, b]
    center_mask = (x >= a) & (x <= b)
    result[center_mask] = h

    # Right slope: linear decrease from b to R
    right_mask = (x > b) & (x <= R)
    result[right_mask] = h * (R - x[right_mask]) / (R - b)

    return result


# Choose PDFs: f(x)= triangular and g(x)= hat
f = pdf_triangular
g = pdf_hat

# --------------------------
# 3. Set Up Grids for the 3D Surface and Convolution
# --------------------------
# Define grid limits and resolution
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
n_points = 200

# Create grid for the 3D surface: z = f(x) * g(y)
x_vals = np.linspace(x_min, x_max, n_points)
y_vals = np.linspace(y_min, y_max, n_points)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X) * g(Y)


# Define the convolution (f*g)(c) = ∫ f(x) g(c-x) dx
def convolution(c):
    integrand = f(x_vals) * g(c - x_vals)
    return trapezoid(integrand, x_vals)


# Convolution domain and computed values
c_min, c_max = -4.0, 4.0
c_vals = np.linspace(c_min, c_max, n_points)
conv_vals = np.array([convolution(ci) for ci in c_vals])

# --------------------------
# 4. Create 2x2 Figure Layout with Enhanced Aesthetics
# --------------------------
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.3, wspace=0.3)

# --- Ax1: 3D Surface with Animated Slice ---
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.set_title(
    'z = f(x)·g(y)\nSlice: x + y = c',
    fontsize=10, fontweight='bold'
)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_zlabel('z', fontsize=10)
surf = ax1.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, edgecolor='none')
slice_line, = ax1.plot([], [], [], 'r', lw=3)

# --- Ax2: Convolution Plot (f*g)(c) with Moving Dot ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title(
    'Convolution (f*g)(c)\n(f*g)(c) = ∫ f(x)·g(c-x) dx',
    fontsize=10, fontweight='bold'
)
ax2.set_xlabel('c', fontsize=10)
ax2.set_ylabel('(f*g)(c)', fontsize=10)
ax2.plot(c_vals, conv_vals, 'b-', lw=2)
conv_point, = ax2.plot([], [], 'ro', markersize=8)
annot_ax2 = ax2.text(0.05, 0.95, "", transform=ax2.transAxes,
                     color='r', fontsize=10, ha='left', va='top')

# --- Ax3: Cross-Section φ(x) = f(x)·g(c-x) ---
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_title(
    'Cross-section: φ(x) = f(x)·g(x)',
    fontsize=10, fontweight='bold'
)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('φ(x)', fontsize=10)
ax3.set_xlim(-1.1, 1.1)
ax3.set_ylim(0, 0.6)
x_integ = np.linspace(-1, 1, 200)
initial_phi = f(x_integ) * g(c_vals[0] - x_integ)
line_cross, = ax3.plot(x_integ, initial_phi, 'm-', lw=2)
area_fill = ax3.fill_between(x_integ, initial_phi, color='m', alpha=0.3)
annot_ax3 = ax3.text(0.05, 0.85, "", transform=ax3.transAxes,
                     color='m', fontsize=10)

# --- Ax4: PDFs f(x) and g(c-x) ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title('PDFs: f(x) and g(c-x)', fontsize=10, fontweight='bold')
ax4.set_xlabel('x', fontsize=10)
ax4.set_ylabel('Value', fontsize=10)
ax4.set_xlim(x_min, x_max)
ax4.set_ylim(0, 1.1)
line_f, = ax4.plot(x_vals, f(x_vals), 'b-', lw=2, label='f(x)')
line_g, = ax4.plot([], [], 'g-', lw=2, label='g(c-x)')
ax4.legend(loc='upper right', fontsize=10)

# Use a mutable container for the fill to update it in the animation
global_area_fill = [area_fill]


# --------------------------
# 5. Animation Update Function
# --------------------------
def update(frame):
    c = c_vals[frame]

    # (a) Update the animated slice in Ax1 (3D surface)
    t_vals = np.linspace(x_min, x_max, 200)
    y_line = c - t_vals
    valid = (y_line >= y_min) & (y_line <= y_max)
    t_valid = t_vals[valid]
    y_valid = y_line[valid]
    z_line = f(t_valid) * g(y_valid)
    slice_line.set_data(t_valid, y_valid)
    slice_line.set_3d_properties(z_line)

    # (b) Update the moving point and annotation in Ax2 (convolution plot)
    conv_current = conv_vals[frame]
    conv_point.set_data([c], [conv_current])
    annot_ax2.set_text(f'(f*g)({c:.2f}) = {conv_current:.3f}')

    # (c) Update the cross-section φ(x) in Ax3
    phi = f(x_integ) * g(c - x_integ)
    line_cross.set_ydata(phi)
    global_area_fill[0].remove()
    global_area_fill[0] = ax3.fill_between(x_integ, phi, color='m', alpha=0.3)
    area_phi = trapezoid(phi, x_integ)
    line_integral = area_phi * np.sqrt(2)
    annot_ax3.set_text(
        f'∫φ(x)dx = (f*g)(c) = {area_phi:.3f}\n'
        f'3D line integral = √2·(f*g)(c) = {line_integral:.3f}'
    )

    # (d) Update g(c-x) in Ax4
    line_g.set_data(x_vals, g(c - x_vals))

    return slice_line, conv_point, line_cross, global_area_fill[0], annot_ax3, line_g, annot_ax2


# --------------------------
# 6. Create and Run the Animation
# --------------------------
anim = FuncAnimation(fig, update, frames=len(c_vals), interval=50, blit=False)
plt.tight_layout()
plt.show()
