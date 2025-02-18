import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapezoid
from math import pi, sqrt, exp

# ---------------------------------------------------------------------
# 1) Define the Hat and Sinc^2 PDFs
# ---------------------------------------------------------------------
def pdf_hat(x):
    """
    Triangular 'hat' PDF on [-1, 1]:
        f(x) = 1 - |x| for |x| < 1,
                0      otherwise.
    This integrates to 1 over [-1,1].
    """
    out = np.maximum(1 - np.abs(x), 0)
    return out

def pdf_sinc_sq(x):
    """
    A nonnegative 'sinc^2' PDF:
        g(x) = [sin(pi*x)/(pi*x)]^2,  with limit g(0)=1.
    Integrates to 1 on (-∞, ∞).
    """
    out = np.zeros_like(x)
    nonzero = (x != 0)
    out[nonzero] = (np.sin(pi*x[nonzero]) / (pi*x[nonzero]))**2
    out[~nonzero] = 1.0  # sinc^2(0) = 1
    return out

# Let's choose f(x)=hat(x) and g(x)=sinc^2(x).
f = pdf_hat
g = pdf_sinc_sq

# ---------------------------------------------------------------------
# 2) Set up a grid and compute the product f(x)*g(y)
#    We'll pick a domain large enough to see the main shape.
# ---------------------------------------------------------------------
x_min, x_max = -2.0, 2.0
y_min, y_max = -2.0, 2.0
n_points = 200

x_vals = np.linspace(x_min, x_max, n_points)
y_vals = np.linspace(y_min, y_max, n_points)
X, Y = np.meshgrid(x_vals, y_vals)

Z = f(X) * g(Y)  # product for 3D surface

# ---------------------------------------------------------------------
# 3) Convolution (f*g)(c) = ∫ f(x)*g(c - x) dx
#    We'll do a numeric approximation with trapezoid rule.
# ---------------------------------------------------------------------
def convolution(c):
    integrand = f(x_vals) * g(c - x_vals)
    return trapezoid(integrand, x_vals)

# Precompute the convolution on a range of c-values
c_min, c_max = -4.0, 4.0
c_vals = np.linspace(c_min, c_max, n_points)
conv_vals = np.array([convolution(ci) for ci in c_vals])

# ---------------------------------------------------------------------
# 4) Create a figure with 3 panels
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1,1], width_ratios=[1,1])

# Left subplot (3D) spans both rows in column 0
ax1 = fig.add_subplot(gs[:, 0], projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x)*g(y)')
ax1.set_title('Product f(x)*g(y) with slice x+y=c')

surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
slice_line, = ax1.plot([], [], [], 'r', lw=3)

# Top-right subplot: Convolution curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('Convolution (f*g)(c)')
ax2.set_xlabel('c')
ax2.set_ylabel('(f*g)(c)')
ax2.plot(c_vals, conv_vals, color='blue', lw=2)
current_point, = ax2.plot([], [], 'ro', markersize=6)

# Bottom-right subplot: f(x) and g(c - x)
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('f(x) and g(c - x)')
ax3.set_xlabel('x')
ax3.set_ylabel('Value')
ax3.set_xlim(x_min, x_max)
# We'll guess a suitable y-limit. The hat is at most 1 at x=0; sinc^2 also <=1.
ax3.set_ylim(0, 1.1)

line_f, = ax3.plot([], [], 'b-', label='f(x)')
line_g, = ax3.plot([], [], 'g-', label='g(c - x)')
ax3.legend(loc='upper right')

# ---------------------------------------------------------------------
# 5) Animation update function
# ---------------------------------------------------------------------
def update(frame):
    c = c_vals[frame]

    # (a) Update the slice x+y=c in 3D
    xx = np.linspace(x_min, x_max, 200)
    yy = c - xx
    valid_idx = (yy >= y_min) & (yy <= y_max)
    xx = xx[valid_idx]
    yy = yy[valid_idx]
    zz = f(xx) * g(yy)
    slice_line.set_data(xx, yy)
    slice_line.set_3d_properties(zz)

    # (b) Update the moving dot on the convolution curve
    current_point.set_data([c], [conv_vals[frame]])

    # (c) Update f(x) and g(c-x) lines
    line_f.set_data(x_vals, f(x_vals))
    line_g.set_data(x_vals, g(c - x_vals))

    return slice_line, current_point, line_f, line_g

# ---------------------------------------------------------------------
# 6) Create the animation
# ---------------------------------------------------------------------
anim = FuncAnimation(
    fig, update, frames=len(c_vals), interval=50, blit=False
    # Use blit=False if you see 3D + blitting issues
)

plt.tight_layout()
plt.show()
