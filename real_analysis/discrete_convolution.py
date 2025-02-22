import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import math

"""
Here, let X and Y be two real-life random variables representing (for example):
  - X = number of customers arriving in a small shop during a certain hour (λ_X=2)
  - Y = number of customers arriving in a larger, independent store in the same hour (λ_Y=4)

We model each with a Poisson distribution:
  X ~ Poisson(λ_X = 2),   Y ~ Poisson(λ_Y = 4),   and X, Y are independent.

The PMFs are:
  P(X = x) = e^{-λ_X} (λ_X^x / x!),   x = 0, 1, 2, ...
  P(Y = y) = e^{-λ_Y} (λ_Y^y / y!),   y = 0, 1, 2, ...

Because X and Y are independent,
  P(X = x, Y = y) = P(X = x) ⋅ P(Y = y).

We truncate X and Y to [0..8] for demonstration purposes, then the sum X+Y
ranges over [0..16]. This example shows:
  1) The PMFs of X and Y,
  2) The 3D joint distribution P(X=x, Y=y),
  3) The PMF of (X+Y) = discrete convolution of P(X) and P(Y),
  4) A step-by-step (sliding) demonstration of how that convolution is formed.
"""

# --------------------------------------------------------------
# 1. Define Poisson PMFs (truncated at a max value for plotting)
# --------------------------------------------------------------
def poisson_pmf(lam, k):
    """Compute P(Z=k) for Z ~ Poisson(lam), at integer k >= 0."""
    return (np.exp(-lam) * (lam**k) / math.factorial(k))

lambda_X = 2.0
lambda_Y = 4.0

max_x = 8  # Truncate X in [0..8]
max_y = 8  # Truncate Y in [0..8]

x_support = np.arange(max_x+1)  # 0..8
y_support = np.arange(max_y+1)  # 0..8

pmf_X = np.array([poisson_pmf(lambda_X, k) for k in x_support])
pmf_Y = np.array([poisson_pmf(lambda_Y, k) for k in y_support])

# --------------------------------------------------------------
# 2. Construct the Joint Distribution and Convolution
# --------------------------------------------------------------
# Because X, Y independent => joint P(X=x, Y=y) = pmf_X[x] * pmf_Y[y].
Z = np.zeros((max_y+1, max_x+1))
for i, x in enumerate(x_support):
    for j, y in enumerate(y_support):
        Z[j, i] = pmf_X[x] * pmf_Y[y]

# Discrete convolution for P(X+Y = n)
pmf_sum = np.convolve(pmf_X, pmf_Y)
sum_support = np.arange(len(pmf_sum))  # 0..(max_x+max_y)=16
n_frames = len(pmf_sum)               # 17 frames

# --------------------------------------------------------------
# 3. Figure Layout
# --------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
plt.suptitle("Two Poisson Random Variables and Their Convolution",
             fontsize=10, fontweight='bold')

gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1])

# (a) Axes for X's PMF
ax_X = fig.add_subplot(gs[0, 0])
ax_X.set_title("PMF of X ~ Poisson(λ=2)", fontsize=10, fontweight='bold')
ax_X.set_xlabel("x")
ax_X.set_ylabel("P(X=x)")
ax_X.set_ylim(0, pmf_X.max() * 1.3)
ax_X.bar(x_support, pmf_X, color='blue', alpha=0.7)
ax_X.set_xticks(x_support)

# (b) Axes for Y's PMF
ax_Y = fig.add_subplot(gs[0, 1])
ax_Y.set_title("PMF of Y ~ Poisson(λ=4)", fontsize=10, fontweight='bold')
ax_Y.set_xlabel("y")
ax_Y.set_ylabel("P(Y=y)")
ax_Y.set_ylim(0, pmf_Y.max() * 1.3)
ax_Y.bar(y_support, pmf_Y, color='green', alpha=0.7)
ax_Y.set_xticks(y_support)

# (c) 3D Joint Distribution
ax3d = fig.add_subplot(gs[1, :], projection='3d')
ax3d.set_title("Joint Distribution P(X=x, Y=y)", fontsize=10, fontweight='bold')
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Probability")
ax3d.set_xlim(-0.5, max_x+0.5)
ax3d.set_ylim(-0.5, max_y+0.5)
ax3d.set_zlim(0, Z.max() * 1.3)

dx = dy = 0.8
# Pre-draw the bars (will recolor in update)
bars = []
for i, x in enumerate(x_support):
    for j, y in enumerate(y_support):
        b = ax3d.bar3d(
            x - 0.4,  # shift so bar is centered at x
            y - 0.4,
            0,
            dx,
            dy,
            Z[j, i],
            color='blue',
            alpha=0.1
        )
        bars.append(b)

# (d) Convolution PMF: P(X+Y = n)
ax_conv = fig.add_subplot(gs[2, 0])
ax_conv.set_title("PMF of (X+Y) via Convolution", fontsize=10, fontweight='bold')
ax_conv.set_xlabel("n = x+y")
ax_conv.set_ylabel("Probability")
ax_conv.set_xlim(-0.5, max_x+max_y+0.5)   # up to 16
ax_conv.set_ylim(0, pmf_sum.max() * 1.3)

stem_marker, stem_lines, baseline = ax_conv.stem(sum_support, pmf_sum,
                                                 basefmt=" ",
                                                 linefmt='r-',
                                                 markerfmt='ro')
moving_point, = ax_conv.plot([], [], 'bo', markersize=8)

# (e) Step-by-Step (Sliding) Convolution
ax_explicit = fig.add_subplot(gs[2, 1])
ax_explicit.set_title("Step-by-Step Convolution Demonstration", fontsize=10, fontweight='bold')
ax_explicit.set_xlabel("Index for X (f)")
ax_explicit.set_ylabel("Probability")
ax_explicit.set_xlim(-2, max_x + max_y)  # give room to slide
ax_explicit.set_ylim(0, 0.3)

# Create "dummy" bars so legend is visible initially
f_dummy = ax_explicit.bar(x_support - 0.2, pmf_X, width=0.4, color='blue', label='f(x)=P(X=x)')
g_dummy = ax_explicit.bar(x_support, np.zeros_like(x_support), width=0.4, color='green', label='g(x - k)=P(Y=x-k)')
p_dummy = ax_explicit.bar(x_support + 0.2, np.zeros_like(x_support), width=0.4, color='magenta', label='f*g')
ax_explicit.legend(fontsize=9, loc='upper right')

# --------------------------------------------------------------
# 4. Animation Update Function
# --------------------------------------------------------------
def update(frame):
    """
    frame goes from 0..16, corresponding to sum_support=0..16.
    We'll highlight the current sum in:
      1) the Convolution PMF (with a moving dot),
      2) the 3D plot bars where x+y = current_sum,
      3) the Step-by-Step demonstration with a shift k.
    We also rotate the 3D plot.
    """
    current_sum = sum_support[frame]
    conv_value = pmf_sum[frame]

    # (a) Convolution PMF: Move the blue dot
    moving_point.set_data([current_sum], [conv_value])

    # (b) 3D Joint Distribution: recolor bars whose x+y = current_sum
    # Rotate the view a bit each frame

    idx = 0
    for i, x in enumerate(x_support):
        for j, y in enumerate(y_support):
            if x + y == current_sum:
                bars[idx].remove()  # remove old bar
                bars[idx] = ax3d.bar3d(
                    x - 0.4, y - 0.4, 0, dx, dy, Z[j, i],
                    color='red', alpha=0.8
                )
            else:
                bars[idx].remove()
                bars[idx] = ax3d.bar3d(
                    x - 0.4, y - 0.4, 0, dx, dy, Z[j, i],
                    color='blue', alpha=0.1
                )
            idx += 1

    # (c) Step-by-step convolution:
    # We'll let shift = current_sum - "center" so that when current_sum=8, shift=0.
    k = current_sum - 8  # you can adjust the '8' to recenter differently
    ax_explicit.cla()
    ax_explicit.set_title("Convolution Demonstration", fontsize=10, fontweight='bold')
    ax_explicit.set_xlabel("Index for X (f)")
    ax_explicit.set_ylabel("Probability")
    ax_explicit.set_xlim(-2, max_x + max_y)
    ax_explicit.set_ylim(0, 0.3)

    # We shift pmf_Y by k, so effectively we are looking at g(x-k)
    # We'll overlay that on the same integer grid as f(x).
    x_green = y_support + k  # domain shift

    # Overlapping region: product f(x)*g(current_sum - x)
    product_vals = []
    for x in x_support:
        y_idx = x - k
        if 0 <= y_idx < len(pmf_Y):
            product_vals.append(pmf_X[x] * pmf_Y[y_idx])
        else:
            product_vals.append(0)

    product_vals = np.array(product_vals)
    conv_calc = product_vals.sum()

    # Re-plot the bars
    ax_explicit.bar(x_support - 0.2, pmf_X, width=0.4, color='blue', label='f(x)')
    ax_explicit.bar(x_green, pmf_Y, width=0.4, color='green', label='g(x-k)')
    ax_explicit.bar(x_support + 0.2, product_vals, width=0.4, color='magenta', label='product')

    # Annotate with sum, shift, and convolution value
    ax_explicit.text(
        0.55, 0.15,
        f"Current sum = {current_sum}\n"
        f"Shift (k) = {k}\n"
        f"Sum of products = {conv_calc:.3f}",
        transform=ax_explicit.transAxes, fontsize=9, color='black',
        bbox=dict(facecolor='white', alpha=0.6)
    )
    ax_explicit.legend(fontsize=9, loc='upper right')

    return (moving_point,)
ax3d.view_init(elev=30, azim=45)  # Rotate around the z-axis

# --------------------------------------------------------------
# 5. Create and Run the Animation
# --------------------------------------------------------------
anim = FuncAnimation(
    fig, update,
    frames=n_frames,  # 17 frames for sums in 0..16
    interval=1000,    # ms between frames
    blit=False
)

plt.tight_layout()
plt.show()
