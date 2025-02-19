import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats

# ---------------------------
# 1. Generate a Skewed Dataset
# ---------------------------
np.random.seed(42)
data = np.random.gamma(shape=2.0, scale=2.0, size=1000)  # Gamma distribution (skewed)


# ---------------------------
# 2. Define the Box-Cox Transformation Function
# ---------------------------
def boxcox_transform(y, lam):
    """Apply the Box-Cox transformation to data y with parameter lam."""
    # When lambda is nearly zero, use the natural logarithm.
    return np.where(np.abs(lam) < 1e-8, np.log(y), (y ** lam - 1) / lam)


# Define a range of lambda values for the animation (from 1 to 0)
lam_vals = np.linspace(1, 0, 100)

# ---------------------------
# 3. Set Up the Figure with Two Panels
# ---------------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Box-Cox Transformation: Transition from Skewed to Near-Normal Distribution", fontsize=16)

# Left Panel: Original Gamma Distribution (static)
n_left, bins_left, _ = ax_left.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax_left.set_title("Original Data Distribution (Gamma)")
ax_left.set_xlabel("Value")
ax_left.set_ylabel("Frequency")
# Annotate with the gamma distribution formula (for shape=2, scale=2)
gamma_formula = r"$f(x)=\frac{1}{4}\,x\,e^{-x/2}$"
ax_left.text(0.05, 0.95, gamma_formula, transform=ax_left.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
ax_left.grid(True)

# Right Panel: Transformed Distribution (animated)
# Initialize with lambda = 1 (almost the identity transformation)
init_trans = boxcox_transform(data, 1)
n_right, bins_right, _ = ax_right.hist(init_trans, bins=30, density=True, color='salmon', edgecolor='black', alpha=0.7)
ax_right.set_title("Transformed Data Distribution (Box-Cox)")
ax_right.set_xlabel("Transformed Value")
ax_right.set_ylabel("Density")
# Add an annotation for the Box-Cox formula and current lambda value (initially λ = 1)
boxcox_formula = (r"$y^{(\lambda)}=\frac{y^\lambda-1}{\lambda}$ for $\lambda\neq0$" + "\n" +
                  r"$y^{(0)}=\log(y)$" + "\n" +
                  f"$\\lambda = 1.00$")
ax_right.text(0.05, 0.95, boxcox_formula, transform=ax_right.transAxes, fontsize=12,
              verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
# Initialize a line for the fitted normal density curve (will be updated)
normal_line, = ax_right.plot([], [], 'k-', lw=2, label='Normal Fit')
ax_right.legend(loc='upper right')
ax_right.grid(True)


# ---------------------------
# 4. Animation Update Function
# ---------------------------
def update(frame):
    lam = lam_vals[frame]
    transformed = boxcox_transform(data, lam)

    # Clear the right panel and replot the histogram of the transformed data
    ax_right.cla()
    ax_right.hist(transformed, bins=bins_right, density=True, color='salmon', edgecolor='black', alpha=0.7,
                  label='Transformed Data')
    ax_right.set_xlabel("Transformed Value")
    ax_right.set_ylabel("Density")
    ax_right.set_title("Transformed Data Distribution (Box-Cox)")

    # Fit a normal distribution to the transformed data and overlay the density curve
    mu, sigma = stats.norm.fit(transformed)
    x = np.linspace(bins_right[0], bins_right[-1], 100)
    p = stats.norm.pdf(x, mu, sigma)
    ax_right.plot(x, p, 'k-', lw=2, label=f'Normal Fit\n$\\mu={mu:.2f}, \\sigma={sigma:.2f}$')

    # Annotate with the Box-Cox transformation formula and current lambda value
    annotation = (r"$y^{(\lambda)}=\frac{y^\lambda-1}{\lambda}$ for $\lambda\neq0$" + "\n" +
                  r"$y^{(0)}=\log(y)$" + "\n" +
                  f"$\\lambda = {lam:.2f}$")
    ax_right.text(0.05, 0.95, annotation, transform=ax_right.transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    ax_right.legend(loc='upper right')
    ax_right.grid(True)
    return []


# ---------------------------
# 5. Create the Animation
# ---------------------------
ani = animation.FuncAnimation(fig, update, frames=len(lam_vals), interval=100, repeat=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
