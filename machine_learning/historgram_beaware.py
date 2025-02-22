import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde

# --------------------------
# Construct a bimodal dataset
# --------------------------
np.random.seed(0)
data1 = np.random.normal(-2, 1, 500)  # First bump centered at -2
data2 = np.random.normal(-4, 1, 500)  # Second bump centered at -4
data = np.concatenate((data1, data2))
min_data, max_data = data.min(), data.max()

# --------------------------
# Compute a continuous PDF using KDE
# --------------------------
kde = gaussian_kde(data)
xs = np.linspace(min_data - 2, max_data + 2, 300)
kde_vals = kde(xs)
y_max = np.max(kde_vals) * 1.1  # Fixed y-axis range for all subplots

# --------------------------
# Create a 2x2 layout for subplots with a smaller figure size
# --------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# --------------------------
# Top-Left: Moderate Bin Size Histogram (Bin width = 0.5)
# --------------------------
ax1 = axs[0, 0]
bin_width_mod = 0.5
bins_mod = np.arange(min_data, max_data + bin_width_mod, bin_width_mod)
ax1.hist(data, bins=bins_mod, density=True, alpha=0.7,
         edgecolor='black', color='lightgreen')
ax1.set_title("Moderate Bin Size (Bin width = 0.5)")
ax1.set_xlabel("Value")
ax1.set_ylabel("Probability Density")
ax1.set_ylim(0, y_max)
ax1.grid(True)

# --------------------------
# Top-Right: Large Bin Size Histogram (Bin width = 1)
# --------------------------
ax2 = axs[0, 1]
bin_width_large = 1
bins_large = np.arange(min_data, max_data + bin_width_large, bin_width_large)
ax2.hist(data, bins=bins_large, density=True, alpha=0.7,
         edgecolor='black', color='skyblue')
ax2.set_title("Large Bin Size (Bin width = 1)")
ax2.set_xlabel("Value")
ax2.set_ylabel("Probability Density")
ax2.set_ylim(0, y_max)
ax2.grid(True)

# --------------------------
# Bottom-Left: Small Bin Size Histogram with Non-uniform Bin Sizes
# --------------------------
ax3 = axs[1, 0]
# Create non-uniform bins: finer bins between -5 and -3, coarser bins elsewhere
bins_small_left = np.linspace(min_data, -5, 10)
bins_small_middle = np.linspace(-5, -3, 30)
bins_small_right = np.linspace(-3, max_data, 10)
bins_small = np.unique(np.concatenate((bins_small_left, bins_small_middle, bins_small_right)))
ax3.hist(data, bins=bins_small, density=True, alpha=0.7,
         edgecolor='black', color='salmon')
ax3.set_title("Small Bin Size (Non-uniform bins)")
ax3.set_xlabel("Value")
ax3.set_ylabel("Probability Density")
ax3.set_ylim(0, y_max)
ax3.grid(True)

# --------------------------
# Bottom-Right: Animated Convergence to Continuous PDF (KDE)
# --------------------------
ax_anim = axs[1, 1]
ax_anim.set_xlabel("Value")
ax_anim.set_ylabel("Probability Density")
ax_anim.set_ylim(0, y_max)
ax_anim.grid(True)

num_frames = 20
bin_width_initial = 5  # Start with a large bin size
bin_width_final = 0.1  # Converge to a very fine bin size

def update(frame):
    ax_anim.cla()  # Clear the animated subplot
    # Interpolate bin width linearly between initial and final values
    current_bin_width = bin_width_initial - (bin_width_initial - bin_width_final) * frame / (num_frames - 1)
    bins_anim = np.arange(min_data, max_data + current_bin_width, current_bin_width)

    # Plot the normalized histogram (empirical density)
    ax_anim.hist(data, bins=bins_anim, density=True, alpha=0.7,
                 edgecolor='black', color='plum', label="Empirical Histogram")
    # Overlay the continuous PDF (KDE)
    ax_anim.plot(xs, kde_vals, 'r-', lw=2, label="Continuous PDF (KDE)")

    ax_anim.set_title(f"Animated Convergence\nBin width = {current_bin_width:.2f}")
    ax_anim.set_xlabel("Value")
    ax_anim.set_ylabel("Probability Density")
    ax_anim.set_ylim(0, y_max)
    ax_anim.grid(True)

    annotation_text = ("As bin width → 0,\n"
                       "the histogram converges to the continuous PDF.")
    ax_anim.text(0.05, 0.95, annotation_text, transform=ax_anim.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Adjust legend to avoid overlapping the plot area
    ax_anim.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              interval=500, repeat=True)

plt.tight_layout()
plt.show()
