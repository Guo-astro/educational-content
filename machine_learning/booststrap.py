import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from scipy.stats import gamma as gamma_dist
from scipy.stats import gaussian_kde

# -----------------------------
#   1. Global Settings
# -----------------------------
np.random.seed(42)
shape, scale = 2.0, 2.0     # Gamma distribution parameters
num_frames = 50             # Number of animation frames
max_sample_size = 5000      # Maximum sample size in the animation
large_sample_size = 100_000 # For static histograms in the left column
x_min, x_max = 0, 20        # Range for the Gamma distribution
bins_list = [
    np.linspace(x_min, x_max, 11),  # ~10 bins
    np.linspace(x_min, x_max, 51),  # ~20 bins
    np.linspace(x_min, x_max, 1001)   # ~30 bins
]

# For the theoretical (continuous) PDF and for plotting
x_grid = np.linspace(x_min, x_max, 200)

# -----------------------------
#   2. Prepare Data for Static Plots (Left Column)
# -----------------------------
# Generate a large sample to make the static histograms stable
large_data = np.random.gamma(shape, scale, size=large_sample_size)

# For row 3 (theoretical PDF)
pdf_vals = gamma_dist(a=shape, scale=scale).pdf(x_grid)

# -----------------------------
#   3. Set Up Figure and Axes Using Gridspec
# -----------------------------
fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 1])

# Left Column: Static Plots (4 rows)
axes_static = [fig.add_subplot(gs[i, 0]) for i in range(4)]
# Middle Column: Animated Plots (4 rows)
axes_anim = [fig.add_subplot(gs[i, 1]) for i in range(4)]
# Right Column: Animated Empirical CDF (one axis spanning all rows)
ax_cdf = fig.add_subplot(gs[:, 2])

# -----------------------------
#   4. Left Column: Static Plots with Math Annotation
# -----------------------------
# Rows 0, 1, 2: Histograms with different bin sizes
for i in range(3):
    counts, edges = np.histogram(large_data, bins_list[i], density=True)
    axes_static[i].bar(
        edges[:-1], counts,
        width=(edges[1] - edges[0]),
        align='edge', color='gray', alpha=0.7
    )
    axes_static[i].set_title(f"Static Gamma Histogram\n~{bins_list[i].size-1} bins")
    axes_static[i].set_xlim(x_min, x_max)
    axes_static[i].set_ylim(0, 0.4)
    axes_static[i].set_xlabel("Value")
    axes_static[i].set_ylabel("Density")
    # Add math text annotation for the empirical distribution function
    axes_static[i].text(
        0.05, 0.95,
        r'$F_n(x)=\frac{1}{n}\sum_{i=1}^{n}\mathbb{1}_{\{X_i\leqq  x\}}$',
        transform=axes_static[i].transAxes, fontsize=10, verticalalignment='top'
    )

# Row 3: Theoretical (continuous) Gamma PDF
axes_static[3].plot(x_grid, pdf_vals, color='red', lw=2)
axes_static[3].set_title("Theoretical Gamma PDF")
axes_static[3].set_xlim(x_min, x_max)
axes_static[3].set_ylim(0, 0.4)
axes_static[3].set_xlabel("Value")
axes_static[3].set_ylabel("Density")

# -----------------------------
#   5. Middle Column: Animated Plots
# -----------------------------
# Animated histograms for rows 0, 1, 2
bar_containers = []
for i in range(3):
    bar_obj = axes_anim[i].bar(
        bins_list[i][:-1],
        np.zeros(len(bins_list[i]) - 1),
        width=(bins_list[i][1] - bins_list[i][0]),
        align='edge', color='blue', alpha=0.6
    )
    bar_containers.append(bar_obj)
    axes_anim[i].set_xlim(x_min, x_max)
    axes_anim[i].set_ylim(0, 0.4)
    axes_anim[i].set_xlabel("Value")
    axes_anim[i].set_ylabel("Density")
    axes_anim[i].set_title(f"Animated ~{bins_list[i].size-1} bins (n=0)")

# Animated KDE for row 3
(line_kde,) = axes_anim[3].plot([], [], color='blue', lw=2)
axes_anim[3].set_xlim(x_min, x_max)
axes_anim[3].set_ylim(0, 0.4)
axes_anim[3].set_xlabel("Value")
axes_anim[3].set_ylabel("Density")
axes_anim[3].set_title("Animated KDE (n=0)")

# -----------------------------
#   6. Right Column: Animated Empirical CDF Plot
# -----------------------------
# Create a line object for the empirical CDF; use steps-post style
line_cdf, = ax_cdf.plot([], [], drawstyle='steps-post', color='green', lw=2)
# Also plot the theoretical CDF (static)
line_cdf_theor, = ax_cdf.plot(x_grid, gamma_dist.cdf(x_grid, shape, scale), 'r--', lw=2)
ax_cdf.set_xlim(x_min, x_max)
ax_cdf.set_ylim(0, 1)
ax_cdf.set_xlabel("Value")
ax_cdf.set_ylabel("Cumulative Probability")
ax_cdf.set_title("Animated Empirical CDF (n=0)")

# -----------------------------
#   7. Animation Functions
# -----------------------------
def init():
    """Initialize animated artists to empty data."""
    # Initialize animated histogram bars
    for bars in bar_containers:
        for b in bars:
            b.set_height(0)
    # Initialize the KDE line
    line_kde.set_data([], [])
    # Initialize the CDF line
    line_cdf.set_data([], [])
    return []

def update(frame):
    """
    For each frame, draw a new sample of size n and update:
      - The 3 animated histograms (middle column rows 0..2)
      - The animated KDE (middle column row 3)
      - The animated empirical CDF (right column)
    """
    n = int((frame + 1) * max_sample_size / num_frames)
    data = np.random.gamma(shape, scale, size=n)

    # Update animated histograms in rows 0..2
    for i in range(3):
        counts, _ = np.histogram(data, bins_list[i], density=True)
        for bar, height in zip(bar_containers[i], counts):
            bar.set_height(height)
        axes_anim[i].set_title(f"Animated ~{bins_list[i].size-1} bins (n={n})")

    # Update the animated KDE in row 3 (middle column)
    kde = gaussian_kde(data)
    y_kde = kde(x_grid)
    line_kde.set_data(x_grid, y_kde)
    axes_anim[3].set_title(f"Animated KDE (n={n})")

    # Update the animated empirical CDF in the right column
    data_sorted = np.sort(data)
    y_cdf = np.arange(1, len(data_sorted)+1) / len(data_sorted)
    # Create step-data: start at (x_min,0), then (data, cdf), then (x_max,1)
    x_cdf = np.concatenate(([x_min], data_sorted, [x_max]))
    y_cdf_full = np.concatenate(([0], y_cdf, [1]))
    line_cdf.set_data(x_cdf, y_cdf_full)
    ax_cdf.set_title(f"Animated Empirical CDF (n={n})")

    updated_artists = []
    for bars in bar_containers:
        updated_artists.extend(bars)
    updated_artists.append(line_kde)
    updated_artists.append(line_cdf)
    return updated_artists

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=num_frames,
    init_func=init, blit=False, interval=300
)

plt.tight_layout()
plt.show()

# (Optional) Save the animation:
# ani.save("gamma_animation.mp4", writer="ffmpeg")
# or
# ani.save("gamma_animation.gif", writer="imagemagick")
