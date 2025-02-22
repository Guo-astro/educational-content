import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set a seed for reproducibility
np.random.seed(0)

# Generate continuous (normally distributed) data
data = np.random.randn(1000)
min_data, max_data = data.min(), data.max()

# Set up the figure with two subplots:
# Left: density histogram (probability density, area = 1)
# Right: bar plot (raw counts per bin)
fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))


def update(frame):
    # Clear previous plots
    ax_hist.cla()
    ax_bar.cla()

    # Adjust bin width for each frame
    bin_width = 0.1 + frame * 0.05
    bins = np.arange(min_data, max_data + bin_width, bin_width)

    # Density histogram: normalization ensures area = 1.
    # Height is given by count / (1000 * bin_width)
    ax_hist.hist(data, bins=bins, density=True, alpha=0.7,
                 edgecolor='black', label="Empirical Histogram")

    # Compute x-range for the true PDF plot
    xs = np.linspace(min_data - 1, max_data + 1, 300)
    true_pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-xs ** 2 / 2)

    # Overlay the true PDF for the standard normal distribution
    ax_hist.plot(xs, true_pdf, 'r-', lw=2, label='True PDF (N(0,1))')

    ax_hist.set_title(f"Density Histogram (Normalized)\nBin width = {bin_width:.2f}")
    ax_hist.set_xlim(min_data, max_data)
    ax_hist.set_ylabel("Probability Density")
    ax_hist.legend()
    ax_hist.grid(True)

    # Add annotation explaining the concept
    annotation_text = (
        "Note:\n"
        "• Histogram appearance depends on bin size.\n"
        "• PDF is a standard measure, invariant to bin width."
    )
    ax_hist.text(0.05, 0.95, annotation_text, transform=ax_hist.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Bar plot: raw counts in each bin.
    counts, _ = np.histogram(data, bins=bins)
    ax_bar.bar(bins[:-1], counts, width=bin_width, align='edge',
               alpha=0.7, edgecolor='black')
    ax_bar.set_title("Bar Plot (Raw Counts)")
    ax_bar.set_xlim(min_data, max_data)
    ax_bar.set_ylabel("Counts")
    ax_bar.grid(True)


# Create the animation: update() is called for each frame.
ani = animation.FuncAnimation(fig, update, frames=np.arange(1, 20),
                              interval=500, repeat=True)

plt.tight_layout()
plt.show()
