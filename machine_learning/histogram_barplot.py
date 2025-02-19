import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate some continuous (normally distributed) data
data = np.random.randn(1000)
min_data, max_data = data.min(), data.max()

# Set up the figure and two subplots:
# - Left: density histogram (continuous data distribution)
# - Right: bar plot of counts (for categorical or binned data)
fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))


def update(frame):
    ax_hist.clear()
    ax_bar.clear()

    # Change bin width as the animation progresses.
    # Here, bin_width increases with each frame.
    bin_width = 0.1 + frame * 0.05
    bins = np.arange(min_data, max_data + bin_width, bin_width)

    # Plot the histogram with density=True
    # This scales the bar heights so that the total area is 1.
    n, bins_out, patches = ax_hist.hist(data, bins=bins, density=True,
                                        alpha=0.7, edgecolor='black')
    ax_hist.set_title(f"Histogram (density=True)\nBin width = {bin_width:.2f}")
    ax_hist.set_xlim(min_data, max_data)

    # Now plot a bar plot of raw counts.
    # Note: In a bar plot the bar heights represent counts directly.
    counts, _ = np.histogram(data, bins=bins)
    # The bar plot shows counts without adjusting for bin width.
    ax_bar.bar(bins[:-1], counts, width=bin_width, align='edge',
               alpha=0.7, edgecolor='black')
    ax_bar.set_title("Bar plot (counts)")
    ax_bar.set_xlim(min_data, max_data)

    # Label y-axes for clarity
    ax_hist.set_ylabel("Probability Density")
    ax_bar.set_ylabel("Counts")


# Create the animation: update() is called for each frame.
ani = animation.FuncAnimation(fig, update, frames=np.arange(1, 20),
                              interval=500, repeat=True)

plt.tight_layout()
plt.show()
