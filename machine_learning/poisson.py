import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from math import factorial, log
from scipy.spatial.distance import jensenshannon
from scipy.signal import welch
from typing import List, Tuple

# -------------------------------
# Utility Functions for CH–Plane & PSD Analysis
# -------------------------------

def ordinal_patterns(time_series: np.ndarray, d: int, tau: int = 1) -> List[Tuple[int, ...]]:
    """Construct ordinal patterns from a time series."""
    N = len(time_series)
    patterns = []
    for i in range(N - (d - 1) * tau):
        window = time_series[i: i + d * tau: tau]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)
    return patterns

def permutation_entropy(time_series: np.ndarray, d: int, tau: int = 1, normalize: bool = True) -> Tuple[float, np.ndarray]:
    """Compute the permutation entropy of a time series."""
    patterns = ordinal_patterns(time_series, d, tau)
    counter = Counter(patterns)
    total = sum(counter.values())
    p_vals = np.array([count / total for count in counter.values()])
    H = -np.sum(p_vals * np.log(p_vals))
    if normalize:
        H /= log(factorial(d))
    return H, p_vals

def jensen_shannon_complexity(p_vals: np.ndarray, d: int) -> float:
    """Compute statistical complexity using Jensen–Shannon divergence."""
    n = factorial(d)
    full_p = np.zeros(n)
    full_p[:len(p_vals)] = p_vals  # pad with zeros if necessary
    p_uniform = np.ones(n) / n
    jsd = jensenshannon(full_p, p_uniform, base=np.e)
    H_norm = -np.sum(full_p[full_p > 0] * np.log(full_p[full_p > 0])) / log(n)
    complexity = jsd * H_norm
    return complexity

def compute_CH_plane(time_series: np.ndarray, d: int, tau: int = 1) -> Tuple[float, float]:
    """Compute normalized permutation entropy and complexity (CH-plane coordinates)."""
    H, p_vals = permutation_entropy(time_series, d, tau)
    C = jensen_shannon_complexity(p_vals, d)
    return H, C

def compute_psd(time_series: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the power spectral density (PSD) using Welch's method."""
    f, Pxx = welch(time_series, fs=fs, nperseg=min(256, len(time_series)))
    return f, Pxx

# -------------------------------
# Simulation Functions
# -------------------------------

def simulate_gbm(S0: float, mu: float, sigma: float, T: float, N: int) -> np.ndarray:
    """
    Simulate a GBM path.
    Returns an array of prices of length N+1.
    """
    dt = T / N
    t = np.linspace(0, T, N + 1)
    # Generate increments and cumulative sum for W
    W = np.random.normal(0, np.sqrt(dt), N).cumsum()
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t[1:] + sigma * W)
    return np.concatenate(([S0], S))

def simulate_non_martingale(S0: float, mu: float, sigma: float, T: float, N: int, phi: float = 0.8) -> np.ndarray:
    """
    Simulate a non-martingale process using an AR(1) model (introducing memory).
    Returns an array of process values of length N+1.
    """
    dt = T / N
    X = np.empty(N + 1)
    X[0] = S0
    noise = np.random.normal(0, sigma * np.sqrt(dt), N)
    for i in range(1, N + 1):
        # AR(1) term introduces persistence/memory
        X[i] = X[i - 1] + mu * dt + phi * (X[i - 1] - S0) + noise[i - 1]
    return X

def compute_returns(prices: np.ndarray, log_return: bool = True) -> np.ndarray:
    """
    Compute returns from a price series.
    If log_return is True, returns are log differences;
    otherwise, simple differences.
    """
    if log_return:
        return np.diff(np.log(prices))
    else:
        return np.diff(prices)

# -------------------------------
# Animation Function for Returns Comparison
# -------------------------------

def animate_returns(df: pd.DataFrame, columns: List[str], d: int, tau: int, fs: float, title: str, color_list: List[str], min_points: int = 20) -> None:
    """
    Animate returns with:
      - Left: Evolving returns time series.
      - Top Right: CH–plane trajectories for each return series (different colors).
      - Bottom Right: PSD spectra (scatter and fit lines) for each return series, shown in their own colors.
    """
    ts_list = [df[col].to_numpy() for col in columns]
    N = len(ts_list[0])
    time_vec = np.arange(N)

    # Histories for CH-plane for each simulation
    H_history = [[] for _ in columns]
    C_history = [[] for _ in columns]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    ax_left = fig.add_subplot(gs[:, 0])
    ax_ch = fig.add_subplot(gs[0, 1])
    ax_psd = fig.add_subplot(gs[1, 1])

    # Left panel: returns time series
    ret_lines = [ax_left.plot([], [], color=color_list[i], lw=2, label=f"Return {i + 1}")[0] for i in range(len(columns))]
    ax_left.set_xlim(0, N)
    all_returns = np.concatenate(ts_list)
    ax_left.set_ylim(np.min(all_returns) * 1.1, np.max(all_returns) * 1.1)
    ax_left.set_title(title)
    ax_left.set_xlabel("Time Step")
    ax_left.set_ylabel("Return")
    ax_left.legend()

    # Top Right: CH-plane trajectories
    ch_lines = [ax_ch.plot([], [], color=color_list[i], lw=2, label=f"Return {i + 1}")[0] for i in range(len(columns))]
    ax_ch.set_xlim(0, 1)
    ax_ch.set_ylim(0, 0.6)
    ax_ch.set_title("Complexity–Entropy Causality Plane")
    ax_ch.set_xlabel("Normalized Permutation Entropy")
    ax_ch.set_ylabel("Statistical Complexity")
    ax_ch.legend(loc="upper right")

    # Bottom Right: PSD panel will show both traces with different colors
    ax_psd.set_xscale('log')
    ax_psd.set_yscale('log')
    ax_psd.set_xlim(1e-3, 1)
    ax_psd.set_ylim(1e-3, 1)
    ax_psd.set_title("PSD Spectrum (Returns)")
    ax_psd.set_xlabel("Frequency")
    ax_psd.set_ylabel("PSD")

    formulas_text = (
        r"Returns: For GBM, $r_t=\ln(S_t/S_{t-1})$; For arithmetic, $r_t=X_t-X_{t-1}$" "\n"
        r"Permutation Entropy: $H(P)=-\sum p(\pi)\ln p(\pi)$ normalized by $\ln(d!)$" "\n"
        r"Statistical Complexity: $C_{JS}=D_{JS}(P,P_e)\times \frac{H(P)}{\ln(d!)}$" "\n"
        r"PSD: $S(f)\propto1/f^\alpha$"
    )
    fig.text(0.5, 0.02, formulas_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    def init():
        for line in ret_lines:
            line.set_data([], [])
        for line in ch_lines:
            line.set_data([], [])
        ax_psd.cla()
        ax_psd.set_xscale('log')
        ax_psd.set_yscale('log')
        ax_psd.set_xlim(1e-3, 1)
        ax_psd.set_ylim(1e-3, 1)
        ax_psd.set_title("PSD Spectrum (Returns)")
        ax_psd.set_xlabel("Frequency")
        ax_psd.set_ylabel("PSD")
        return ret_lines + ch_lines

    def update(frame):
        artists = []
        # Update left panel: returns time series
        for i in range(len(columns)):
            x_data = time_vec[:frame + 1]
            y_data = ts_list[i][:frame + 1]
            ret_lines[i].set_data(x_data, y_data)
            artists.append(ret_lines[i])
        if frame >= min_points:
            # Update CH-plane trajectories
            for i in range(len(columns)):
                current_returns = ts_list[i][:frame + 1]
                try:
                    H_val, C_val = compute_CH_plane(current_returns, d, tau)
                except Exception:
                    H_val, C_val = 0.0, 0.0
                H_history[i].append(H_val)
                C_history[i].append(C_val)
                ch_lines[i].set_data(H_history[i], C_history[i])
                artists.append(ch_lines[i])
            # Update PSD panel for each return series
            ax_psd.cla()
            ax_psd.set_xscale('log')
            ax_psd.set_yscale('log')
            ax_psd.set_xlim(1e-3, 1)
            ax_psd.set_ylim(1e-3, 1)
            ax_psd.set_title("PSD Spectrum (Returns)")
            ax_psd.set_xlabel("Frequency")
            ax_psd.set_ylabel("PSD")
            for i in range(len(columns)):
                current_series = ts_list[i][:frame + 1]
                f_vals, Pxx_vals = compute_psd(current_series, fs)
                valid = f_vals > 0
                if np.any(valid):
                    f_valid = f_vals[valid]
                    Pxx_valid = Pxx_vals[valid]
                    # Scatter plot the PSD points for this trace
                    ax_psd.scatter(f_valid, Pxx_valid, color=color_list[i], s=10, label=f"Trace {i+1} PSD")
                    # Fit a line if enough points exist
                    if np.sum(valid) > 10:
                        log_f = np.log(f_valid)
                        log_Pxx = np.log(Pxx_valid)
                        coeffs = np.polyfit(log_f, log_Pxx, 1)
                        alpha = -coeffs[0]
                        b = coeffs[1]
                        f_fit = np.linspace(np.min(f_valid), np.max(f_valid), 100)
                        Pxx_fit = np.exp(b) * f_fit ** (-alpha)
                        ax_psd.plot(f_fit, Pxx_fit, '--', color=color_list[i], lw=2,
                                    label=f"Trace {i+1} Fit (α={alpha:.2f})")
            ax_psd.legend(loc="upper right")
        return artists

    ani = animation.FuncAnimation(fig, update, frames=N, init_func=init,
                                  blit=False, interval=50, repeat=False)
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.show()

# -------------------------------
# Main Execution: Simulate, Compute Returns & Animate
# -------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    # Simulation parameters
    S0 = 100.0     # initial value
    mu = 0.05      # drift
    sigma = 0.2    # volatility
    T = 1          # time horizon (1 year)
    N = 252        # number of time steps (trading days)
    d = 4          # embedding dimension for CH analysis
    tau = 1        # time delay
    fs = 1.0       # sampling frequency

    # --- GBM (Martingale-like process) ---
    # Generate 2 independent GBM series and compute log returns
    prices_gbm1 = simulate_gbm(S0, mu, sigma, T, N)
    prices_gbm2 = simulate_gbm(S0, mu, sigma, T, N)
    returns_gbm1 = compute_returns(prices_gbm1, log_return=True)
    returns_gbm2 = compute_returns(prices_gbm2, log_return=True)
    df_gbm_returns = pd.DataFrame({
        'GBM Return 1': returns_gbm1,
        'GBM Return 2': returns_gbm2
    })

    # --- Non-Martingale Process (with AR(1) colored noise) ---
    # Generate 2 independent non-martingale series and compute simple differences
    proc_nm1 = simulate_non_martingale(S0, mu, sigma, T, N, phi=0.8)
    proc_nm2 = simulate_non_martingale(S0, mu, sigma, T, N, phi=0.8)
    returns_nm1 = compute_returns(proc_nm1, log_return=False)
    returns_nm2 = compute_returns(proc_nm2, log_return=False)
    df_nm_returns = pd.DataFrame({
        'Non-Martingale Return 1': returns_nm1,
        'Non-Martingale Return 2': returns_nm2
    })

    # Animate returns for GBM with a chosen color list (for left, CH-plane, and PSD)
    print("Animating GBM (Martingale) Returns...")
    animate_returns(df_gbm_returns, ['GBM Return 1', 'GBM Return 2'],
                    d, tau, fs, title="GBM Returns (Log Returns)", color_list=['blue', 'green'])

    # Animate returns for Non-Martingale Process with a different color list
    print("Animating Non-Martingale Returns...")
    animate_returns(df_nm_returns, ['Non-Martingale Return 1', 'Non-Martingale Return 2'],
                    d, tau, fs, title="Non-Martingale Returns (Arithmetic Differences)", color_list=['red', 'magenta'])
