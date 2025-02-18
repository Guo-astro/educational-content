import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import statsmodels.api as sm

# -----------------------------
# 1. Load real data: Sunspots
# -----------------------------
sun_data = sm.datasets.sunspots.load_pandas().data
# Subset: years 1900–1950
sun_data = sun_data[(sun_data.YEAR >= 1900) & (sun_data.YEAR <= 1950)]
ts = sun_data['SUNACTIVITY'].values
years = sun_data['YEAR'].values

# -----------------------------
# 2. Manual ACF calculation
# -----------------------------
def manual_acf(series, max_lag):
    """
    Compute autocorrelation up to 'max_lag' by direct summation:
      ACF(k) = Corr(X_t, X_{t-k}) for t = k..(n-1).
    """
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)

    acf_vals = []
    for lag in range(max_lag + 1):
        c = 0.0
        for t in range(lag, n):
            c += (series[t] - mean) * (series[t - lag] - mean)
        # denominator: var * (n - lag)
        acf_vals.append(c / (var * (n - lag)))
    return acf_vals

# -----------------------------
# 3. Manual PACF calculation
#    via the "residual" or
#    "sequential regression" approach
# -----------------------------
def partial_corr_regression(series, k):
    """
    Compute partial correlation at lag k by:
      1) Regress X_t on [X_{t-1}, ..., X_{t-(k-1)}] => get residual e_t
      2) Regress X_{t-k} on [X_{t-1}, ..., X_{t-(k-1)}] => get residual e_{t-k}
      3) partial_corr = Corr(e_t, e_{t-k})

    Returns: partial_corr, e_t, e_tk (the residual vectors)
    """
    n = len(series)
    # We need at least k data points to do lag k
    if k == 0 or k >= n:
        return np.nan, None, None

    # Build design matrix for times t in [k..n-1]
    # Each row: [X_{t-1}, X_{t-2}, ..., X_{t-(k-1)}]
    # We'll skip lag k itself to isolate the smaller lags
    Y = series[k:]  # shape: (n-k,)
    M = []
    for t in range(k, n):
        row = []
        for j in range(1, k):
            row.append(series[t - j])
        M.append(row)

    M = np.array(M)  # shape: (n-k, k-1)
    # Regress Y on M => e_t
    # If k=1, M is empty => e_t = Y - mean(Y)
    if k == 1:
        # No smaller lags, so partial corr at lag=1 is just the correlation with lag 1
        # We'll handle it in the same approach for consistency:
        e_t = Y - np.mean(Y)
    else:
        # OLS for Y ~ M
        # We'll do a quick closed-form solution: beta = (M'M)^{-1} M'Y
        # But let's handle intercept by adding a column of 1's
        intercept_col = np.ones((M.shape[0], 1))
        M2 = np.hstack([intercept_col, M])  # shape: (n-k, k)
        beta = np.linalg.lstsq(M2, Y, rcond=None)[0]  # shape: (k,)
        Y_hat = M2 @ beta
        e_t = Y - Y_hat

    # Similarly, define Z = series[t-k] for t in [k..n-1]
    Z = []
    for t in range(k, n):
        Z.append(series[t - k])
    Z = np.array(Z)  # shape: (n-k,)

    if k == 1:
        e_tk = Z - np.mean(Z)
    else:
        # Regress Z on M (same design matrix M2)
        beta_z = np.linalg.lstsq(M2, Z, rcond=None)[0]
        Z_hat = M2 @ beta_z
        e_tk = Z - Z_hat

    # partial correlation is correlation(e_t, e_tk)
    corr = np.corrcoef(e_t, e_tk)[0, 1]
    return corr, e_t, e_tk

def manual_pacf(series, max_lag):
    """
    Compute partial autocorrelations for k=0..max_lag
    using the sequential regression approach.
    Returns an array of length (max_lag+1).
    """
    pacf_vals = []
    for k in range(max_lag + 1):
        if k == 0:
            pacf_vals.append(1.0)  # by definition
        else:
            c, _, _ = partial_corr_regression(series, k)
            pacf_vals.append(c)
    return pacf_vals

# -----------------------------
# 4. Shifted series for ACF
# -----------------------------
def shifted_series(series, lag):
    """
    Return a new array the same length as 'series',
    but shifted by 'lag' positions.
    The first 'lag' entries are set to np.nan for display.
    """
    if lag == 0:
        return series.copy()
    n = len(series)
    shifted = np.empty(n)
    shifted[:lag] = np.nan
    shifted[lag:] = series[:n - lag]
    return shifted

# -----------------------------
# 5. Precompute ACF & PACF
# -----------------------------
max_lag = 12
acf_vals = manual_acf(ts, max_lag)      # length max_lag+1
pacf_vals = manual_pacf(ts, max_lag)    # length max_lag+1

# We'll store the residual e_t vs. e_{t-k} for each lag
residual_pairs = {}  # lag -> (e_t, e_tk)
for k in range(1, max_lag + 1):
    corr_k, e_t, e_tk = partial_corr_regression(ts, k)
    residual_pairs[k] = (e_t, e_tk)

# -----------------------------
# 6. Setup figure
#  2x2 subplots:
#   [0,0]: Show original vs. shifted for ACF
#   [0,1]: ACF bar chart
#   [1,0]: Residual scatter for partial correlation at lag k
#   [1,1]: PACF bar chart
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax_shift = axes[0, 0]  # original vs. shifted
ax_acf   = axes[0, 1]  # ACF bars
ax_resid = axes[1, 0]  # scatter of e_t vs. e_{t-k}
ax_pacf  = axes[1, 1]  # PACF bars

ax_shift.set_title("Time Series & Shifted (for ACF)")
ax_shift.set_xlabel("Year")
ax_shift.set_ylabel("Sunspots")

ax_acf.set_title("Autocorrelation (ACF)")
ax_acf.set_xlabel("Lag")
ax_acf.set_ylabel("ACF")
ax_acf.set_xlim(0, max_lag)
ax_acf.set_ylim(-1, 1)
ax_acf.axhline(0, color='black', linewidth=1)

ax_resid.set_title("Residuals for Partial Corr at Lag k")
ax_resid.set_xlabel("e(t-k)")
ax_resid.set_ylabel("e(t)")

ax_pacf.set_title("Partial Autocorrelation (PACF)")
ax_pacf.set_xlabel("Lag")
ax_pacf.set_ylabel("PACF")
ax_pacf.set_xlim(0, max_lag)
ax_pacf.set_ylim(-1, 1)
ax_pacf.axhline(0, color='black', linewidth=1)

fig.tight_layout()

# Plot the original data once (for reference)
ax_shift.plot(years, ts, 'k-o', label='Original')

# -----------------------------
# 7. Animation update function
# -----------------------------
def update(frame):
    """
    frame goes from 1..max_lag
    We'll:
      - Show how we SHIFT for ACF at lag=frame in top-left
      - Update ACF bars up to 'frame'
      - Show residual scatter e(t) vs e(t-k) for partial correlation at 'frame'
      - Update PACF bars up to 'frame'
    """
    lag = frame

    # --- (1) SHIFT for ACF in [0,0]
    ax_shift.cla()
    ax_shift.set_title(f"Time Series & Shifted (lag={lag})")
    ax_shift.set_xlabel("Year")
    ax_shift.set_ylabel("Sunspots")
    ax_shift.plot(years, ts, 'k-o', label='Original')

    shifted_data = shifted_series(ts, lag)
    ax_shift.plot(years, shifted_data, 'r-o', label=f"Shifted by {lag}")
    ax_shift.legend(loc='best')

    # show correlation at this lag
    valid_mask = ~np.isnan(shifted_data)
    if valid_mask.sum() > 0:
        corr = np.corrcoef(ts[lag:], shifted_data[lag:])[0, 1]
        ax_shift.text(0.02, 0.90,
                      f"Corr (lag={lag}) = {corr:.3f}",
                      transform=ax_shift.transAxes, fontsize=9, color='red')

    # --- (2) ACF bars in [0,1]
    ax_acf.cla()
    ax_acf.set_title("Autocorrelation (ACF)")
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("ACF")
    ax_acf.set_xlim(0, max_lag)
    ax_acf.set_ylim(-1, 1)
    ax_acf.axhline(0, color='black', linewidth=1)

    # Plot ACF from 1..lag
    acf_lags = np.arange(1, lag + 1)
    acf_heights = [acf_vals[k] for k in acf_lags]
    ax_acf.bar(acf_lags, acf_heights, color='blue', alpha=0.7)

    # --- (3) Residual scatter in [1,0] for partial correlation
    ax_resid.cla()
    ax_resid.set_title(f"Residuals for Partial Corr (lag={lag})")
    ax_resid.set_xlabel("e(t-k)")
    ax_resid.set_ylabel("e(t)")

    e_t, e_tk = residual_pairs[lag]
    if e_t is not None and e_tk is not None:
        ax_resid.scatter(e_tk, e_t, color='green', alpha=0.7)
        # correlation
        corr_resid = np.corrcoef(e_t, e_tk)[0, 1]
        ax_resid.text(0.05, 0.9,
                      f"partial_corr={corr_resid:.3f}",
                      transform=ax_resid.transAxes,
                      fontsize=9, color='green')

    # --- (4) PACF bars in [1,1]
    ax_pacf.cla()
    ax_pacf.set_title("Partial Autocorrelation (PACF)")
    ax_pacf.set_xlabel("Lag")
    ax_pacf.set_ylabel("PACF")
    ax_pacf.set_xlim(0, max_lag)
    ax_pacf.set_ylim(-1, 1)
    ax_pacf.axhline(0, color='black', linewidth=1)

    pacf_lags = np.arange(1, lag + 1)
    pacf_heights = [pacf_vals[k] for k in pacf_lags]
    ax_pacf.bar(pacf_lags, pacf_heights, color='orange', alpha=0.7)

    return []

# -----------------------------
# 8. Animate
# -----------------------------
anim = FuncAnimation(fig, update, frames=range(1, max_lag + 1),
                     interval=1200, blit=False)
plt.show()
