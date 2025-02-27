import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def chebyshev_bound(k):
    """Compute the Chebyshev upper bound 1/k²."""
    return 1.0 / (k ** 2)


def plot_normal_distribution_with_chebyshev(k, mu=0, sigma=1):
    """
    Plot the standard normal PDF and highlight the tail region where |X-μ| ≥ kσ.

    Mathematical annotation:
      Normal PDF: f(x)=1/(√(2π)σ) exp[-(x-μ)²/(2σ²)]
      Tail probability: P(|X-μ| ≥ kσ)
      Chebyshev bound: 1/k²
    """
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    pdf = stats.norm.pdf(x, mu, sigma)

    # Tail probability: P(|X-μ| ≥ kσ)
    tail_prob = 2 * (1 - stats.norm.cdf(k, mu, sigma))
    bound = chebyshev_bound(k)

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, label='Normal PDF', color='blue')
    plt.fill_between(x, 0, pdf, where=(np.abs(x - mu) >= k * sigma), color='gray', alpha=0.3,
                     label=r'Tail region: $|X-\mu| >= k\sigma$')
    plt.axvline(mu + k * sigma, color='red', linestyle='--', label=r'$\mu+k\sigma$')
    plt.axvline(mu - k * sigma, color='red', linestyle='--', label=r'$\mu-k\sigma$')
    plt.title(f'Normal Distribution with Chebyshev Interval (k={k})')
    plt.xlabel('x')
    plt.ylabel('Probability Density')

    formula_text = (
            r"Normal PDF: $f(x)=\frac{1}{\sqrt{2\pi}\,\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$"
            "\n" +
            rf"$P(|X-\mu|> k\sigma)={tail_prob:.3f}$, $1/k^2={bound:.3f}$"
    )
    plt.text(mu, max(pdf) * 0.8, formula_text, horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.show()


def plot_poisson_distribution_with_chebyshev(k, lam=10):
    """
    Plot the Poisson PMF and highlight the region where |X-μ| ≥ kσ.

    Mathematical annotation:
      Poisson PMF: P(X=x)= (λ^x e^(−λ))/(x!)
      μ = λ, σ = √λ
      Tail probability: P(|X-μ| ≥ kσ)
      Chebyshev bound: 1/k²
    """
    mu, sigma = lam, np.sqrt(lam)
    x = np.arange(0, lam + 4 * sigma + 1)
    pmf = stats.poisson.pmf(x, lam)

    # Tail probability: P(|X-μ| ≥ kσ)
    tail_mask = np.abs(x - mu) >= k * sigma
    tail_prob = np.sum(pmf[tail_mask])
    bound = chebyshev_bound(k)

    plt.figure(figsize=(10, 6))
    plt.bar(x, pmf, label='Poisson PMF', alpha=0.7, color='green')
    plt.axvline(mu + k * sigma, color='red', linestyle='--', label=r'$\mu+k\sigma$')
    plt.axvline(mu - k * sigma, color='red', linestyle='--', label=r'$\mu-k\sigma$')
    plt.title(f'Poisson Distribution (λ={lam}) with Chebyshev Interval (k={k})')
    plt.xlabel('x')
    plt.ylabel('Probability')

    formula_text = (
            r"Poisson PMF: $P(X=x)=\frac{\lambda^x e^{-\lambda}}{x!}$" "\n" +
            rf"$\mu=\lambda={lam}$, $\sigma=\sqrt{{\lambda}}={sigma:.2f}$" "\n" +
            rf"$P(|X-\mu|> k\sigma)={tail_prob:.3f}$, $1/k^2={bound:.3f}$"
    )
    plt.text(mu, max(pmf) * 0.7, formula_text, horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.show()


def plot_normal_chebyshev_vs_k():
    """
    Plot the actual tail probability for the standard normal distribution
    versus the Chebyshev bound (1/k²) as k varies.

    Annotation:
      Chebyshev inequality: P(|X-μ| ≥ kσ) ≤ 1/k²
    """
    k_values = np.linspace(1, 4, 100)
    actual_tail = np.array([2 * (1 - stats.norm.cdf(k)) for k in k_values])
    chebyshev_bounds = np.array([chebyshev_bound(k) for k in k_values])

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, actual_tail, label='Actual Tail Probability (Normal)', color='blue')
    plt.plot(k_values, chebyshev_bounds, label='Chebyshev Bound (1/k²)', color='red', linestyle='--')
    plt.xlabel('k (standard deviations)')
    plt.ylabel('Probability')
    plt.title('Tail Probability vs. Chebyshev Bound for the Normal Distribution')

    formula_text = r"Chebyshev: $P(|X-\mu|> k\sigma)\le \frac{1}{k^2}$"
    plt.text(1.5, max(actual_tail) * 0.8, formula_text, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.show()


def plot_cauchy_distribution(mu=0, scale=1):
    """
    Plot the Cauchy distribution.

    Mathematical annotation:
      Cauchy PDF: f(x)=1/(π·scale·[1+((x-μ)/scale)²])
    Note: Chebyshev inequality is not applicable as the Cauchy distribution
          does not have finite mean or variance.
    """
    x = np.linspace(mu - 10 * scale, mu + 10 * scale, 1000)
    pdf = stats.cauchy.pdf(x, loc=mu, scale=scale)

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, label='Cauchy PDF', color='purple')
    plt.title('Cauchy Distribution (No Finite Mean or Variance)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')

    formula_text = (
        r"Cauchy PDF: $f(x)=\frac{1}{\pi\,scale\,(1+((x-\mu)/scale)^2)}$"
        "\nChebyshev inequality not applicable."
    )
    plt.text(mu, max(pdf) * 0.8, formula_text, horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.show()


def plot_chebyshev_equality_distribution(k, mu=0, sigma=1):
    """
    Plot a discrete distribution that exactly attains equality in Chebyshev's inequality.

    The distribution is defined by:
      P(X = μ - kσ) = 1/(2k²)
      P(X = μ)       = 1 - 1/k²
      P(X = μ + kσ) = 1/(2k²)

    This yields:
      Var(X) = σ²  and  P(|X-μ| ≥ kσ) = 1/k².

    Mathematical annotation in the plot shows these formulas and equality.
    """
    x_points = np.array([mu - k * sigma, mu, mu + k * sigma])
    probabilities = np.array([1 / (2 * k ** 2), 1 - 1 / k ** 2, 1 / (2 * k ** 2)])

    tail_prob = probabilities[0] + probabilities[2]  # This equals 1/k².
    bound = chebyshev_bound(k)

    plt.figure(figsize=(10, 6))
    plt.stem(x_points, probabilities, basefmt=" ", use_line_collection=True,
             linefmt='blue', markerfmt='bo', label='Discrete PMF')
    plt.axvline(mu - k * sigma, color='red', linestyle='--', label=r'$\mu-k\sigma$')
    plt.axvline(mu + k * sigma, color='red', linestyle='--', label=r'$\mu+k\sigma$')

    plt.title(f'Discrete Distribution Achieving Chebyshev Equality (k={k})')
    plt.xlabel('x')
    plt.ylabel('Probability')

    formula_text = (
            f"Distribution:\n"
            rf"$P(X={mu - k * sigma:.2f})=\frac{{1}}{{2k^2}}={1 / (2 * k ** 2):.3f}$" "\n" +
            rf"$P(X={mu})=1-\frac{{1}}{{k^2}}={1 - 1 / k ** 2:.3f}$" "\n" +
            rf"$P(X={mu + k * sigma:.2f})=\frac{{1}}{{2k^2}}={1 / (2 * k ** 2):.3f}$" "\n" +
            rf"$P(|X-\mu|\ge k\sigma)={tail_prob:.3f} = 1/k^2={bound:.3f}$"
    )
    plt.text(mu, max(probabilities) * 1.1, formula_text, horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    k_value = 2

    # Normal distribution demonstration.
    plot_normal_distribution_with_chebyshev(k_value)

    # Poisson distribution demonstration.
    plot_poisson_distribution_with_chebyshev(k_value, lam=10)

    # Comparison of actual tail probability vs. Chebyshev bound for Normal distribution.
    plot_normal_chebyshev_vs_k()

    # Cauchy distribution demonstration.
    plot_cauchy_distribution()

    # Discrete distribution that exactly satisfies (attains equality in) Chebyshev's inequality.
    plot_chebyshev_equality_distribution(k_value)
