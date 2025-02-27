import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots


###############################################################################
#                           1) TARGET DISTRIBUTIONS
###############################################################################

# --- A. Normal Target (in d dimensions) ---
def normal_pdf(x):
    """
    Multidimensional standard normal PDF.
    x: 1D, 2D, or 3D numpy array.
    Returns the product of 1D standard normals = exp(-0.5*||x||^2) / ( (2*pi)^(d/2) ).
    """
    # x can be shape (d,). We'll handle 1D, 2D, or 3D.
    d = np.size(x)
    r2 = np.sum(x ** 2)
    return np.exp(-0.5 * r2) / ((2 * np.pi) ** (0.5 * d))


# --- B. Poisson Target (in d dimensions) ---
def poisson_pmf(x, lam=4.0):
    """
    Multidimensional Poisson with parameter lam, factorized:
      pi(x) = Prod_{i=1..d} [ lam^(x[i]) * exp(-lam) / x[i]! ].
    x: array-like of shape (d,) with nonnegative integer entries.
    """
    # If d=1, x is a scalar or shape (1,). If d=2 or d=3, x is a 2D or 3D vector of nonnegative ints.
    # We'll compute the product of 1D Poisson pmfs.
    if np.amin(x) < 0:
        # If any coordinate is negative, pmf=0
        return 0.0

    # Convert x to an integer array
    x_int = np.array(x, dtype=int)
    d = x_int.size

    # log pmf for each coordinate: x[i]*log(lam) - lam - log(x[i]!)
    # Then sum across coords, exponentiate.
    # But for small x[i], we can do factorial directly or use math.lgamma.
    from math import lgamma
    log_pmf_sum = 0.0
    for i in range(d):
        xi = x_int[i]
        # log( lam^xi ) = xi*log(lam)
        # log( exp(-lam) ) = -lam
        # log( 1/xi! ) = -log(xi!)
        # log(xi!) = lgamma(xi+1)
        log_pmf_sum += xi * np.log(lam) - lam - lgamma(xi + 1)

    return np.exp(log_pmf_sum)


###############################################################################
#                     2) PROPOSAL DISTRIBUTIONS & MH UPDATES
###############################################################################

# === A. Normal MCMC with drifted Gaussian proposal (continuous) ===
def normal_proposal(x, drift=0.3, step=0.4):
    """
    Non-symmetric proposal for a d-dimensional Normal chain:
      x' = x + drift + Normal(0, step^2 I).
    x: shape (d,). Return x': shape(d,).
    """
    d = x.size
    return x + drift + np.random.normal(0, step, size=d)


def normal_proposal_pdf(x_new, x_old, drift=0.3, step=0.4):
    """
    q(x_new|x_old) for the above drifted Gaussian.
    """
    # Probability density of x_new under Normal(mean=x_old+drift, cov=step^2 I).
    # This is a standard multivariate normal formula with dimension d.
    d = x_old.size
    diff = x_new - (x_old + drift)
    r2 = np.sum(diff ** 2)
    coeff = 1.0 / ((2 * np.pi * step ** 2) ** (d / 2))
    return coeff * np.exp(-0.5 * r2 / (step ** 2))


# === B. Poisson MCMC with biased random-walk proposal (discrete) ===
def poisson_proposal(x, p=0.7):
    """
    Non-symmetric discrete proposal on each coordinate:
      - With probability p, we do x[i] -> x[i] + 1
      - With probability (1-p), we do x[i] -> max(x[i]-1, 0)  (can't go negative)
    x: shape(d,). Return x': shape(d,).
    """
    d = x.size
    x_new = np.array(x, dtype=int).copy()
    for i in range(d):
        # Propose step +1 with prob p, or -1 with prob (1-p), unless x[i]=0
        if x_new[i] == 0:
            # if x[i]=0, then with prob p -> 1, with prob (1-p)-> 0
            if np.random.rand() < p:
                x_new[i] = x_new[i] + 1
            else:
                # remain 0
                pass
        else:
            # x[i] > 0
            if np.random.rand() < p:
                x_new[i] = x_new[i] + 1
            else:
                x_new[i] = x_new[i] - 1
    return x_new


def poisson_proposal_prob(x_new, x_old, p=0.7):
    """
    q(x_new|x_old): Probability of jumping from x_old to x_new
    under the discrete random-walk described above.
    We'll factorize across each coordinate i.
    """
    d = x_old.size
    x_old = np.array(x_old, dtype=int)
    x_new = np.array(x_new, dtype=int)

    prob = 1.0
    for i in range(d):
        old_i = x_old[i]
        new_i = x_new[i]

        # Cases:
        if old_i == 0:
            # with prob p -> old_i+1, with prob (1-p)-> remain 0
            if new_i == 1:
                prob *= p
            elif new_i == 0:
                prob *= (1 - p)
            else:
                # impossible jump
                return 0.0
        else:
            # old_i > 0
            if new_i == old_i + 1:
                prob *= p
            elif new_i == old_i - 1:
                prob *= (1 - p)
            elif new_i == old_i:
                # Not a valid jump in this scheme (we always +/-1 if old_i>0)
                return 0.0
            else:
                # More than +/-1 => impossible
                return 0.0

    return prob


###############################################################################
#                    3) RUNNING THE CHAINS (Normal vs Poisson)
###############################################################################

def run_mh_normal(d=1, correct=True, N=300, drift=0.3, step=0.4):
    """
    Metropolis-Hastings for a d-dimensional standard Normal target,
    using a drifted Gaussian proposal (non-symmetric).
    If correct=True, we include the proposal ratio -> satisfies Detailed Balance.
    If correct=False, we omit the proposal ratio -> breaks Detailed Balance.
    Returns: chain of shape (N,d) [or (N,) if d=1].
    """
    # Initialize
    x = np.zeros(d)  # start at the origin
    chain = np.zeros((N, d))
    chain[0] = x

    for i in range(1, N):
        x_prop = normal_proposal(x, drift=drift, step=step)

        # forward/backward proposal densities
        q_forward = normal_proposal_pdf(x_prop, x, drift=drift, step=step)
        q_backward = normal_proposal_pdf(x, x_prop, drift=drift, step=step)

        # ratio of proposal densities
        prop_ratio = q_backward / q_forward if q_forward > 0 else 0

        # ratio of target densities
        pi_ratio = normal_pdf(x_prop) / normal_pdf(x)

        # acceptance
        if correct:
            alpha = min(1, pi_ratio * prop_ratio)
        else:
            alpha = min(1, pi_ratio)

        if np.random.rand() < alpha:
            x = x_prop
        chain[i] = x

    # if d=1, flatten chain to shape (N,)
    if d == 1:
        return chain.reshape(N)
    return chain


def run_mh_poisson(d=1, correct=True, N=300, p=0.7, lam=4.0):
    """
    Metropolis-Hastings for a d-dimensional Poisson target (parameter lam),
    using a biased random-walk proposal with prob p for +1, (1-p) for -1.
    If correct=True, we include the proposal ratio -> Detailed Balance.
    If correct=False, omit it -> Not DB.
    Returns: chain of shape (N,d) [or (N,) if d=1].
    """
    x = np.zeros(d, dtype=int)  # start at 0
    chain = np.zeros((N, d), dtype=int)
    chain[0] = x

    for i in range(1, N):
        x_prop = poisson_proposal(x, p=p)

        q_forward = poisson_proposal_prob(x_prop, x, p=p)
        q_backward = poisson_proposal_prob(x, x_prop, p=p)
        prop_ratio = (q_backward / q_forward) if q_forward > 0 else 0

        pi_ratio = poisson_pmf(x_prop, lam=lam) / poisson_pmf(x, lam=lam)

        if correct:
            alpha = min(1, pi_ratio * prop_ratio)
        else:
            alpha = min(1, pi_ratio)

        if np.random.rand() < alpha:
            x = x_prop
        chain[i] = x

    if d == 1:
        return chain.reshape(N)
    return chain


###############################################################################
#               4) ANIMATION: 3 ROWS (1D, 2D, 3D) x 4 COLS
###############################################################################

def animate_all():
    # Number of MCMC steps
    N = 300

    # -----------------------------------------------------------------------
    # Generate the 12 chains:
    #   Normal: (1D,2D,3D) x (DB, NotDB)
    #   Poisson: (1D,2D,3D) x (DB, NotDB)
    # -----------------------------------------------------------------------

    # 1D Normal: DB vs NotDB
    normal_1d_db = run_mh_normal(d=1, correct=True, N=N, drift=0.3, step=0.4)
    normal_1d_notdb = run_mh_normal(d=1, correct=False, N=N, drift=0.3, step=0.4)

    # 1D Poisson: DB vs NotDB
    poisson_1d_db = run_mh_poisson(d=1, correct=True, N=N, p=0.7, lam=4.0)
    poisson_1d_notdb = run_mh_poisson(d=1, correct=False, N=N, p=0.7, lam=4.0)

    # 2D Normal: DB vs NotDB
    normal_2d_db = run_mh_normal(d=2, correct=True, N=N, drift=0.3, step=0.4)
    normal_2d_notdb = run_mh_normal(d=2, correct=False, N=N, drift=0.3, step=0.4)

    # 2D Poisson: DB vs NotDB
    poisson_2d_db = run_mh_poisson(d=2, correct=True, N=N, p=0.7, lam=4.0)
    poisson_2d_notdb = run_mh_poisson(d=2, correct=False, N=N, p=0.7, lam=4.0)

    # 3D Normal: DB vs NotDB
    normal_3d_db = run_mh_normal(d=3, correct=True, N=N, drift=0.3, step=0.4)
    normal_3d_notdb = run_mh_normal(d=3, correct=False, N=N, drift=0.3, step=0.4)

    # 3D Poisson: DB vs NotDB
    poisson_3d_db = run_mh_poisson(d=3, correct=True, N=N, p=0.7, lam=4.0)
    poisson_3d_notdb = run_mh_poisson(d=3, correct=False, N=N, p=0.7, lam=4.0)

    # -----------------------------------------------------------------------
    # Set up the figure: 3 rows x 4 columns
    #   Row 1 = 1D, Row 2 = 2D, Row 3 = 3D
    #   Col 1 = DB Normal, Col 2 = Not DB Normal
    #   Col 3 = DB Poisson, Col 4 = Not DB Poisson
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))

    # Row 1 (1D)
    ax_1d_norm_db = fig.add_subplot(3, 4, 1)
    ax_1d_norm_notdb = fig.add_subplot(3, 4, 2)
    ax_1d_pois_db = fig.add_subplot(3, 4, 3)
    ax_1d_pois_notdb = fig.add_subplot(3, 4, 4)

    # Row 2 (2D)
    ax_2d_norm_db = fig.add_subplot(3, 4, 5)
    ax_2d_norm_notdb = fig.add_subplot(3, 4, 6)
    ax_2d_pois_db = fig.add_subplot(3, 4, 7)
    ax_2d_pois_notdb = fig.add_subplot(3, 4, 8)

    # Row 3 (3D)
    ax_3d_norm_db = fig.add_subplot(3, 4, 9, projection='3d')
    ax_3d_norm_notdb = fig.add_subplot(3, 4, 10, projection='3d')
    ax_3d_pois_db = fig.add_subplot(3, 4, 11, projection='3d')
    ax_3d_pois_notdb = fig.add_subplot(3, 4, 12, projection='3d')

    # Titles for columns
    ax_1d_norm_db.set_title("1D Normal - DB")
    ax_1d_norm_notdb.set_title("1D Normal - Not DB")
    ax_1d_pois_db.set_title("1D Poisson - DB")
    ax_1d_pois_notdb.set_title("1D Poisson - Not DB")

    ax_2d_norm_db.set_title("2D Normal - DB")
    ax_2d_norm_notdb.set_title("2D Normal - Not DB")
    ax_2d_pois_db.set_title("2D Poisson - DB")
    ax_2d_pois_notdb.set_title("2D Poisson - Not DB")

    ax_3d_norm_db.set_title("3D Normal - DB")
    ax_3d_norm_notdb.set_title("3D Normal - Not DB")
    ax_3d_pois_db.set_title("3D Poisson - DB")
    ax_3d_pois_notdb.set_title("3D Poisson - Not DB")

    # Common function to clear and set axis limits
    def setup_1d_axes(ax):
        ax.set_xlim(-8, 15)  # extended range for normal vs. possibly drifting
        ax.set_ylim(0, 80)
        ax.set_xlabel("x")
        ax.set_ylabel("Frequency")

    def setup_2d_axes(ax):
        ax.set_xlim(-8, 15)
        ax.set_ylim(-8, 15)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")

    def setup_3d_axes(ax):
        ax.set_xlim(-8, 15)
        ax.set_ylim(-8, 15)
        ax.set_zlim(-8, 15)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Initialize each row's axis
    for ax in [ax_1d_norm_db, ax_1d_norm_notdb, ax_1d_pois_db, ax_1d_pois_notdb]:
        setup_1d_axes(ax)
    for ax in [ax_2d_norm_db, ax_2d_norm_notdb, ax_2d_pois_db, ax_2d_pois_notdb]:
        setup_2d_axes(ax)
    for ax in [ax_3d_norm_db, ax_3d_norm_notdb, ax_3d_pois_db, ax_3d_pois_notdb]:
        setup_3d_axes(ax)

    # -----------------------------------------------------------------------
    # Update function for FuncAnimation
    # -----------------------------------------------------------------------
    def update(frame):
        # 1) Clear each axis
        ax_1d_norm_db.cla();
        setup_1d_axes(ax_1d_norm_db)
        ax_1d_norm_notdb.cla();
        setup_1d_axes(ax_1d_norm_notdb)
        ax_1d_pois_db.cla();
        setup_1d_axes(ax_1d_pois_db)
        ax_1d_pois_notdb.cla();
        setup_1d_axes(ax_1d_pois_notdb)

        ax_2d_norm_db.cla();
        setup_2d_axes(ax_2d_norm_db)
        ax_2d_norm_notdb.cla();
        setup_2d_axes(ax_2d_norm_notdb)
        ax_2d_pois_db.cla();
        setup_2d_axes(ax_2d_pois_db)
        ax_2d_pois_notdb.cla();
        setup_2d_axes(ax_2d_pois_notdb)

        ax_3d_norm_db.cla();
        setup_3d_axes(ax_3d_norm_db)
        ax_3d_norm_notdb.cla();
        setup_3d_axes(ax_3d_norm_notdb)
        ax_3d_pois_db.cla();
        setup_3d_axes(ax_3d_pois_db)
        ax_3d_pois_notdb.cla();
        setup_3d_axes(ax_3d_pois_notdb)

        # 2) Plot partial chain up to 'frame'

        # --- Row 1: 1D Histograms ---
        ax_1d_norm_db.hist(normal_1d_db[:frame], bins=30, range=(-8, 15), color='blue', alpha=0.7)
        ax_1d_norm_notdb.hist(normal_1d_notdb[:frame], bins=30, range=(-8, 15), color='red', alpha=0.7)
        ax_1d_pois_db.hist(poisson_1d_db[:frame], bins=range(0, 16), color='green', alpha=0.7, align='left')
        ax_1d_pois_notdb.hist(poisson_1d_notdb[:frame], bins=range(0, 16), color='orange', alpha=0.7, align='left')

        ax_1d_norm_db.set_title("1D Normal - DB")
        ax_1d_norm_notdb.set_title("1D Normal - Not DB")
        ax_1d_pois_db.set_title("1D Poisson - DB")
        ax_1d_pois_notdb.set_title("1D Poisson - Not DB")

        ax_1d_norm_db.text(-7.5, 70, f"Step: {frame}", fontsize=9,
                           bbox=dict(facecolor='white', alpha=0.7))

        # --- Row 2: 2D Scatter ---
        ax_2d_norm_db.scatter(normal_2d_db[:frame, 0], normal_2d_db[:frame, 1],
                              s=10, c='blue', alpha=0.5)
        ax_2d_norm_notdb.scatter(normal_2d_notdb[:frame, 0], normal_2d_notdb[:frame, 1],
                                 s=10, c='red', alpha=0.5)
        ax_2d_pois_db.scatter(poisson_2d_db[:frame, 0], poisson_2d_db[:frame, 1],
                              s=10, c='green', alpha=0.5)
        ax_2d_pois_notdb.scatter(poisson_2d_notdb[:frame, 0], poisson_2d_notdb[:frame, 1],
                                 s=10, c='orange', alpha=0.5)

        ax_2d_norm_db.set_title("2D Normal - DB")
        ax_2d_norm_notdb.set_title("2D Normal - Not DB")
        ax_2d_pois_db.set_title("2D Poisson - DB")
        ax_2d_pois_notdb.set_title("2D Poisson - Not DB")

        ax_2d_norm_db.text(-7.5, 13, f"Step: {frame}", fontsize=9,
                           bbox=dict(facecolor='white', alpha=0.7))

        # --- Row 3: 3D Scatter ---
        ax_3d_norm_db.scatter(normal_3d_db[:frame, 0], normal_3d_db[:frame, 1], normal_3d_db[:frame, 2],
                              s=10, c='blue', alpha=0.5)
        ax_3d_norm_notdb.scatter(normal_3d_notdb[:frame, 0], normal_3d_notdb[:frame, 1], normal_3d_notdb[:frame, 2],
                                 s=10, c='red', alpha=0.5)
        ax_3d_pois_db.scatter(poisson_3d_db[:frame, 0], poisson_3d_db[:frame, 1], poisson_3d_db[:frame, 2],
                              s=10, c='green', alpha=0.5)
        ax_3d_pois_notdb.scatter(poisson_3d_notdb[:frame, 0], poisson_3d_notdb[:frame, 1], poisson_3d_notdb[:frame, 2],
                                 s=10, c='orange', alpha=0.5)

        ax_3d_norm_db.set_title("3D Normal - DB")
        ax_3d_norm_notdb.set_title("3D Normal - Not DB")
        ax_3d_pois_db.set_title("3D Poisson - DB")
        ax_3d_pois_notdb.set_title("3D Poisson - Not DB")

        ax_3d_norm_db.text2D(0.05, 0.95, f"Step: {frame}",
                             transform=ax_3d_norm_db.transAxes, fontsize=9,
                             bbox=dict(facecolor='white', alpha=0.7))

        return []

    ani = FuncAnimation(fig, update, frames=N, interval=200, blit=False)
    plt.tight_layout()
    plt.show()


# --- Run animation if main ---
if __name__ == "__main__":
    animate_all()
