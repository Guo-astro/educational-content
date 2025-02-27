import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
#  Astrophysical constants in chosen units
# ------------------------------------------------------------
G = 4.3009e-3  # [pc (km/s)^2 / Msun] - gravitational constant in astro units

# ------------------------------------------------------------
#  Simulation parameters
# ------------------------------------------------------------
M_bh = 1.0e5  # [Msun]  Black hole mass
cloud_radius = 3.0  # [pc]    Initial radius of the cloud
num_particles = 800  # Number of test particles
# Initial cloud center-of-mass state
initial_distance = 10.0  # [pc]   Center-of-mass x-position (BH at origin)
impact_parameter = 5.0  # [pc]   Center-of-mass y-offset
initial_speed = 10.0  # [km/s] Center-of-mass velocity along +x direction

# Time stepping
dt = 0.01  # [Myr]  Timestep (0.01 Myr ~ 1e7 years)
num_steps = 500  # Number of frames in the animation


# ------------------------------------------------------------
#  Initialize the cloud's particle distribution
#    We place them uniformly in a sphere of radius cloud_radius,
#    all sharing the same bulk (center-of-mass) velocity initially.
# ------------------------------------------------------------
def initialize_cloud_3d(npart, radius, center, v_center):
    """
    Generate particle positions in a uniform sphere of given radius.
    center = (cx, cy, cz) in pc
    v_center = (vx, vy, vz) in km/s
    Returns positions, velocities as NumPy arrays.
    """
    cx, cy, cz = center
    vx, vy, vz = v_center

    positions = []

    # Rejection sampling for uniform sphere
    while len(positions) < npart:
        rx, ry, rz = np.random.uniform(-1, 1, 3)
        if rx * rx + ry * ry + rz * rz <= 1.0:
            # inside unit sphere => scale by 'radius'
            px = cx + radius * rx
            py = cy + radius * ry
            pz = cz + radius * rz
            positions.append((px, py, pz))

    positions = np.array(positions)

    # All particles share the same initial velocity
    velocities = np.zeros_like(positions)
    velocities[:, 0] = vx
    velocities[:, 1] = vy
    velocities[:, 2] = vz

    return positions, velocities


# ------------------------------------------------------------
#  Acceleration from the black hole at the origin
# ------------------------------------------------------------
def compute_acceleration_3d(positions, M_bh):
    """
    positions: (N, 3) array in pc
    M_bh: black hole mass in Msun
    Returns acceleration (N, 3) in (km/s) / Myr
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Distances in pc
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)

    # To avoid singularities at the origin, add a small softening if needed
    softening = 1e-4
    r_safe = np.maximum(r, softening)

    # Gravitational acceleration magnitude: G M_bh / r^2
    # Units: G in [pc (km/s)^2 / Msun], M_bh in Msun, r in pc
    # => a_mag in (km/s)^2 / pc
    a_mag = G * M_bh / (r_safe ** 2)

    # Convert to vector components; direction is -r_hat
    ax = -a_mag * x / r_safe  # in (km/s)^2/pc * dimensionless
    ay = -a_mag * y / r_safe
    az = -a_mag * z / r_safe

    # a has units of (km/s)^2 / pc, but we interpret dt in Myr,
    # so dv = a * dt is in km/s. That is consistent with 1 km/s ~ 1 pc/Myr.

    return np.column_stack((ax, ay, az))


# ------------------------------------------------------------
#  Main script
# ------------------------------------------------------------
# Initialize cloud in 3D
pos, vel = initialize_cloud_3d(
    num_particles,
    cloud_radius,
    center=(-initial_distance, impact_parameter, 0.0),
    v_center=(initial_speed, 0.0, 0.0)
)

# Set up 3D figure
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)
ax.set_zlim(-15, 15)
ax.set_xlabel('x [pc]')
ax.set_ylabel('y [pc]')
ax.set_zlabel('z [pc]')
ax.set_title('3D Tidal Stretching - BH at Origin')

# Plot initial scatter
scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=5, c='blue', alpha=0.6)
# Mark the black hole
ax.scatter([0], [0], [0], s=50, c='black', marker='x')


def update(frame):
    global pos, vel
    # Compute BH gravitational acceleration
    acc = compute_acceleration_3d(pos, M_bh)  # (N,3)
    # Update velocity (Euler method)
    vel += acc * dt  # dt in Myr => dv in km/s
    # Update position
    pos += vel * dt  # dt in Myr => dx in pc

    # Update scatter plot data
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])

    return scatter,


anim = FuncAnimation(fig, update, frames=num_steps, interval=40, blit=False)

# If you want to save the animation as an MP4:
# anim.save('tidal_3d_pc_msun_kms.mp4', writer='ffmpeg', fps=30)

plt.show()
