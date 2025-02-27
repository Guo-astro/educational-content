import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------
# 1) Global constants in astrophysical units
# ----------------------------------------------------
G = 4.3009e-3  # [pc (km/s)^2 / Msun] - gravitational constant

# ----------------------------------------------------
# 2) Simulation parameters
# ----------------------------------------------------
M_bh = 1.0e5  # [Msun]  Black hole mass
cloud_radius = 1.0  # [pc]    Initial radius of the cloud
num_particles = 600  # Number of test particles

# Initial center-of-mass orbital parameters
initial_distance = 10.0  # [pc] (cloud center's x-position, BH at origin)
impact_parameter = 3.0  # [pc] (cloud center's y-offset)
initial_speed = 8.0  # [km/s] along +x direction

# Time integration
dt = 0.01  # [Myr] time step
num_frames = 400  # number of animation frames

# Shock (crowding) detection parameters (3D neighbor check)
neighbor_radius = 0.4  # [pc] radius to check for local crowding
shock_threshold = 30  # if neighbor count > shock_threshold => shock zone


# ----------------------------------------------------
# 3) Initialize the 3D cloud
# ----------------------------------------------------
def initialize_cloud_3d(npart, radius, center, v_center):
    """
    Place npart particles uniformly in a sphere of 'radius' (pc).
    center = (cx, cy, cz) in pc
    v_center = (vx, vy, vz) in km/s
    Returns:
      positions: (npart, 3) array in pc
      velocities: (npart, 3) array in km/s
    """
    cx, cy, cz = center
    vx, vy, vz = v_center

    positions = []
    # Rejection sampling for uniform sphere
    while len(positions) < npart:
        rx, ry, rz = np.random.uniform(-1, 1, 3)
        if rx * rx + ry * ry + rz * rz <= 1.0:
            px = cx + radius * rx
            py = cy + radius * ry
            pz = cz + radius * rz
            positions.append((px, py, pz))

    positions = np.array(positions)
    velocities = np.zeros_like(positions)
    velocities[:, 0] = vx
    velocities[:, 1] = vy
    velocities[:, 2] = vz

    return positions, velocities


def compute_bh_acceleration_3d(positions, M_bh):
    """
    positions: shape (N,3) in pc
    M_bh: black hole mass in Msun
    Returns acceleration (N,3) in (km/s)/Myr
    using G in [pc (km/s)^2 / Msun].
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)

    # Avoid singularity at r=0 with small softening
    softening = 1e-4
    r_safe = np.maximum(r, softening)

    # a_mag = G * M_bh / r^2  => units: (km/s)^2/pc
    a_mag = G * M_bh / (r_safe ** 2)
    # Vector components: direction is -r_hat
    ax = -a_mag * x / r_safe
    ay = -a_mag * y / r_safe
    az = -a_mag * z / r_safe

    return np.column_stack((ax, ay, az))


# ----------------------------------------------------
# 4) Set up simulation state
# ----------------------------------------------------
pos, vel = initialize_cloud_3d(
    num_particles,
    cloud_radius,
    center=(-initial_distance, impact_parameter, 0.0),
    v_center=(initial_speed, 0.0, 0.0)
)

# ----------------------------------------------------
# 5) Matplotlib figure with two subplots:
#    - top-down (x-y)
#    - side view (x-z)
# ----------------------------------------------------
fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5))

ax_top.set_xlim(-15, 15)
ax_top.set_ylim(-10, 10)
ax_top.set_xlabel('x [pc]')
ax_top.set_ylabel('y [pc]')
ax_top.set_title('Top-Down (Orbital Plane) View')

ax_side.set_xlim(-15, 15)
ax_side.set_ylim(-5, 5)
ax_side.set_xlabel('x [pc]')
ax_side.set_ylabel('z [pc]')
ax_side.set_title('Side (Vertical) View')

# Scatter for normal (blue) and shock (red) points in each view
scatter_top_normal, = ax_top.plot([], [], 'o', color='blue', markersize=2, alpha=0.6, label='Normal')
scatter_top_shock, = ax_top.plot([], [], 'o', color='red', markersize=4, alpha=0.8, label='Shock')

scatter_side_normal, = ax_side.plot([], [], 'o', color='blue', markersize=2, alpha=0.6)
scatter_side_shock, = ax_side.plot([], [], 'o', color='red', markersize=4, alpha=0.8)

ax_top.legend(loc='upper right')

# Text objects for leading/trailing labels in each plot
# We'll create them now and update them each frame:
text_leading_top = ax_top.text(0, 0, "", color='black', fontsize=8)
text_trailing_top = ax_top.text(0, 0, "", color='black', fontsize=8)
text_leading_side = ax_side.text(0, 0, "", color='black', fontsize=8)
text_trailing_side = ax_side.text(0, 0, "", color='black', fontsize=8)


# ----------------------------------------------------
# 6) Update function for animation
# ----------------------------------------------------
def update(frame):
    global pos, vel

    # -- (A) Integrate one step
    acc = compute_bh_acceleration_3d(pos, M_bh)
    vel += acc * dt  # (km/s) + ( (km/s)/Myr * Myr ) => km/s
    pos += vel * dt  # (pc) + ( km/s * Myr ) => pc

    # -- (B) Simple 3D neighbor-based "shock" detection
    #         If a particle has >shock_threshold neighbors within neighbor_radius
    #         in 3D, label it as shock region.
    X = pos[:, 0]
    Y = pos[:, 1]
    Z = pos[:, 2]

    # We'll store the indices of shock vs. normal
    shock_indices = []
    normal_indices = []

    for i in range(len(pos)):
        dx = X - X[i]
        dy = Y - Y[i]
        dz = Z - Z[i]
        dist2 = dx * dx + dy * dy + dz * dz
        # Count neighbors inside neighbor_radius
        neighbors = np.count_nonzero(dist2 < neighbor_radius * neighbor_radius) - 1
        if neighbors >= shock_threshold:
            shock_indices.append(i)
        else:
            normal_indices.append(i)

    # Convert to arrays for easier indexing
    shock_indices = np.array(shock_indices, dtype=int)
    normal_indices = np.array(normal_indices, dtype=int)

    # -- (C) Update top-down (x-y) scatter
    scatter_top_normal.set_data(X[normal_indices], Y[normal_indices])
    scatter_top_shock.set_data(X[shock_indices], Y[shock_indices])

    # -- (D) Update side (x-z) scatter
    scatter_side_normal.set_data(X[normal_indices], Z[normal_indices])
    scatter_side_shock.set_data(X[shock_indices], Z[shock_indices])

    # -- (E) Label leading & trailing edges
    # Leading edge = particle with max x
    # Trailing edge = particle with min x
    lead_idx = np.argmax(X)
    trail_idx = np.argmin(X)

    # Top-down
    text_leading_top.set_position((X[lead_idx], Y[lead_idx]))
    text_leading_top.set_text("Leading Edge")
    text_trailing_top.set_position((X[trail_idx], Y[trail_idx]))
    text_trailing_top.set_text("Trailing Edge")

    # Side view
    text_leading_side.set_position((X[lead_idx], Z[lead_idx]))
    text_leading_side.set_text("Leading Edge")
    text_trailing_side.set_position((X[trail_idx], Z[trail_idx]))
    text_trailing_side.set_text("Trailing Edge")

    return (scatter_top_normal, scatter_top_shock,
            scatter_side_normal, scatter_side_shock,
            text_leading_top, text_trailing_top,
            text_leading_side, text_trailing_side)


anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# If desired, save to MP4 (requires ffmpeg):
# anim.save('tidal_3d_shock_views.mp4', writer='ffmpeg', fps=30)

plt.tight_layout()
plt.show()
