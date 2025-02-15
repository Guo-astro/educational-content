import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lib.utils.gen_gif import auto_save_gif

plt.style.use("dark_background")


#############################
# 1) Define the Strict Solution at x=0
#############################
def free_gaussian_packet(t, sigma0=1.0, k0=1.0, v=None, t_s=None):
    """
    Returns psi(0,t) for a 1D free-particle Gaussian wave packet
    traveling with group velocity v = k0 (in units hbar=1, m=1).

    sigma(t)^2 = sigma0^2 * (1 + i t/t_s).
    We pick t_s = 2*sigma0^2 by default if not provided.

    This function is a complex number representing the wavefunction at x=0.
    """
    if t_s is None:
        t_s = 2.0 * sigma0 ** 2  # typical choice
    if v is None:
        v = k0  # group velocity in these units

    # Complex time-dependent width
    sigma_t_sq = sigma0 ** 2 * (1.0 + 1j * t / t_s)

    # Amplitude prefactor
    prefactor = 1.0 / ((2.0 * np.pi * sigma_t_sq) ** 0.25)

    # Gaussian factor (x=0)
    gauss = np.exp(-(-v * t) ** 2 / (4.0 * sigma_t_sq))

    # Plane-wave phase factor (x=0)
    phase = np.exp(-1j * (0.5 * k0 ** 2) * t)

    return prefactor * gauss * phase


#############################
# 2) Main Plotting / Animation
#############################
def main():
    # Parameters for the wave packets
    sigma0 = 1.0
    k0 = 2.0  # wave number => group velocity v = 2
    # Time range
    t_min, t_max = -6.0, 6.0
    fps = 30
    duration = 15
    n_frames = fps * duration
    t_vals = np.linspace(t_min, t_max, n_frames)

    # Evaluate wavefunctions at x=0
    psi_plus = np.array([free_gaussian_packet(t, sigma0, k0, v=+k0) for t in t_vals])
    psi_minus = np.array([free_gaussian_packet(t, sigma0, k0, v=-k0) for t in t_vals])

    # Real/Imag parts
    x1_vals = psi_plus.real
    y1_vals = psi_plus.imag
    z1_vals = t_vals  # interpret time as 'vertical' axis

    x2_vals = psi_minus.real
    y2_vals = psi_minus.imag
    z2_vals = -t_vals  # so it "goes" in opposite direction visually

    # Offsets for the "3-plane" style
    proj_offset_re_time = -1.5
    proj_offset_im_time = -1.5
    proj_offset_re_im = 0

    # Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #############################
    # Text boxes
    #############################
    fig.text(
        0.01, 0.98,
        r"Governing Equation: $i\,\partial_t\psi(x,t)=-\frac{1}{2}\,\partial_x^2\psi(x,t)$",
        color="white", fontsize=12
    )
    fig.text(
        0.01, 0.94,
        r"Entangled State (conceptual): $|\Psi\rangle = \frac{1}{\sqrt{2}}(|+\rangle_1|-\rangle_2 + |-\rangle_1|+\rangle_2)$",
        color="white", fontsize=10
    )
    fig.text(
        0.01, 0.90,
        r"Strict Solution for Each Photon at $x=0$: $\psi_\pm(0,t) = \frac{1}{(2\pi\,\sigma(t)^2)^{\!1/4}} \exp\![-\frac{(\mp v\,t)^2}{4\,\sigma(t)^2}]\exp\![-i\,\frac{k_0^2}{2}\,t]$",
        color="white", fontsize=8
    )

    ax.set_xlabel("Re[ψ(0,t)]", color="white")
    ax.set_ylabel("Im[ψ(0,t)]", color="white")
    ax.set_zlabel("t (Photon 1 ↑, Photon 2 ↓)", color="white")

    # Remove grid lines, ticks, and background panes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    # Bounding box
    x_min = min(x1_vals.min(), x2_vals.min()) - 1
    x_max = max(x1_vals.max(), x2_vals.max()) + 1
    y_min = min(y1_vals.min(), y2_vals.min()) - 1
    y_max = max(y1_vals.max(), y2_vals.max()) + 1
    z_min = min(z1_vals.min(), z2_vals.min()) - 1
    z_max = max(z1_vals.max(), z2_vals.max()) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    #############################
    # 3D Spirals with Color Gradients
    #############################
    def make_spiral(x, y, z, cmap, tvals):
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(t_min, t_max)
        spiral = Line3DCollection(segments, cmap=cmap, norm=norm, lw=2)
        spiral.set_array(tvals)  # color by time
        return spiral

    spiral1 = make_spiral(x1_vals, y1_vals, z1_vals, plt.cm.plasma, t_vals)
    spiral2 = make_spiral(x2_vals, y2_vals, z2_vals, plt.cm.winter, t_vals)
    ax.add_collection3d(spiral1)
    ax.add_collection3d(spiral2)

    #############################
    # Static 2D Projections ("shadows")
    #############################
    ax.plot(x1_vals, [proj_offset_re_time] * len(y1_vals), z1_vals, color="red", lw=1.0, alpha=0.8, linestyle="--")
    ax.plot([proj_offset_im_time] * len(x1_vals), y1_vals, z1_vals, color="blue", lw=1.0, alpha=0.8, linestyle="--")
    ax.plot(x1_vals, y1_vals, [proj_offset_re_im] * len(z1_vals), color="white", lw=1.0, alpha=0.5, linestyle="--")

    ax.plot(x2_vals, [proj_offset_re_time] * len(y2_vals), z2_vals, color="red", lw=1.0, alpha=0.8, linestyle="--")
    ax.plot([proj_offset_im_time] * len(x2_vals), y2_vals, z2_vals, color="blue", lw=1.0, alpha=0.8, linestyle="--")
    ax.plot(x2_vals, y2_vals, [proj_offset_re_im] * len(z2_vals), color="white", lw=1.0, alpha=0.5, linestyle="--")

    #############################
    # Translucent Planes (no mesh)
    #############################
    Nx, Ny = 2, 2
    # y = proj_offset_re_time
    Xp = np.linspace(x_min, x_max, Nx)
    Zp = np.linspace(z_min, z_max, Ny)
    Xp, Zp = np.meshgrid(Xp, Zp)
    Yp = proj_offset_re_time * np.ones_like(Xp)
    ax.plot_surface(Xp, Yp, Zp, alpha=0.3, color="red", edgecolor='none')

    # x = proj_offset_im_time
    Yp2 = np.linspace(y_min, y_max, Nx)
    Zp2 = np.linspace(z_min, z_max, Ny)
    Yp2, Zp2 = np.meshgrid(Yp2, Zp2)
    Xp2 = proj_offset_im_time * np.ones_like(Yp2)
    ax.plot_surface(Xp2, Yp2, Zp2, alpha=0.3, color="blue", edgecolor='none')

    # z = proj_offset_re_im
    Xp3 = np.linspace(x_min, x_max, Nx)
    Yp3 = np.linspace(y_min, y_max, Ny)
    Xp3, Yp3 = np.meshgrid(Xp3, Yp3)
    Zp3 = proj_offset_re_im * np.ones_like(Xp3)
    ax.plot_surface(Xp3, Yp3, Zp3, alpha=0.5, color="grey", edgecolor='none')

    #############################
    # Animated Markers & "Entanglement" Link
    #############################
    arrow1, = ax.plot([], [], [], color="orange", marker='o', lw=2, label="Wave Packet 1")
    arrow2, = ax.plot([], [], [], color="cyan", marker='o', lw=2, label="Wave Packet 2")
    dash1, = ax.plot([], [], [], 'r--', lw=1)
    dash2, = ax.plot([], [], [], 'r--', lw=1)
    ent_line, = ax.plot([], [], [], '--', color="magenta", lw=2, label="Entanglement Link (Conceptual)")

    def update(frame):
        # Update markers for Photon 1
        x1f, y1f, z1f = x1_vals[frame], y1_vals[frame], z1_vals[frame]
        arrow1.set_data([0, x1f], [0, y1f])
        arrow1.set_3d_properties([0, z1f])
        dash1.set_data([x1f, x1f], [y1f, y1f])
        dash1.set_3d_properties([z1f, proj_offset_re_im])

        # Update markers for Photon 2
        x2f, y2f, z2f = x2_vals[frame], y2_vals[frame], z2_vals[frame]
        arrow2.set_data([0, x2f], [0, y2f])
        arrow2.set_3d_properties([0, z2f])
        dash2.set_data([x2f, x2f], [y2f, y2f])
        dash2.set_3d_properties([z2f, proj_offset_re_im])

        # Update entanglement link between the two photons
        ent_line.set_data([x1f, x2f], [y1f, y2f])
        ent_line.set_3d_properties([z1f, z2f])
        pulse = 2.0 + np.abs(np.sin(2 * np.pi * frame / n_frames))
        ent_line.set_linewidth(pulse)

        # Rotate the view: complete one full rotation over the duration of the animation
        azimuth = 75 + (frame * 360 / n_frames)
        ax.view_init(elev=25, azim=azimuth)

        return arrow1, dash1, arrow2, dash2, ent_line

    # Note: set blit=False so the full figure (including the view) updates
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    auto_save_gif(ani, "simple_entanglement.gif")

    ax.dist = 15
    ax.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
