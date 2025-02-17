import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

plt.style.use("dark_background")


def free_gaussian_packet(x, t, sigma0=1.0, k0=1.0, v=None, t_s=None):
    """
    Returns the complex Gaussian wave packet ψ(x,t):

      ψ(x,t) = 1/(2π σ(t)^2)^(1/4) exp[-(x - v t)²/(4σ(t)^2)]
               exp[i(k₀(x - v t) - (k₀²/2)t)]

    with σ(t)² = σ₀² (1 + i t/tₛ) (with tₛ = 2σ₀² by default).
    """
    if t_s is None:
        t_s = 2.0 * sigma0 ** 2
    if v is None:
        v = k0
    sigma_t_sq = sigma0 ** 2 * (1 + 1j * t / t_s)
    prefactor = 1.0 / ((2 * np.pi * sigma_t_sq) ** 0.25)
    gauss = np.exp(-((x - v * t) ** 2) / (4.0 * sigma_t_sq))
    phase = np.exp(1j * (k0 * (x - v * t) - 0.5 * k0 ** 2 * t))
    return prefactor * gauss * phase


def psi_plus(x, t, sigma0, k0):
    return free_gaussian_packet(x, t, sigma0, k0, v=k0)


def psi_minus(x, t, sigma0, k0):
    return free_gaussian_packet(x, t, sigma0, k0, v=-k0)


def psi_entangled(x1, x2, t, sigma0, k0):
    """
    Entangled two-photon state:
      Ψ(x₁,x₂,t)=1/√2 [ψ₊(x₁,t)ψ₋(x₂,t)+ψ₋(x₁,t)ψ₊(x₂,t)]
    """
    return (1 / np.sqrt(2)) * (psi_plus(x1, t, sigma0, k0) * psi_minus(x2, t, sigma0, k0) +
                               psi_minus(x1, t, sigma0, k0) * psi_plus(x2, t, sigma0, k0))


def main():
    # Parameters for the wave packets and animation
    sigma0 = 1.0
    k0 = 2.0
    fps = 30
    duration = 10  # seconds
    n_frames = fps * duration
    t_vals = np.linspace(-2, 2, n_frames)

    # Spatial coordinate for the conditional wave functions
    x_range = np.linspace(-20, 20, 1000)

    # Increase the separation distance for better visualization:
    offset_x = 40  # separation between Photon 1 and Photon 2

    # Define axis limits for the main plot
    x1_min, x1_max = -20, 20      # For Photon 1 (conditional on x₂=0)
    x2_max = offset_x + 20        # For Photon 2 (conditional on x₁=0)
    re_min, re_max = -1.5, 1.5
    im_min, im_max = -1.5, 1.5

    # Adjusted offsets for the projection planes (now closer to the data)
    proj_offset_z = im_min      # x–Re plane: fix z (Im) at -1.5
    proj_offset_y = re_min      # x–Im plane: fix y (Re) at -1.5
    proj_offset_x = x1_min      # Re–Im plane: fix x at -20

    # Create figure and 3D axis
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Re(ψ)", fontsize=12)
    ax.set_zlabel("Im(ψ)", fontsize=12)
    ax.set_xlim(x1_min, x2_max)
    ax.set_ylim(re_min, re_max)
    ax.set_zlim(im_min, im_max)

    # Remove grid and panes for a clean look
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    # --- Draw Projection Planes ---
    # x–Re plane (Im fixed at proj_offset_z)
    X_plane, Y_plane = np.meshgrid(np.linspace(x1_min, x2_max, 2),
                                   np.linspace(re_min, re_max, 2))
    Z_plane = np.full_like(X_plane, proj_offset_z)
    ax.plot_surface(X_plane, Y_plane, Z_plane, color='dimgray', alpha=0.5, zorder=0)

    # x–Im plane (Re fixed at proj_offset_y)
    X_plane2, Z_plane2 = np.meshgrid(np.linspace(x1_min, x2_max, 2),
                                     np.linspace(im_min, im_max, 2))
    Y_plane2 = np.full_like(X_plane2, proj_offset_y)
    ax.plot_surface(X_plane2, Y_plane2, Z_plane2, color='dimgray', alpha=0.5, zorder=0)

    # Re–Im plane (x fixed at proj_offset_x)
    Y_plane3, Z_plane3 = np.meshgrid(np.linspace(re_min, re_max, 2),
                                     np.linspace(im_min, im_max, 2))
    X_plane3 = np.full_like(Y_plane3, proj_offset_x)
    ax.plot_surface(X_plane3, Y_plane3, Z_plane3, color='dimgray', alpha=0.5, zorder=0)

    # --- Add Informative Text ---
    fig.text(0.0, 0.88,
             r"Governing Equation: $i\frac{\partial \psi}{\partial t}=-\frac{1}{2}\frac{\partial^2 \psi}{\partial x^2}$",
             color='white', fontsize=12)
    fig.text(0.0, 0.83,
             r"Entangled State: $\Psi(x_1,x_2,t)=\frac{1}{\sqrt{2}}\left[\psi_+(x_1,t)\psi_-(x_2,t)+\psi_-(x_1,t)\psi_+(x_2,t)\right]$",
             color='white', fontsize=12)
    fig.text(0.0, 0.78,
             r"$\psi_+(x,t)=\frac{1}{(2\pi\sigma(t)^2)^{1/4}}\exp\left[-\frac{(x-k_0t)^2}{4\sigma(t)^2}\right]\exp\left[i\left(k_0(x-k_0t)-\frac{k_0^2}{2}t\right)\right]$",
             color='white', fontsize=12)
    fig.text(0.0, 0.73,
             r"$\psi_-(x,t)=\frac{1}{(2\pi\sigma(t)^2)^{1/4}}\exp\left[-\frac{(x+k_0t)^2}{4\sigma(t)^2}\right]\exp\left[i\left(k_0(x+k_0t)-\frac{k_0^2}{2}t\right)\right]$",
             color='white', fontsize=12)
    # --- Add Informative Text with Dirac Notation ---
    fig.text(0.0, 0.68,
             r"In Dirac Notation:",
             color='white', fontsize=12)
    fig.text(0.0, 0.63,
             r"Governing Equation: $i\frac{\partial}{\partial t}\vert\psi\rangle = -\frac{1}{2}\frac{\partial^2}{\partial x^2}\vert\psi\rangle$",
             color='white', fontsize=12)
    fig.text(0.0, 0.58,
             r"Entangled State: $\vert\Psi(t)\rangle = \frac{1}{\sqrt{2}}\left(\vert\psi_+(t)\rangle_1\vert\psi_-(t)\rangle_2 + \vert\psi_-(t)\rangle_1\vert\psi_+(t)\rangle_2\right)$",
             color='white', fontsize=12)
    fig.text(0.0, 0.53,
             r"$\vert\psi_+(t)\rangle = \frac{1}{(2\pi\sigma(t)^2)^{1/4}}\exp\!\left[-\frac{(x-k_0t)^2}{4\sigma(t)^2}\right]\exp\!\left[i\left(k_0(x-k_0t)-\frac{k_0^2}{2}t\right)\right]$",
             color='white', fontsize=12)
    fig.text(0.0, 0.48,
             r"$\vert\psi_-(t)\rangle = \frac{1}{(2\pi\sigma(t)^2)^{1/4}}\exp\!\left[-\frac{(x+k_0t)^2}{4\sigma(t)^2}\right]\exp\!\left[i\left(k_0(x+k_0t)-\frac{k_0^2}{2}t\right)\right]$",
             color='white', fontsize=12)
    # --- Compute the Conditional Wave Functions ---
    t0 = t_vals[0]
    # For Photon 1, fix x₂ = 0:
    phi1 = psi_entangled(x_range, 0, t0, sigma0, k0)
    # For Photon 2, fix x₁ = 0:
    phi2 = psi_entangled(0, x_range, t0, sigma0, k0)

    # --- Plot the Main Spiral Curves (Conditional States) ---
    line1, = ax.plot(x_range, np.real(phi1), np.imag(phi1), color='dodgerblue', lw=2,
                     label="Photon 1 (x₂=0)")
    line2, = ax.plot(x_range + offset_x, np.real(phi2), np.imag(phi2), color='orangered', lw=2,
                     label="Photon 2 (x₁=0)")

    # --- Draw Projections for Photon 1 ---
    proj1_xRe, = ax.plot(x_range, np.real(phi1), zs=proj_offset_z, color='dodgerblue', lw=1.5, ls='--')
    proj1_xIm, = ax.plot(x_range, np.full_like(x_range, proj_offset_y), np.imag(phi1),
                         color='dodgerblue', lw=1.5, ls='--')
    proj1_ReIm, = ax.plot(np.full_like(x_range, proj_offset_x), np.real(phi1), np.imag(phi1),
                          color='dodgerblue', lw=1.5, ls='--')

    # --- Draw Projections for Photon 2 (with x offset) ---
    proj2_xRe, = ax.plot(x_range + offset_x, np.real(phi2), zs=proj_offset_z, color='orangered', lw=1.5, ls='--')
    proj2_xIm, = ax.plot(x_range + offset_x, np.full_like(x_range, proj_offset_y), np.imag(phi2),
                         color='orangered', lw=1.5, ls='--')
    proj2_ReIm, = ax.plot(np.full_like(x_range, proj_offset_x), np.real(phi2), np.imag(phi2),
                          color='orangered', lw=1.5, ls='--')

    # --- Draw a Connecting Line (correlation) ---
    idx1 = np.argmax(np.abs(phi1))
    idx2 = np.argmax(np.abs(phi2))
    conn_line, = ax.plot([x_range[idx1], x_range[idx2] + offset_x],
                         [np.real(phi1[idx1]), np.real(phi2[idx2])],
                         [np.imag(phi1[idx1]), np.imag(phi2[idx2])],
                         color='lime', lw=2, ls=':')

    # --- Draw Projections for the Connecting Line ---
    conn_proj_xRe, = ax.plot([x_range[idx1], x_range[idx2] + offset_x],
                             [np.real(phi1[idx1]), np.real(phi2[idx2])],
                             [proj_offset_z, proj_offset_z],
                             color='lime', lw=1.5, ls='--')
    conn_proj_xIm, = ax.plot([x_range[idx1], x_range[idx2] + offset_x],
                             [proj_offset_y, proj_offset_y],
                             [np.imag(phi1[idx1]), np.imag(phi2[idx2])],
                             color='lime', lw=1.5, ls='--')
    conn_proj_ReIm, = ax.plot([proj_offset_x, proj_offset_x],
                              [np.real(phi1[idx1]), np.real(phi2[idx2])],
                              [np.imag(phi1[idx1]), np.imag(phi2[idx2])],
                              color='lime', lw=1.5, ls='--')

    # --- Add Time Annotation ---
    time_text = ax.text2D(0.85, 0.85, f"t = {t0:.2f}", transform=ax.transAxes, color='white', fontsize=12)
    ax.legend(loc="upper right", fontsize=12)

    def update(frame):
        t = t_vals[frame]
        phi1 = psi_entangled(x_range, 0, t, sigma0, k0)
        phi2 = psi_entangled(0, x_range, t, sigma0, k0)

        # Update the spiral curves
        line1.set_data(x_range, np.real(phi1))
        line1.set_3d_properties(np.imag(phi1))
        line2.set_data(x_range + offset_x, np.real(phi2))
        line2.set_3d_properties(np.imag(phi2))

        # Update projections for Photon 1
        proj1_xRe.set_data(x_range, np.real(phi1))
        proj1_xRe.set_3d_properties(np.full_like(x_range, proj_offset_z))
        proj1_xIm.set_data(x_range, np.full_like(x_range, proj_offset_y))
        proj1_xIm.set_3d_properties(np.imag(phi1))
        proj1_ReIm.set_data(np.full_like(x_range, proj_offset_x), np.real(phi1))
        proj1_ReIm.set_3d_properties(np.imag(phi1))

        # Update projections for Photon 2
        proj2_xRe.set_data(x_range + offset_x, np.real(phi2))
        proj2_xRe.set_3d_properties(np.full_like(x_range, proj_offset_z))
        proj2_xIm.set_data(x_range + offset_x, np.full_like(x_range, proj_offset_y))
        proj2_xIm.set_3d_properties(np.imag(phi2))
        proj2_ReIm.set_data(np.full_like(x_range, proj_offset_x), np.real(phi2))
        proj2_ReIm.set_3d_properties(np.imag(phi2))

        # Update connecting line
        idx1 = np.argmax(np.abs(phi1))
        idx2 = np.argmax(np.abs(phi2))
        conn_line.set_data([x_range[idx1], x_range[idx2] + offset_x],
                           [np.real(phi1[idx1]), np.real(phi2[idx2])])
        conn_line.set_3d_properties([np.imag(phi1[idx1]), np.imag(phi2[idx2])])

        # Update connecting line projections
        conn_proj_xRe.set_data([x_range[idx1], x_range[idx2] + offset_x],
                               [np.real(phi1[idx1]), np.real(phi2[idx2])])
        conn_proj_xRe.set_3d_properties([proj_offset_z, proj_offset_z])
        conn_proj_xIm.set_data([x_range[idx1], x_range[idx2] + offset_x],
                               [proj_offset_y, proj_offset_y])
        conn_proj_xIm.set_3d_properties([np.imag(phi1[idx1]), np.imag(phi2[idx2])])
        conn_proj_ReIm.set_data([proj_offset_x, proj_offset_x],
                                [np.real(phi1[idx1]), np.real(phi2[idx2])])
        conn_proj_ReIm.set_3d_properties([np.imag(phi1[idx1]), np.imag(phi2[idx2])])

        # Update time annotation
        time_text.set_text(f"t = {t:.2f}")

        # --- Rotate the view slowly ---
        ax.view_init(elev=30, azim=(frame / n_frames * 360))
        return (line1, line2,
                proj1_xRe, proj1_xIm, proj1_ReIm,
                proj2_xRe, proj2_xIm, proj2_ReIm,
                conn_line, conn_proj_xRe, conn_proj_xIm, conn_proj_ReIm,
                time_text)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    # To save the animation, uncomment the following line:
    # ani.save("entangled_two_photon_rotate.gif", writer="pillow", fps=fps)
    plt.show()


if __name__ == "__main__":
    main()
