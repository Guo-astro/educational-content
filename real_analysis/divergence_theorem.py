import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.animation import FuncAnimation

# Create a figure with two 3D subplots side by side
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121, projection='3d')  # Left: Streamline animation
ax2 = fig.add_subplot(122, projection='3d')  # Right: Local derivative demo

#######################################
# Common Vector Field: F(x,y,z) = (x,y,z)
#######################################

# This function returns the field at (x,y,z)
def F(x, y, z):
    return np.array([x, y, z])

#######################################
# Left Plot: Streamline Animation
#######################################

ax1.set_title("Streamline Animation for F(x,y,z)=(x,y,z)")
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])
ax1.set_zlim([-2, 2])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Choose a fixed direction Q0 for the streamline (a ray from the origin)
Q0 = np.array([0.8, 0.5, 0.3])
# Draw a static dashed line along the ray from the origin
t_line = np.linspace(0, 2, 100)
line_pts = np.outer(t_line, Q0)
ax1.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2],
         color='blue', linestyle='--', label='Streamline')

# Initialize the moving arrow and its annotation (to be updated)
arrow1 = None
text1 = ax1.text2D(0.05, 0.95, '', transform=ax1.transAxes,
                   fontsize=14, color='magenta')

#######################################
# Right Plot: Local Derivative Demonstration
#######################################

ax2.set_title("Local Derivative Demonstration for F(x,y,z)=(x,y,z)")
ax2.set_xlim([0.5, 2])
ax2.set_ylim([0.5, 2])
ax2.set_zlim([0.5, 2])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Base point for the demonstration
P = np.array([1.0, 1.0, 1.0])
ax2.scatter(*P, color='blue', s=50, label='Base Point')

# Initialize objects for the derivative demo:
arrow2 = None   # red arrow for the field difference
line2 = None    # dashed line showing displacement
text2 = ax2.text2D(0.05, 0.95, '', transform=ax2.transAxes,
                   fontsize=14, color='magenta')

#######################################
# Animation Update Function
#######################################

# For left plot, we let t go from 0 to 2π (cyclic motion).
total_frames_left = 360
# For the derivative demo, we divide the cycle into 3 segments (x, y, z).
segment_frames = 60
total_frames_right = 3 * segment_frames  # 180 frames

def update(frame):
    global arrow1, arrow2, line2

    ###############################
    # Update Left Plot (ax1): Streamline Animation
    ###############################
    # Compute a parameter t that cycles from 0 to 2pi
    t = (frame % total_frames_left) * (2 * np.pi / total_frames_left)
    # Let the arrow's position oscillate along the fixed streamline:
    # Q(t) = Q0 * (1 + 0.5*sin(t)), so it moves between 0.5*Q0 and 1.5*Q0.
    Q_left = Q0 * (1 + 0.5 * np.sin(t))
    # The field at that point is F(Q_left) = Q_left.
    vec_left = Q_left
    # Remove previous arrow if it exists
    if arrow1 is not None:
        arrow1.remove()
    # Draw the new arrow at Q_left, with direction vec_left.
    arrow1 = ax1.quiver(Q_left[0], Q_left[1], Q_left[2],
                        vec_left[0], vec_left[1], vec_left[2],
                        length=0.5, color='red', normalize=True)
    # Update text: display the parameter t and current position
    text1.set_text(f"$t = {t:.2f}$\n$Q(t)=({Q_left[0]:.2f}, {Q_left[1]:.2f}, {Q_left[2]:.2f})$")

    ###############################
    # Update Right Plot (ax2): Local Derivative Demo
    ###############################
    # Use frame modulo the right-cycle total (180 frames)
    frame_right = frame % total_frames_right
    dt = 0.3  # maximum displacement magnitude for each segment
    if frame_right < segment_frames:
        # Segment 1: Displace along the x-direction.
        frac = frame_right / segment_frames  # goes from 0 to 1
        d_disp = np.array([frac * dt, 0, 0])
        direction = "x"
    elif frame_right < 2 * segment_frames:
        # Segment 2: Displace along the y-direction.
        frac = (frame_right - segment_frames) / segment_frames
        d_disp = np.array([0, frac * dt, 0])
        direction = "y"
    else:
        # Segment 3: Displace along the z-direction.
        frac = (frame_right - 2 * segment_frames) / segment_frames
        d_disp = np.array([0, 0, frac * dt])
        direction = "z"
    # Compute the displaced point Q = P + d_disp.
    Q_right = P + d_disp
    # The field difference: F(Q_right) - F(P) = (Q_right - P) because F is linear.
    dF = Q_right - P
    # Remove previous arrow and line if they exist
    if arrow2 is not None:
        arrow2.remove()
    if line2 is not None:
        line2.remove()
    # Draw the red arrow at P representing the difference vector dF.
    arrow2 = ax2.quiver(P[0], P[1], P[2],
                        dF[0], dF[1], dF[2],
                        length=0.5, color='red', normalize=True)
    # Draw a dashed green line from P to Q_right
    line2, = ax2.plot([P[0], Q_right[0]], [P[1], Q_right[1]], [P[2], Q_right[2]],
                      color='green', linestyle='--')
    # Update text annotation for the derivative demonstration.
    if direction == "x":
        text2.set_text(f"$\\Delta x = {d_disp[0]:.2f}$\n"
                       f"$\\Delta F_x = {dF[0]:.2f}$\n"
                       r"$\frac{\Delta F_x}{\Delta x} \approx 1$")
    elif direction == "y":
        text2.set_text(f"$\\Delta y = {d_disp[1]:.2f}$\n"
                       f"$\\Delta F_y = {dF[1]:.2f}$\n"
                       r"$\frac{\Delta F_y}{\Delta y} \approx 1$")
    else:
        text2.set_text(f"$\\Delta z = {d_disp[2]:.2f}$\n"
                       f"$\\Delta F_z = {dF[2]:.2f}$\n"
                       r"$\frac{\Delta F_z}{\\Delta z} \approx 1$")

    return arrow1, arrow2, text1, text2, line2

# Create the animation.
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

plt.tight_layout()
plt.show()
