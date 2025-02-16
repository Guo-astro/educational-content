import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import RadioButtons
from matplotlib.animation import FuncAnimation

# Enable LaTeX rendering (if you have a full LaTeX installation)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
params = {'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)

# --- Define Octahedron Geometry ---
vertices = np.array([
    [1, 0, 0],    # A
    [-1, 0, 0],   # B
    [0, 1, 0],    # C
    [0, -1, 0],   # D
    [0, 0, 1],    # E
    [0, 0, -1]    # F
])
vertex_labels = ['A', 'B', 'C', 'D', 'E', 'F']

faces = [
    [0, 2, 4],
    [2, 1, 4],
    [1, 3, 4],
    [3, 0, 4],
    [0, 2, 5],
    [2, 1, 5],
    [1, 3, 5],
    [3, 0, 5]
]

# --- Generate Octahedral Group (48 operations) ---
def generate_octahedral_group():
    matrices = []
    for perm in permutations([0, 1, 2]):
        for signs in product([-1, 1], repeat=3):
            mat = np.zeros((3, 3), dtype=int)
            for i, j in enumerate(perm):
                mat[i, j] = signs[i]
            matrices.append(mat)
    return matrices

group_matrices = generate_octahedral_group()
print("Number of group operations:", len(group_matrices))  # Should print 48

# --- Helper: Create a text string for the transformation equation ---
def get_formula_text(mat):
    def format_row(coeff, var):
        s = r"${}' = ".format(var)
        s += f"{coeff[0]}x"
        if coeff[1] >= 0:
            s += f" + {coeff[1]}y"
        else:
            s += f" - {abs(coeff[1])}y"
        if coeff[2] >= 0:
            s += f" + {coeff[2]}z"
        else:
            s += f" - {abs(coeff[2])}z"
        s += "$"
        return s
    line1 = format_row(mat[0], "x")
    line2 = format_row(mat[1], "y")
    line3 = format_row(mat[2], "z")
    s = "Transformation Equation:\n" + line1 + "\n" + line2 + "\n" + line3
    return s

# --- Helper: Compute a descriptive label for a transformation ---
def get_transformation_label(mat):
    det = np.round(np.linalg.det(mat))
    if np.allclose(mat, np.eye(3)):
        return "Identity"
    if det == 1:
        # Rotation: compute rotation angle and axis.
        angle = np.degrees(np.arccos((np.trace(mat) - 1) / 2))
        eigvals, eigvecs = np.linalg.eig(mat)
        # Find eigenvector corresponding to eigenvalue ~1.
        idx = np.argmin(np.abs(eigvals - 1))
        axis = eigvecs[:, idx].real
        axis = axis / np.linalg.norm(axis)
        angle_rounded = round(angle, 1)
        axis_str = f"({axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f})"
        return f"Rotation {angle_rounded}° about axis {axis_str}"
    else:
        # Improper transformation: check if pure reflection (trace ~1) or rotatory reflection.
        if np.isclose(np.trace(mat), 1, atol=1e-5):
            eigvals, eigvecs = np.linalg.eig(mat)
            idx = np.argmin(np.abs(eigvals + 1))
            normal = eigvecs[:, idx].real
            normal = normal / np.linalg.norm(normal)
            normal_str = f"({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
            return f"Reflection across plane with normal {normal_str}"
        else:
            eigvals, eigvecs = np.linalg.eig(mat)
            idx = np.argmin(np.abs(eigvals + 1))
            normal = eigvecs[:, idx].real
            normal = normal / np.linalg.norm(normal)
            normal_str = f"({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
            return f"Rotatory Reflection with reversal direction {normal_str}"

# --- Plotting Function for an Octahedron ---
def plot_octahedron(ax, verts, title_text=""):
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=30)
    # Plot vertices
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], color='red', s=50)
    for i, (x, y, z) in enumerate(verts):
        ax.text(x, y, z, f' {vertex_labels[i]}', color='black', fontsize=12)
    # Plot faces as translucent polygons
    face_verts = [verts[face] for face in faces]
    poly3d = Poly3DCollection(face_verts, alpha=0.3, edgecolor='k')
    poly3d.set_facecolor('cyan')
    ax.add_collection3d(poly3d)
    ax.set_title(title_text, fontsize=14, pad=20)

# --- Helper: Draw the transformation indicator (axis or mirror plane) ---
def draw_transformation_indicator(ax, mat):
    det = np.round(np.linalg.det(mat))
    if det == 1:
        # Rotation: draw the axis (eigenvector with eigenvalue 1).
        eigvals, eigvecs = np.linalg.eig(mat)
        idx = np.argmin(np.abs(eigvals - 1))
        axis = eigvecs[:, idx].real
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-5:
            return  # Skip if degenerate (e.g. identity)
        axis = axis / axis_norm
        L = 2.5
        line_points = np.array([-L * axis, L * axis])
        ax.plot(line_points[:,0], line_points[:,1], line_points[:,2],
                color='magenta', linewidth=2, label="Rotation Axis")
    else:
        # For improper transformations: check if pure reflection.
        if np.isclose(np.trace(mat), 1, atol=1e-5):
            # Pure reflection: eigenvalues should be (1,1,-1)
            eigvals, eigvecs = np.linalg.eig(mat)
            idx = np.argmin(np.abs(eigvals + 1))
            normal = eigvecs[:, idx].real
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-5:
                return
            normal = normal / n_norm
            # Construct two in-plane vectors
            if abs(normal[0]) < abs(normal[1]):
                vtemp = np.array([1, 0, 0])
            else:
                vtemp = np.array([0, 1, 0])
            u = np.cross(normal, vtemp)
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            # Create a grid for the plane
            grid_range = np.linspace(-2, 2, 5)
            U, V = np.meshgrid(grid_range, grid_range)
            X = U*u[0] + V*v[0]
            Y = U*u[1] + V*v[1]
            Z = U*u[2] + V*v[2]
            ax.plot_surface(X, Y, Z, color='orange', alpha=0.3)
        else:
            # Rotatory reflection: draw the reversal direction as a dashed arrow.
            eigvals, eigvecs = np.linalg.eig(mat)
            idx = np.argmin(np.abs(eigvals + 1))
            normal = eigvecs[:, idx].real
            n_norm = np.linalg.norm(normal)
            if n_norm < 1e-5:
                return
            normal = normal / n_norm
            L = 2.5
            arrow_points = np.array([[0, 0, 0], L*normal])
            ax.plot(arrow_points[:,0], arrow_points[:,1], arrow_points[:,2],
                    color='green', linestyle='--', linewidth=2, label="Reversal Direction")

# --- Create Figure with Two Subplots ---
fig = plt.figure(figsize=(12, 6))
ax_base = fig.add_subplot(121, projection='3d')
ax_trans = fig.add_subplot(122, projection='3d')

# Plot the original octahedron on the base (left subplot)
plot_octahedron(ax_base, vertices, "Original Octahedron (Base)")
# Initially, the transformed view is the same as the base.
plot_octahedron(ax_trans, vertices, "Transformed Octahedron")

# Global handle for animation so it persists.
current_anim_trans = None

# --- Animation for Right Plot (Transformation Animation) ---
def animate_transformation(mat, frames=30, interval=50):
    global current_anim_trans
    transformed_vertices = (mat @ vertices.T).T  # Apply transformation to all vertices.
    def update_frame(frame):
        t = frame / frames
        # Linear interpolation between original and transformed vertices.
        intermediate_vertices = (1 - t) * vertices + t * transformed_vertices
        ax_trans.clear()
        plot_octahedron(ax_trans, intermediate_vertices, f"Transformed Octahedron\n{get_formula_text(mat)}")
        draw_transformation_indicator(ax_trans, mat)
    current_anim_trans = FuncAnimation(fig, update_frame, frames=frames+1,
                                         interval=interval, repeat=False)
    return current_anim_trans

# --- Create Transformation Options with Labels ---
# Precompute a list of labels for each transformation.
transformation_options = []
transform_dict = {}  # Map label -> matrix.
for i, mat in enumerate(group_matrices):
    label = f"{i}: {get_transformation_label(mat)}"
    transformation_options.append(label)
    transform_dict[label] = mat

# --- Create RadioButtons for Transformation Selection ---
# (Note: With 48 options the list is long. Consider reducing the options if needed.)
radio_ax = plt.axes([0.78, 0.1, 0.20, 0.8], facecolor='lightgoldenrodyellow')
radio = RadioButtons(radio_ax, transformation_options, active=0)

# --- Callback for RadioButtons ---
def radio_callback(label):
    mat = transform_dict[label]
    animate_transformation(mat)
    fig.canvas.draw_idle()

radio.on_clicked(radio_callback)

plt.show()
