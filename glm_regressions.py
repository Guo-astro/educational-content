import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import math

##############################################################################
# PDF / PMF helper functions
##############################################################################

def normal_pdf(y, mean, sd):
    return (1.0 / (sd * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((y - mean)/sd)**2)

def beta_pdf(y, alpha, beta):
    from math import gamma
    if y < 0 or y > 1:
        return 0.0
    B = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    return (1.0/B) * (y**(alpha-1)) * ((1-y)**(beta-1))

def gamma_pdf(y, shape, rate):
    from math import gamma
    if y <= 0:
        return 0.0
    return (rate**shape / gamma(shape)) * (y**(shape-1)) * np.exp(-rate*y)

def poisson_pmf(k, lam):
    from math import factorial
    if k < 0:
        return 0.0
    return (lam**k)*np.exp(-lam)/float(factorial(k))

def negbinom_pmf(k, r, p):
    from math import comb
    if k < 0:
        return 0.0
    return comb(k + r - 1, k)*(p**k)*((1-p)**r)

##############################################################################
# Configuration for UI improvements: raise regression line and text above bars.
##############################################################################
line_z_offset = 0.3  # extra offset for regression lines/text in the z-direction
offset_x_default = 0.1
offset_z_default = 0.1

##############################################################################
# Plot a 2×3 figure of subplots in 3D with regression lines and annotations.
##############################################################################
fig = plt.figure(figsize=(16, 10))

# Create 6 subplots (row-major order)
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax6 = fig.add_subplot(2, 3, 6, projection='3d')

##############################################################################
# 1) Linear (Gaussian)
##############################################################################
ax1.set_title("Linear regression (Gaussian)")
ax1.set_xlabel(r"$y$")
ax1.set_ylabel(r"$X$ index")
ax1.set_zlabel("PDF")

x_indices = [1, 2, 3]
y_vals = np.linspace(0, 8, 300)
stdev = 0.7
means = [2, 4, 6]

for x_idx, m in zip(x_indices, means):
    pdf_vals = [normal_pdf(y, m, stdev) for y in y_vals]
    ax1.plot(y_vals, [x_idx]*len(y_vals), pdf_vals)

reg_line_x = means
reg_line_y = x_indices
reg_line_z = [normal_pdf(m, m, stdev) for m in means]
reg_line_z_plot = [z + line_z_offset for z in reg_line_z]
ax1.plot(reg_line_x, reg_line_y, reg_line_z_plot, color='black', lw=2, marker='o')
for x_idx, m, z in zip(x_indices, means, reg_line_z):
    ax1.text(m + offset_x_default, x_idx, z + offset_z_default + line_z_offset,
             f"$E(y|x={x_idx})={m}$",
             color='black',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

##############################################################################
# 2) Beta regression
##############################################################################
ax2.set_title("Beta regression")
ax2.set_xlabel(r"$y \in [0,1]$")
ax2.set_ylabel(r"$X$ index")
ax2.set_zlabel("PDF")

betas = [(3,7), (5,5), (7,3)]
y_beta = np.linspace(0, 1, 300)

for x_idx, (a, b) in zip(x_indices, betas):
    pdf_vals = [beta_pdf(y, a, b) for y in y_beta]
    ax2.plot(y_beta, [x_idx]*len(y_beta), pdf_vals)

reg_line_x = []
reg_line_y = x_indices
reg_line_z = []
for x_idx, (a, b) in zip(x_indices, betas):
    E_val = a / (a + b)
    reg_line_x.append(E_val)
    reg_line_z.append(beta_pdf(E_val, a, b))
reg_line_z_plot = [z + line_z_offset for z in reg_line_z]
ax2.plot(reg_line_x, reg_line_y, reg_line_z_plot, color='black', lw=2, marker='o')
for x_idx, E_val, z in zip(x_indices, reg_line_x, reg_line_z):
    ax2.text(E_val + 0.02, x_idx, z + 0.02 + line_z_offset,
             f"$E(y|x={x_idx})={E_val:.2f}$",
             color='black',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

##############################################################################
# 3) Gamma regression
##############################################################################
ax3.set_title("Gamma regression")
ax3.set_xlabel(r"$y>0$")
ax3.set_ylabel(r"$X$ index")
ax3.set_zlabel("PDF")

shape = 2
g_means = [2, 4, 6]
rates = [shape / m for m in g_means]
y_gam = np.linspace(0, 15, 400)

for x_idx, (mean_val, rate) in zip(x_indices, zip(g_means, rates)):
    pdf_vals = [gamma_pdf(y, shape, rate) for y in y_gam]
    ax3.plot(y_gam, [x_idx]*len(y_gam), pdf_vals)

reg_line_x = g_means
reg_line_y = x_indices
reg_line_z = [gamma_pdf(m, shape, rate) for m, rate in zip(g_means, rates)]
reg_line_z_plot = [z + line_z_offset for z in reg_line_z]
ax3.plot(reg_line_x, reg_line_y, reg_line_z_plot, color='black', lw=2, marker='o')
for x_idx, m, z in zip(x_indices, g_means, reg_line_z):
    ax3.text(m + offset_x_default, x_idx, z + offset_z_default + line_z_offset,
             f"$E(y|x={x_idx})={m}$",
             color='black',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

##############################################################################
# 4) Poisson regression
##############################################################################
ax4.set_title("Poisson regression")
ax4.set_xlabel(r"$k \in \{0,1,2,\dots\}$")
ax4.set_ylabel(r"$X$ index")
ax4.set_zlabel("PMF")

lams = [1, 4, 7]
max_k = 15
dx = 0.8
dy = 0.8

for x_idx, lam in zip(x_indices, lams):
    for k in range(max_k + 1):
        zval = poisson_pmf(k, lam)
        ax4.bar3d(x=k, y=x_idx, z=0, dx=dx, dy=dy, dz=zval, alpha=0.6)
reg_line_x = lams
reg_line_y = x_indices
reg_line_z = [poisson_pmf(lam, lam) for lam in lams]
reg_line_z_plot = [z + line_z_offset for z in reg_line_z]
ax4.plot(reg_line_x, reg_line_y, reg_line_z_plot, color='black', lw=2, marker='o')
for x_idx, lam, z in zip(x_indices, lams, reg_line_z):
    ax4.text(lam + 0.2, x_idx, z + 0.2 + line_z_offset,
             f"$E(y|x={x_idx})={lam}$",
             color='black',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

##############################################################################
# 5) Negative-binomial regression
##############################################################################
ax5.set_title("Negative-binomial regression")
ax5.set_xlabel(r"$k \in \{0,1,2,\dots\}$")
ax5.set_ylabel(r"$X$ index")
ax5.set_zlabel("PMF")

r_val = 3
ps = [0.2, 0.5, 0.8]
max_k_nb = 20

for x_idx, p in zip(x_indices, ps):
    for k in range(max_k_nb + 1):
        zval = negbinom_pmf(k, r_val, p)
        ax5.bar3d(x=k, y=x_idx, z=0, dx=dx, dy=dy, dz=zval, alpha=0.6)
reg_line_x = []
reg_line_y = x_indices
reg_line_z = []
for x_idx, p in zip(x_indices, ps):
    E_val = (r_val * p) / (1 - p)
    k_point = round(E_val)
    reg_line_x.append(k_point)
    reg_line_z.append(negbinom_pmf(k_point, r_val, p))
reg_line_z_plot = [z + line_z_offset for z in reg_line_z]
ax5.plot(reg_line_x, reg_line_y, reg_line_z_plot, color='black', lw=2, marker='o')
for x_idx, p, E_val, k_point, z in zip(x_indices, ps, [(r_val*p)/(1-p) for p in ps],
                                         reg_line_x, reg_line_z):
    ax5.text(k_point + 0.2, x_idx, z + 0.2 + line_z_offset,
             f"$E(y|x={x_idx})={E_val:.2f}$",
             color='black',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

##############################################################################
# 6) Logistic regression (Bernoulli)
##############################################################################
ax6.set_title("Logistic regression (Bernoulli)")
ax6.set_xlabel(r"$y \in \{0,1\}$")
ax6.set_ylabel(r"$X$ index")
ax6.set_zlabel("Probability")

ps_bern = [0.2, 0.5, 0.8]

for x_idx, p in zip(x_indices, ps_bern):
    ax6.bar3d(x=0, y=x_idx, z=0, dx=0.4, dy=0.4, dz=(1-p), alpha=0.6, color='red')
    ax6.bar3d(x=1, y=x_idx, z=0, dx=0.4, dy=0.4, dz=p, alpha=0.6, color='blue')
reg_line_x = []
reg_line_y = x_indices
reg_line_z = []
for x_idx, p in zip(x_indices, ps_bern):
    z_val = (1-p) + (2*p - 1)*p
    reg_line_x.append(p)
    reg_line_z.append(z_val)
reg_line_z_plot = [z + line_z_offset for z in reg_line_z]
ax6.plot(reg_line_x, reg_line_y, reg_line_z_plot, color='black', lw=2, marker='o')
for x_idx, p, z in zip(x_indices, ps_bern, reg_line_z):
    ax6.text(p + 0.05, x_idx, z + 0.05 + line_z_offset,
             f"$E(y|x={x_idx})={p:.2f}$",
             color='black',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

##############################################################################
# Auto-rotation via animation
##############################################################################
def animate(angle):
    # Rotate all 6 axes by updating the azimuth angle
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.view_init(elev=30, azim=angle)
    return []

# Create the animation (adjust frames and interval as desired)
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), interval=100)

plt.subplots_adjust(wspace=0.3, hspace=0.3, right=0.9)
plt.show()
