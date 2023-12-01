import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import functions as fn

matplotlib.use('Qt5Agg')

# Relation between intensity and size.
def g(v):
    return np.power(v, 2)

def g_1(v):
    return 2.0 * v

def F_1(v, v_p, Sigma_p, u, g, Sigma_u, g_1):
    return - (v - v_p) / Sigma_p + (u - g) / Sigma_u * g_1

# Data from environment.
u = 2.0
Sigma_u = 1.0
v_p = 3.0
Sigma_p = 1.0

# Auxiliary data.
Delta_t = 0.01
t_min = 0.0
t_max = 5.0
n_t = int((t_max - t_min) / Delta_t) + 1
np_t = np.empty(n_t)
np_phi = np.empty(n_t)

# Run gradient ascent method.
phi = v_p
for i in range(0, n_t):
    phi = phi + F_1(phi, v_p, Sigma_p, u, g(phi), Sigma_u, g_1(phi)) * Delta_t
    np_t[i] = i
    np_phi[i] = phi
print(f"phi = {phi}")

# Plot time evolution.
plt.plot(np_t, np_phi)
plt.show()
