import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import Exercise_1 as ex1

def g_1(v):
    '''First derivative of g.'''
    return 2.0 * v

def F_1(v, v_p, Sigma_p, u, g, Sigma_u, g_1):
    '''First derivative of free energy.'''
    return - (v - v_p) / Sigma_p + (u - g) / Sigma_u * g_1

# Time simulation data.
Delta_t = 0.01
t_min = 0.0
t_max = 5.0

def main():
    # Auxiliary data.
    n_t = int((t_max - t_min) / Delta_t) + 1
    np_t = np.empty(n_t)
    np_phi = np.empty(n_t)

    # Run gradient ascent method.
    phi = ex1.v_p
    for i in range(0, n_t):
        phi = phi + F_1(phi, ex1.v_p, ex1.Sigma_p, ex1.u, ex1.g(phi), ex1.Sigma_u, g_1(phi)) * Delta_t
        np_t[i] = i
        np_phi[i] = phi
    print(f"phi = {phi}")

    # Plot time evolution.
    matplotlib.use('Qt5Agg')
    plt.plot(np_t, np_phi)
    plt.show()

if __name__ == "__main__":
    main()
