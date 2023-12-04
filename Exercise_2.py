import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import Exercise_1 as ex1

def g_1(v):
    '''First derivative of g.'''
    return 2.0 * v

def F_1(v):
    '''First derivative of free energy.'''
    return - (v - ex1.v_p) / ex1.Sigma_p + (ex1.u - ex1.g(v)) / ex1.Sigma_u * g_1(v)

# Time simulation data.
Delta_t = 0.01
t_max = 5.0

def main():
    # Auxiliary data.
    n_t = int(t_max / Delta_t) + 1
    t = np.empty(n_t)
    phi = np.empty(n_t)

    # Run gradient ascent method.
    t[0] = 0.0
    phi[0] = ex1.v_p
    for i in range(1, n_t):
        t[i] = i * Delta_t
        phi[i] = phi[i - 1] + F_1(phi[i - 1]) * Delta_t
    print(f"phi = {phi[-1]}")

    # Plot time evolution.
    matplotlib.use('Qt5Agg')
    plt.plot(t, phi)
    plt.show()

if __name__ == "__main__":
    main()
