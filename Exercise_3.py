import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import Exercise_1 as ex1
import Exercise_2 as ex2

def phi_1(phi, epsilon_p, epsilon_u):
    return epsilon_u * ex2.g_1(phi) - epsilon_p

def epsilon_p_1(phi, epsilon_p):
    return phi - ex1.v_p - ex1.Sigma_p * epsilon_p

def epsilon_u_1(phi, epsilon_u):
    return ex1.u - ex1.g(phi) - ex1.Sigma_u * epsilon_u

def main():
    # Auxiliary data.
    n_t = int(ex2.t_max / ex2.Delta_t) + 1
    t = np.empty(n_t)
    phi = np.empty(n_t)
    epsilon_p = np.empty(n_t)
    epsilon_u = np.empty(n_t)

    # Simulate neural model.
    t[0] = 0.0
    phi[0] = ex1.v_p
    epsilon_p[0] = 0.0
    epsilon_u[0] = 0.0
    for i in range(1, n_t):
        t[i] = i * ex2.Delta_t
        phi[i] = phi[i - 1] + phi_1(phi[i - 1], epsilon_p[i - 1], epsilon_u[i - 1]) * ex2.Delta_t
        epsilon_p[i] = epsilon_p[i - 1] + epsilon_p_1(phi[i - 1], epsilon_p[i - 1]) * ex2.Delta_t
        epsilon_u[i] = epsilon_u[i - 1] + epsilon_u_1(phi[i - 1], epsilon_u[i - 1]) * ex2.Delta_t
    print(f"phi = {phi[-1]}")
    print(f"epsilon_p = {epsilon_p[-1]}")
    print(f"epsilon_u = {epsilon_u[-1]}")

    # Plot time evolution.
    matplotlib.use('Qt5Agg')
    plt.plot(t, phi)
    plt.plot(t, epsilon_p)
    plt.plot(t, epsilon_u)
    plt.show()

if __name__ == "__main__":
    main()
