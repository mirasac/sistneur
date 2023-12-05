import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import Exercise_2 as ex2

v_mean = 5.0
v_variance = 2.0
t_max = 20.0
alpha = 0.01
n_trials = 1000

def main():
    # Auxiliary data.
    rng = np.random.default_rng()
    trial = np.empty(n_trials + 1)
    sigma = np.empty(n_trials + 1)
    g = v_mean
    n_t = int(t_max / ex2.Delta_t) + 1

    # Initial conditions for trials simulation.
    trial[0] = 0
    sigma[0] = 1.0
    # Trials simulation.
    for i_trial in range(1, n_trials + 1):
        trial[i_trial] = i_trial
        # Initial conditions for time simulation.
        phi = rng.normal(v_mean, math.sqrt(v_variance))
        epsilon = 0.0
        e = 0.0
        # Time simulation.
        for i_t in range(1, n_t):
            epsilon_prev = epsilon
            e_prev = e
            epsilon = epsilon_prev + (phi - g - e_prev) * ex2.Delta_t
            e = e_prev + (sigma[i_trial - 1] * epsilon_prev - e_prev) * ex2.Delta_t
        sigma[i_trial] = sigma[i_trial - 1] + alpha * (epsilon * e - 1.0)

    # Plot time evolution.
    matplotlib.use('Qt5Agg')
    plt.plot(trial, sigma)
    plt.show()

if __name__ == "__main__":
    main()
