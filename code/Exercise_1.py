import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import utilities as ut

def g(v):
    '''Relation between intensity and size.'''
    return np.power(v, 2)

# Data from environment.
u = 2.0
Sigma_u = 1.0
v_p = 3.0
Sigma_p = 1.0

def main():
    # Data for inference.
    v_min = 0.01
    v_max = 5.0
    num = 10000
    v_step = (v_max - v_min) / num
    v = np.linspace(0.01, 5.0, num=10000)

    # Build needed probability distributions.
    prior = ut.gaussian(v, v_p, Sigma_p)
    likelihood = ut.gaussian(u, g(v), Sigma_u)
    evidence = np.sum(prior * likelihood) * v_step

    # Get posterior probability.
    posterior = prior * likelihood / evidence

    # Analysis of data.
    i_max_prior = np.argmax(prior)
    i_max_likelihood = np.argmax(likelihood)
    i_max_posterior = np.argmax(posterior)
    print(f"mode(p(v|u)) = {v[i_max_posterior]}")

    # Plot posterior.
    matplotlib.use('Qt5Agg')
    plt.plot(v, posterior)
    plt.axvline(v[i_max_posterior])
    plt.show()

if __name__ == '__main__':
    main()
