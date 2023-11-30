import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import functions as fn

matplotlib.use('Qt5Agg')

# Relation between intensity and size.
def g(v):
    return np.power(v, 2)

# Data from environment.
u = 2.0
Sigma_u = 1.0
v_p = 3.0
Sigma_p = 1.0

# Data for inference.
v_min = 0.01
v_max = 5.0
num = 10000
v_step = (v_max - v_min) / num
v = np.linspace(0.01, 5.0, num=10000)

# Build needed probability distributions.
prior = fn.gaussian(v, v_p, Sigma_p)
likelihood = fn.gaussian(u, g(v), Sigma_u)
evidence = np.sum(prior * likelihood) * v_step

# Get posterior probability.
posterior = prior * likelihood / evidence

# Plot data.
plt.plot(v, posterior)
plt.show()
