import numpy as np

PATH_FIGURES = "figures/"

def gaussian(x, mu, Sigma):
    return np.exp(- np.power(x - mu, 2) / (2.0 * Sigma)) / np.sqrt(2.0 * np.pi * Sigma)
