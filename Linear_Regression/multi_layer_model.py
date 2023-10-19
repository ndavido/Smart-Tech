#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed(0)
n_pts = 500

X, y = datasets.make_circles(
    n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()
