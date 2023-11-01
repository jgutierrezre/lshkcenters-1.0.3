import numpy as np
import pandas as pd

import sys

from LSHkCenters.LSHkCenters import LSHkCenters

np.set_printoptions(threshold=sys.maxsize)

X = np.array(
    [[0, 0, 0], [0, 1, 1], [0, 0, 0], [1, 0, 1], [2, 2, 2], [2, 3, 2], [2, 3, 2]]
)
y = np.array([0, 0, 0, 0, 1, 1, 1])

# X = np.array(
#     [[0, 0], [0, 1], [1, 2]]
# )
# y = np.array([0, 0, 1])


# data = pd.read_csv("new/datasets/soybean-small.data.csv", header=None)
# data = data.apply(lambda col: pd.factorize(col)[0])

# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values
# y = np.unique(y, return_inverse=True)[1]


kcens = LSHkCenters(X, y, n_init=5, k=2)
kcens.SetupLSH()
print(kcens.DoCluster())