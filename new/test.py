import numpy as np
import pandas as pd

import sys

from myproject import LSHkCenters

np.set_printoptions(threshold=sys.maxsize)

# X = np.array(
#     [[0, 0, 0], [0, 1, 1], [0, 0, 0], [1, 0, 1], [2, 2, 2], [2, 3, 2], [2, 3, 2]]
# )
# y = np.array([0, 0, 0, 0, 1, 1, 1])

# X = np.array(
#     [[0, 0], [0, 1], [1, 2]]
# )
# y = np.array([0, 0, 1])

# 1. Soybean-small

data = pd.read_csv("new/datasets/soybean-small.data.csv", header=None)
data = data.apply(lambda col: pd.factorize(col)[0])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = np.unique(y, return_inverse=True)[1]

# 2. Balance Scale

# data = pd.read_csv("new/datasets/balance_scale/balance-scale.data", header=None)
# data = data.iloc[:624, :]
# data = data.apply(lambda col: pd.factorize(col)[0])

# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values
# y = np.unique(y, return_inverse=True)[1]

# 3. Zoo
# data = pd.read_csv("new/datasets/zoo/zoo.data", header=None)
# data = data.apply(lambda col: pd.factorize(col)[0])

# X = data.iloc[:, 1:].values
# y = data.iloc[:, 0].values
# y = np.unique(y, return_inverse=True)[1]


kcens = LSHkCenters(X, y)
print()
print(kcens.do_cluster())
