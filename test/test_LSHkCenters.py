import numpy as np
from LSHkCenters.LSHkCenters import LSHkCenters

X = np.array([[0,0,0],[0,1,1],[0,0,0],[1,0,1],[2,2,2],[2,3,2],[2,3,2]])
y = np.array([0,0,0,0,1,1,1])

kcens = LSHkCenters(X,y,n_init=5,k=2)
kcens.SetupLSH()
kcens.DoCluster()
kcens.CalcScore()
kcens.CalcFuzzyScore()