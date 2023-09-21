Python implementations of the LSH-k-Centers algorithms for clustering categorical data:

## Installation:
### Using pip: 
```shell
pip install lshkcenters
```

### Import the packages:
```shell
import numpy as np
from LSHkCenters.LSHkCenters import LSHkCenters
```
### Generate a simple categorical dataset:

```shell
X = np.array([[0,0,0],[0,1,1],[0,0,0],[1,0,1],[2,2,2],[2,3,2],[2,3,2]])
y = np.array([0,0,0,0,1,1,1])
```

### LSH-k-Centers: 

```shell
kcens = LSHkCenters(X,y,n_init=5,k=2)
kcens.SetupLSH()
kcens.DoCluster()

```

### Built-in evaluattion metrics:
```shell
kcens.CalcScore()
```

### Out come:
```shell
Purity: 1.000 NMI: 1.00 ARI: 1.00 Sil:  -1.00 Acc: 1.00 Recall: 1.00 Precision: 1.00
```

### Built-in fuzzy evaluattion metrics:
```shell
kcens.CalcFuzzyScore()
```

### Out come:
```shell
Fuzzy scores PC:1.00 NPC:1.00 FHV↓:0.02 FS↓:-2000.00 XB↓:0.11 BH↓:0.06 BWS:-2000.00 FPC:3.50 SIL_R:0.70 FSIL:0.70 MPO:12.15 NPE:0.01 PE:0.01 PEB:0.01
```


## References:
*To be updated*
