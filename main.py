FilePath = 'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/'
ZeroFolder = FilePath
ZeroFile = 'Mon1527.txt'
BarnFolder = ZeroFolder
BarnFile = 'Mon1501.txt'
CalFolder = ZeroFolder
CalFile = 'IanYawAndDynCalMk2.mat'
VectorFolder = ZeroFolder
VectFile = 'MinMon20210607150154'

from scipy.io import loadmat
Cal1 = loadmat(FilePath+CalFile)
Cal2=Cal1["Cal"]

Dynfit = (Cal2[0][0][0])
Yawfit = (Cal2[0][0][1])
LDyn = (Cal2[0][0][2])
LYaw = (Cal2[0][0][3])
LDyn_0 = (Cal2[0][0][4])

import numpy as np

from numpy import linspace
yawcal=linspace(-45,45,91)


# Define a polynomial with coefficients [2, 3, 1]
p = np.array([2, 3, 1])

# Evaluate the polynomial at x = 2
x = 2
y = np.polyval(p, x)

print(y) # Output: 15


import numpy as np






