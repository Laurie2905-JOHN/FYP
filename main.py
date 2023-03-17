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
Cal=(Cal1["Cal"])
print(Cal)

import numpy as np

print(len(Cal))

from numpy import linspace
yawcal=linspace(-45,45,91)



