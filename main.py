import numpy as np
import scipy.io as sio


# Constants
rho = 997
fs = 16  # sample rate

# File retrieving
logfolder = "C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/"
ZeroFolder = logfolder
ZeroFile = 'Mon1527.txt'
BarnFolder = ZeroFolder
BarnFile = 'Mon1501.txt'
CalFolder = ZeroFolder
CalFile = 'IanYawAndDynCalMk2.mat'
VectorFolder = ZeroFolder
VectFile = 'MinMon20210607150154'
Cal = sio.loadmat(CalFolder + CalFile)

Cal = Cal["Cal"]
Dynfit = Cal[0][0][0].flatten()
Yawfit = Cal[0][0][1].flatten()
LDyn = Cal[0][0][2].flatten()
LYaw = Cal[0][0][3].flatten()
LDyn_0 = Cal[0][0][4].flatten()

# Evaluating yawcal for a polynomial Cal.Yawfit and dyncal
yawcal = np.zeros((91, 2))
yawcal[:, 0] = np.linspace(-45, 45, 91)
yawcal[:, 1] = np.polyval(Yawfit,yawcal[:, 0])
dyncal = np.polyval(Dynfit, yawcal[:, 0])


# Importing Zeroes
zeros = {}
zeros['pr_raw'] = np.loadtxt(ZeroFolder + ZeroFile, delimiter=',')
zeros['pr_mean'] = np.mean(zeros['pr_raw'][1300:1708, :], axis=0)

# Loading actual Barnacle data
prb = {}
prb['raw'] = np.loadtxt(BarnFolder + BarnFile, delimiter=',')
prb['raw'] -= zeros['pr_mean']


# Data analysis
prb['denom'] = np.mean(prb['raw'][:, :4], axis = 1)
prb['Lyaw'] = (prb['raw'][:, 1] - prb['raw'][:, 3]) / prb['denom']
prb['Lpitch'] = (prb['raw'][:, 0] - prb['raw'][:, 2]) / prb['denom']

from scipy import interpolate

interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind = 'linear' ,fill_value = 'extrapolate')

# Large disparity between MATLAB in ayaw
prb['ayaw'] = interp(prb['Lyaw'])
prb['apitch'] = interp(prb['Lpitch'])

# Disparity caused by ayaw
prb['pitchbigger'] = np.abs(prb['apitch']) > np.abs(prb['ayaw'])
prb['amax'] = prb['pitchbigger'] * prb['apitch'] + (1 - prb['pitchbigger']) * prb['ayaw']
interp1 = interpolate.interp1d(yawcal[:, 0], dyncal, kind = 'linear' ,fill_value = 'extrapolate')
prb['ldyn'] = interp1(prb['amax'])



