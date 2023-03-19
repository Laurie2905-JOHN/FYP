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
Dynfit = Cal[0][0][0]
Yawfit = Cal[0][0][1]
LDyn = Cal[0][0][2]
LYaw = Cal[0][0][3]
LDyn_0 = Cal[0][0][4]



# Evaluating yawcal for a polynomial Cal.Yawfit and dyncal
yawcal = np.zeros((91, 2))
yawcal[:, 0] = np.linspace(-45, 45, 91)


yawcal[:, 1] = np.polyval(Yawfit, yawcal[:, 0])
dyncal = np.polyval(Dynfit, yawcal[:, 0])

# Importing Zeroes
zeros = {}
zeros['pr_raw'] = np.loadtxt(ZeroFolder + ZeroFile)
zeros['pr_mean'] = np.mean(zeros['pr_raw'][1300:1708, :], axis=0)

# Loading actual Barnacle data
prb = {}
prb['raw'] = np.loadtxt(BarnFolder + BarnFile)
prb['raw'] -= zeros['pr_mean']

# Data analysis
prb['denom'] = np.mean(prb['raw'][:, :4], axis=1)
prb['Lyaw'] = (prb['raw'][:, 1] - prb['raw'][:, 3]) / prb['denom']
prb['Lpitch'] = (prb['raw'][:, 0] - prb['raw'][:, 2]) / prb['denom']
prb['ayaw'] = np.interp(prb['Lyaw'], yawcal[:, 1], yawcal[:, 0], left=None, right=None, period=None)
prb['apitch'] = np.interp(prb['Lpitch'], yawcal[:, 1], yawcal[:, 0], left=None, right=None, period=None)
prb['pitchbigger'] = np.abs(prb['apitch']) > np.abs(prb['ayaw'])
prb['amax'] = prb['pitchbigger'] * prb['apitch'] + (1 - prb['pitchbigger']) * prb['ayaw']
prb['ldyn'] = np.interp(prb['amax'], yawcal[:, 0], dyncal, left=None, right=None, period=None)

# Splitting into velocities
prb['U1'] = np.sqrt(2 * -prb['ldyn'] * np.mean(prb['raw'][:, :4], axis=1) / rho)
prb['U1'][np.imag(prb['U1']) > 0] = 0
prb['Ux'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.cos(np.deg2rad(prb['ayaw']))
prb['Uy'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.sin(np.deg2rad(prb['ayaw']))


