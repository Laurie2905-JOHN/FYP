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

# Loading vector data
vect_raw = np.loadtxt(VectorFolder + VectFile + '.dat')
vect_start = vect_raw[:, 1]
vect_Ux = -vect_raw[:, 3]
vect_Uy = vect_raw[:, 4]
vect_Uz = vect_raw[:, 5]

# Vector time data
vect_t = vect_start / fs + np.linspace(0, len(vect_raw)-1/fs, len(vect_raw))

vect_U1 = np.sqrt(vect_Ux**2 + vect_Uy**2 + vect_Uz**2)
vect_apitch = np.arcsin(vect_Uz / vect_U1)
vect_ayaw = np.arctan(vect_Uy / vect_Ux)

# cut data so that each file is the same length
if vect_t[-1] > prb.t[-1]:
    vect_Ux = vect_Ux[vect_t <= prb.t[-1]]
    vect_Uy = vect_Uy[vect_t <= prb.t[-1]]
    vect_Uz = vect_Uz[vect_t <= prb.t[-1]]
    vect_ayaw = vect_ayaw[vect_t <= prb.t[-1]]
    vect_apitch = vect_apitch[vect_t <= prb.t[-1]]
    vect_t = vect_t[vect_t <= prb.t[-1]]
elif prb.t[-1] > vect_t[-1]:
    prb.Ux = prb.Ux[prb.t <= vect_t[-1]]
    prb.Uy = prb.Uy[prb.t <= vect_t[-1]]
    prb.Uz = prb.Uz[prb.t <= vect_t[-1]]
    prb.ayaw = prb.ayaw[prb.t <= vect_t[-1]]
    prb.apitch = prb.apitch[prb.t <= vect_t[-1]]
    prb.t = prb.t[prb.t <= vect_t[-1]]

prb.Ux = prb.Ux[prb.t >= vect_t[0]]
prb.Uy = prb.Uy[prb.t >= vect_t[0]]
prb.Uz = prb.Uz[prb.t >= vect_t[0]]
prb.ayaw = prb.ayaw[prb.t >= vect_t[0]]
prb.apitch = prb.apitch[prb.t >= vect_t[0]]
prb.t = prb.t[prb.t >= vect_t[0]]

if vect_start < 0:
    vect_Ux = vect_Ux[vect_t >= prb.t[0]]
    vect_Uy = vect_Uy[vect_t >= prb.t[0]]
    vect_Uz = vect_Uz[vect_t >= prb.t[0]]
    vect_t = vect_t[vect_t >= prb.t[0]]

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(prb.t, prb.Ux, 'k')
axs[0, 0].plot(vect_t, vect_Ux, 'r')
axs[0, 1].plot(prb.t, prb.Uy, 'k')
axs[0, 1].plot(vect_t, vect_Uy, 'r')
axs[1, 0].plot(prb.t, prb.Uz, 'k')

