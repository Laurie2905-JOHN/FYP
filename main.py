import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

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

# Linear extrapolation of endpoints
#slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
#y_extrap_left = y[0] - slope * (x[0] - x[1])
#slope = (y[1] - y[0]) / (x[1] - x[0])
#y_extrap_right = y[-1] + slope * (x[-1] - x[-2])

ayaw_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind = 'linear' ,fill_value = 'extrapolate')
apitch_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind = 'linear' ,fill_value = 'extrapolate')

# Large disparity between MATLAB in ayaw

prb['ayaw'] = ayaw_interp(prb['Lyaw'])
prb['apitch'] = apitch_interp(prb['Lpitch'])


#plt.scatter(yawcal[:, 1], yawcal[:, 0])
#plt.scatter(prb['Lyaw'], prb['ayaw'])
#plt.show()


#plt.scatter(yawcal[:, 1], yawcal[:, 0])
#plt.scatter(prb['Lpitch'], prb['apitch'])
#plt.show()


prb['pitchbigger'] = np.abs(prb['apitch']) > np.abs(prb['ayaw'])
prb['amax'] = prb['pitchbigger'] * prb['apitch'] + (1 - prb['pitchbigger']) * prb['ayaw']

interp1 = interpolate.interp1d(yawcal[:, 0], dyncal, kind = 'linear' ,fill_value = 'extrapolate')
prb['ldyn'] = interp1(prb['amax'])

#plt.scatter(yawcal[:, 0], dyncal)
#plt.scatter(prb['amax'], prb['ldyn'] )
#plt.show()

# Splitting into velocities
prb['U1'] = np.sqrt(2 * -prb['ldyn'] * np.mean(prb['raw'][:, :4], axis=1) / rho)
prb['U1'][np.imag(prb['U1']) > 0] = 0



prb['Ux'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.cos(np.deg2rad(prb['ayaw']))
prb['Uy'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.sin(np.deg2rad(prb['ayaw']))
prb['Uz'] = prb['U1'] * np.sin(np.deg2rad(prb['apitch']))

prb['t'] = np.linspace(0,prb['raw'].shape[0]/fs,prb['raw'].shape[0]);





# Loading vector data
vect = {}
vect['raw'] = np.loadtxt(VectorFolder + VectFile + '.dat')
vect['start'] = vect['raw'][:, 1]
vect['Ux'] = -vect['raw'][:, 2]
vect['Uy'] = vect['raw'][:, 3]
vect['Uz'] = vect['raw'][:, 4]


# Vector time data
vect['t'] = (vect['start'] / fs) + np.linspace(0, (len(vect['raw'])-1)/fs, len(vect['raw']))

vect['U1'] = np.sqrt(vect['Ux']**2 + vect['Uy']**2 + vect['Uz']**2)
vect['apitch']  = np.arcsin(vect['Uz'] / vect['U1'])
vect['ayaw']  = np.arctan(vect['Uy'] / vect['Ux'])



# cut data so that each file is the same length
if vect['t'][-1] > prb['t'][-1]:
    vect['Ux'] = vect['Ux'] [vect['t'] <= prb['t'][-1]]
    vect['Uy'] = vect['Uy'][vect['t'] <= prb['t'][-1]]
    vect['Uz'] = vect['Uz'][vect['t'] <= prb['t'][-1]]
    vect['ayaw'] = vect['ayaw'][vect['t'] <= prb['t'][-1]]
    vect['apitch'] = vect['apitch'][vect['t'] <= prb['t'][-1]]
    vect['t'] = vect['t'][vect['t'] <= prb['t'][-1]]

elif prb['t'][-1] > vect['t'][-1]:
    prb['Ux'] = prb['Ux'][prb['t'] <= vect['t'][-1]]
    prb['Uy'] = prb['Uz'][prb['t'] <= vect['t'][-1]]
    prb['Uz'] = prb['Uz'][prb['t']<= vect['t'][-1]]
    prb['ayaw'] = prb['ayaw'][prb['t'] <= vect['t'][-1]]
    prb['apitch'] = prb['apitch'][prb['t'] <= vect['t'][-1]]
    prb['t'] = prb['t'][prb['t'] <= vect['t'][-1]]

prb['Ux'] = prb['Ux'][prb['t'] >= vect['t'][0]]
prb['Uy'] = prb['Uy'][prb['t'] >= vect['t'][0]]
prb['Uz'] = prb['Uz'][prb['t'] >= vect['t'][0]]
prb['ayaw'] = prb['ayaw'][prb['t'] >= vect['t'][0]]
prb['apitch'] = prb['apitch'][prb['t'] >= vect['t'][0]]
prb['t'] = prb['t'][prb['t']>= vect['t'][0]]


#if vect['start'] < 0:
    #vect['Ux'] = vect['Ux'][vect['t'] >= prb['t'][0]]
    #vect['Uy'] = vect['Uy'][vect['t'] >= prb['t'][0]]
   # vect['Uz'] = vect['Uz'][vect['t'] >= prb['t'][0]]
    #vect['t'] = vect['t'][vect['t'] >= prb['t'][0]]




fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(prb['t'], prb['Ux'], 'k')
axs[0, 0].plot(vect['t'],  vect['Ux'] , 'r')

axs[0, 1].plot(prb['t'], prb['Uy'], 'k')
axs[0, 1].plot(vect['t'], vect['Uy'], 'r')

axs[1, 0].plot(prb['t'], prb['Uz'], 'k')
axs[1, 0].plot(vect['t'], vect['Uz'], 'r')

print(len(vect['Ux']))
print(len(prb['Ux']))

plt.show()