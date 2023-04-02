import numpy as np
def cal_velocity(file_paths):
    import numpy as np
    import scipy.io as sio

    # Constants
    rho = 997
    fs = 16  # sample rate

    # File retrieving
    ZeroFolder = "C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/"
    ZeroFile = 'Mon1527.txt'
    CalFolder = ZeroFolder
    CalFile = 'IanYawAndDynCalMk2.mat'
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
    yawcal[:, 1] = np.polyval(Yawfit, yawcal[:, 0])
    dyncal = np.polyval(Dynfit, yawcal[:, 0])
    dyncal = dyncal * LDyn_0

    # Importing Zeroes
    zeros = {}
    zeros['pr_raw'] = np.loadtxt(ZeroFolder + ZeroFile, delimiter=',')
    zeros['pr_mean'] = np.mean(zeros['pr_raw'][1300:1708, :], axis=0)

    # Loading actual Barnacle data
    prb = {}
    for i, file_path in enumerate(file_paths):
        file_name = file_path.split("/")[-1]
        prb[file_name] = {'raw': {}}
        prb[file_name]['raw'] = np.loadtxt(file_path, delimiter=',')
        prb[file_name]['raw'] -= zeros['pr_mean']
        # Data analysis
        prb[file_name]['denom'] = np.mean(prb[file_name]['raw'][:, :4], axis=1)
        prb[file_name]['Lyaw'] = (prb[file_name]['raw'][:, 1] - prb[file_name]['raw'][:, 3]) / prb[file_name]['denom']
        prb[file_name]['Lpitch'] = (prb[file_name]['raw'][:, 0] - prb[file_name]['raw'][:, 2]) / prb[file_name]['denom']

        from scipy import interpolate

        ayaw_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
        apitch_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
        prb[file_name]['ayaw'] = ayaw_interp(prb[file_name]['Lyaw'])
        prb[file_name]['apitch'] = apitch_interp(prb[file_name]['Lpitch'])
        prb[file_name]['pitchbigger'] = np.abs(prb[file_name]['apitch']) > np.abs(prb[file_name]['ayaw'])
        prb[file_name]['amax'] = prb[file_name]['pitchbigger'] * prb[file_name]['apitch'] + (1 - prb[file_name]['pitchbigger']) * prb[file_name]['ayaw']
        ldyn_interp = interpolate.interp1d(yawcal[:, 0], dyncal, kind='linear', fill_value='extrapolate')
        prb[file_name]['ldyn'] = ldyn_interp(prb[file_name]['amax'])

        # Splitting into velocities
        prb[file_name]['U1'] = np.sqrt(2 * -prb[file_name]['ldyn'] * np.mean(prb[file_name]['raw'][:, :4], axis=1) / rho)
        prb[file_name]['U1'][np.imag(prb[file_name]['U1']) > 0] = 0
        prb[file_name]['Ux'] = prb[file_name]['U1'] * np.cos(np.deg2rad(prb[file_name]['apitch'])) * np.cos(np.deg2rad(prb[file_name]['ayaw']))
        prb[file_name]['Uy'] = prb[file_name]['U1'] * np.cos(np.deg2rad(prb[file_name]['apitch'])) * np.sin(np.deg2rad(prb[file_name]['ayaw']))
        prb[file_name]['Uz'] = prb[file_name]['U1'] * np.sin(np.deg2rad(prb[file_name]['apitch']))
        prb[file_name]['t'] = np.linspace(0, prb[file_name]['raw'].shape[0] / fs, prb[file_name]['raw'].shape[0]);

    return prb
import time
import sys
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

start_time = time.time()

file_paths = [

    'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 1.txt',
    'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 2.txt']


prb = cal_velocity(file_paths)
vels = ['t']
file = 'Example 1.txt'
#df = {file: {vel: prb[file][vel] for vel in vels}}


smallt =10
bigt = 50

dff = {file: {vel: prb[file][vel] for vel in vels}}

df = {file: {vel: [] for vel in vels}}

if smallt != None or bigt != None:

    t = dff[file]['t']
    len(t)

    if smallt != None and bigt != None:
        mask = (t >= smallt) & (t < bigt)

    if smallt != None:
        mask = (t >= smallt)

    if bigt != None:
        mask = (t < bigt)

    df[file]['t'] = t[mask]
    for vel in vels:
        df[file][vel] = dff[file][vel][mask]

else:
    df = dff

list_all = []

if len(vels) == 1:
    stacked = df[file][vels[0]]
    list_all.append(stacked)

if len(vels) == 2:
    stacked = np.stack((df[file][vels[0]], df[file][vels[1]]), axis=1)
    list_all.append(stacked)

if len(vels) == 3:
    stacked1 = np.stack((df[file][vels[0]], df[file][vels[1]]), axis=1)
    stacked2 = df[file][vels[2]].reshape(-1, 1)
    stacked = np.concatenate((stacked1, stacked2), axis=1)
    list_all.append(stacked)

k = 0
if len(vels) == 4:
    while k < len(vels) - 1:
        stacked = np.stack((df[file][vels[k]], df[file][vels[k + 1]]), axis=1)
        list_all.append(stacked)
        k = k + 2

if len(vels) == 1:
    list_all = list_all[0]
    str_all = np.array2string(list_all, separator=',\n', threshold=sys.maxsize)

else:
    list_all = np.concatenate(list_all, axis=1)
    str_all = np.array2string(list_all, separator=',', threshold=sys.maxsize)

vels_str = ','.join(vels)

str_all = vels_str + '\n' + str_all

str_all = str_all.replace(' ', '')

str_all = str_all.replace('],', '')

str_all = str_all.replace(']]', '')

str_all = str_all.replace('[[', '')

str_all = str_all.replace('[', '')

str_all = str_all.replace(']', '')

text = dict(content=str_all, filename="hello.txt")



print(str_all)







