
from dash import Dash, dcc, Output, Input  # pip install dash
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import plotly.express as px
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
dyncal = dyncal * LDyn_0

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

ayaw_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind = 'linear' ,fill_value = 'extrapolate')
apitch_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind = 'linear' ,fill_value = 'extrapolate')
prb['ayaw'] = ayaw_interp(prb['Lyaw'])
prb['apitch'] = apitch_interp(prb['Lpitch'])
prb['pitchbigger'] = np.abs(prb['apitch']) > np.abs(prb['ayaw'])
prb['amax'] = prb['pitchbigger'] * prb['apitch'] + (1 - prb['pitchbigger']) * prb['ayaw']
ldyn_interp = interpolate.interp1d(yawcal[:, 0],dyncal, kind = 'linear' ,fill_value = 'extrapolate')
prb['ldyn'] = ldyn_interp(prb['amax'])


# Splitting into velocities
prb['U1'] = np.sqrt(2 * -prb['ldyn'] * np.mean(prb['raw'][:, :4], axis=1) / rho )
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
vect['Uy'] = -vect['raw'][:, 3]
vect['Uz'] = vect['raw'][:, 4]


# Vector time data
vect['start'] = 795
vect['t'] = (vect['start'] / fs) + np.linspace(0, (len(vect['raw'])-1)/fs, len(vect['raw']))
vect['U1'] = np.sqrt(vect['Ux']**2 + vect['Uy']**2 + vect['Uz']**2)
vect['apitch']  = np.arcsin(vect['Uz'] / vect['U1'])
vect['ayaw']  = np.arctan(vect['Uy'] / vect['Ux'])


if vect['t'][-1] > prb['t'][-1]:
    mask = vect['t'] >= prb['t'][-1]
    vect['Ux'] = np.delete(vect['Ux'], np.where(mask))
    vect['Uy'] = np.delete(vect['Uy'], np.where(mask))
    vect['Uz'] = np.delete(vect['Uz'], np.where(mask))
    vect['ayaw'] = np.delete(vect['ayaw'], np.where(mask))
    vect['apitch'] = np.delete(vect['apitch'], np.where(mask))
    vect['t'] = np.delete(vect['t'], np.where(mask))

elif prb['t'][-1] > vect['t'][-1]:
    mask = prb['t'] >= vect['t'][-1]
    prb['Ux'] = np.delete(prb['Ux'], np.where(mask))
    prb['Uy'] = np.delete(prb['Uy'], np.where(mask))
    prb['Uz'] = np.delete(prb['Uz'], np.where(mask))
    prb['ayaw'] = np.delete(prb['ayaw'], np.where(mask))
    prb['apitch'] = np.delete(prb['apitch'], np.where(mask))
    prb['t'] = np.delete(prb['t'], np.where(mask))

mask = prb['t'] <= vect['t'][0]
prb['Ux'] = np.delete(prb['Ux'], np.where(mask))
prb['Uy'] = np.delete(prb['Uy'], np.where(mask))
prb['Uz'] = np.delete(prb['Uz'], np.where(mask))
prb['ayaw'] = np.delete(prb['ayaw'], np.where(mask))
prb['apitch'] = np.delete(prb['apitch'], np.where(mask))
prb['t'] = np.delete(prb['t'], np.where(mask))

mask = vect['t'] <= prb['t'][0]
vect['Ux'] = np.delete(vect['Ux'], np.where(mask))
vect['Uy'] = np.delete(vect['Uy'], np.where(mask))
vect['Uz'] = np.delete(vect['Uz'], np.where(mask))
vect['t'] = np.delete(vect['t'], np.where(mask))

# Building Interface

# Build your components
app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])
mytitle = dcc.Markdown(children='# App that analyzes Olympic medals')
mygraph = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options=['Ux', 'Uy', 'Uz'],
                        value='Ux',  # initial value displayed when page first loads
                        clearable=False)

# Customize your own Layout
app.layout = dbc.Container([mytitle, mygraph, dropdown])

# Callback allows components to interact
@app.callback(
    Output(mygraph, component_property='figure'),
    Input(dropdown, component_property='value')
)


def update_graph(user_input):  # function arguments come from the component property of the Input

    if user_input == 'Ux':
        fig = px.line(x=prb['t'], y=prb['Ux'])

    elif user_input == 'Uy':
        fig = px.line(x=prb['t'], y=prb['Uy'])

    elif user_input == 'Uz':
        fig = px.line(x=prb['t'], y=prb['Uz'])

    return fig  # returned objects are assigned to the component property of the Output

# Run app
if __name__=='__main__':
    app.run_server(port=8053)
