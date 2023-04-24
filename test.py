# Import libs
from dash import Dash, dcc, Output, Input, ctx, State, dash_table
from dash.dash import no_update
from dash.exceptions import PreventUpdate
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
import sys
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import warnings
import base64
import io
import math
from dash.dependencies import Output, Input
from flask_caching.backends import FileSystemCache
from dash_extensions.callback import CallbackCache, Trigger
from dash_extensions.callback import CallbackCache
# Ignore warning of square root of negative number

def cal_velocity(contents, file_names):

    ## function to calculate velocities from Barnacle voltage data

    # Import libs
    import base64
    import numpy as np
    import scipy.io as sio
    from scipy import interpolate

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
    # Raw data which is zero reading data from slack water
    zeros['pr_raw'] = np.loadtxt(ZeroFolder + ZeroFile, delimiter=',')
    # Taking average of zero readings for each transducer
    zeros['pr_mean'] = np.mean(zeros['pr_raw'][1300:1708, :], axis=0) # 1300-1708 was genuinely slack water

    # Loading actual Barnacle data
    # Decoding Barnacle data
    content_string = contents
    decoded = base64.b64decode(content_string)
    decoded_str = decoded.removeprefix(b'u\xabZ\xb5\xecm\xfe\x99Z\x8av\xda\xb1\xee\xb8')
    lines = decoded_str.decode().split('\r\n')[:-1]

    # Assigning dictionaries
    prb = {}
    prb_final = {}
    # Calculating velocities
    # For loop allows calculation for multiple files if needed
    for i, file_name in enumerate(file_names):
        prb[file_name] = {'raw': {}}
        # Assigning numpy array to dictionary
        prb[file_name]['raw'] = np.array([list(map(float, line.split(','))) for line in lines])
        # Subtracting zero readings from the data
        prb[file_name]['raw'] -= zeros['pr_mean']
        # Data analysis
        # Calculating the mean of each row of the angled probes
        prb[file_name]['denom'] = np.mean(prb[file_name]['raw'][:, :4], axis=1)
        # Calculating Lyaw and Lpitch
        prb[file_name]['Lyaw'] = (prb[file_name]['raw'][:, 1] - prb[file_name]['raw'][:, 3]) / prb[file_name]['denom']
        prb[file_name]['Lpitch'] = (prb[file_name]['raw'][:, 0] - prb[file_name]['raw'][:, 2]) / prb[file_name]['denom']
        # Interpolating for each yaw and pitch angle
        ayaw_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
        apitch_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
        prb[file_name]['ayaw'] = ayaw_interp(prb[file_name]['Lyaw'])
        prb[file_name]['apitch'] = apitch_interp(prb[file_name]['Lpitch'])
        # Bodge: whatever one is bigger interpolate for Ldyn
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

        # Taking data needed
        prb_final = {'Ux': {}}
        prb_final = {'Uy': {}}
        prb_final = {'Uz': {}}
        prb_final = {'t': {}}

        prb_final['Ux'] = prb[file_name]['Ux']
        prb_final['Uy'] = prb[file_name]['Uy']
        prb_final['Uz'] = prb[file_name]['Uz']
        prb_final['t'] = prb[file_name]['t']

    return prb_final

# Create app.
app = Dash(prevent_initial_callbacks=True)
app.layout = html.Div([
    dcc.Graph('graph',figure = {}),
    dcc.Upload(
        id='submit_files',
        children=html.Div([
            html.A('Select BARNACLE Files')
        ]),
        style={
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '15px',
            'textAlign': 'center',
            'width': '100%',
        },
        className="text-primary",
        # Allow multiple files to be uploaded
        multiple=True,
        max_size=-1,
    ),
    dcc.Loading(dcc.Store(id="store"), fullscreen=True, type="dot")
])
# Create (server side) cache. Works with any flask caching backend.
cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))



@cc.cached_callback(Output("store", "data"), [Input("submit_files", "contents"), Input("submit_files",
                                                                                       "filename")])  # Trigger is like Input, but excluded from args
def query_data(contents, file_names):
    print(contents)
    data = cal_velocity(contents[0], file_names)
    return data



@cc.callback(Output("graph", "figure"), [Input("store", "data")])
def update_graph(df):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['t'], y=df['Ux'], mode='lines'))

    return fig


# This call registers the callbacks on the application.
cc.register(app)

# Run app
if __name__== '__main__':
    app.run_server(debug=True)