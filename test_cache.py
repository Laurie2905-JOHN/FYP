# Import libs
from dash import Dash, dcc, Output, Input, ctx, State, dash_table
from dash.dash import no_update
from dash.exceptions import PreventUpdate
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import sys
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import warnings
import base64
import io
import os
import math
import diskcache
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import dash
from dash import html, dash_table
from dash.long_callback import DiskcacheLongCallbackManager
from dash_extensions.callback import CallbackCache
from flask_caching.backends import FileSystemCache
from dash_extensions.callback import CallbackCache, Trigger

cc = CallbackCache()

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Ignore warning of square root of negative number
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def calculate_turbulence_intensity(u, v, w):

    N = len(u)

    # Calculate mean velocities
    U = np.mean(u)
    V = np.mean(v)
    W = np.mean(w)

    # Calculate velocity fluctuations
    u_prime = u - U
    v_prime = v - V
    w_prime = w - W

    # Calculate mean squared velocity fluctuations
    mean_u_prime_sq = np.mean(np.square(u_prime))
    mean_v_prime_sq = np.mean(np.square(v_prime))
    mean_w_prime_sq = np.mean(np.square(w_prime))

    # Calculate RMS of velocity fluctuations
    u_prime_RMS = np.sqrt(mean_u_prime_sq)
    v_prime_RMS = np.sqrt(mean_v_prime_sq)
    w_prime_RMS = np.sqrt(mean_w_prime_sq)

    # Calculate magnitude of mean flow velocity
    U_mag = np.sqrt(U**2 + V**2 + W**2)

    # Calculate turbulence intensity
    TI = (np.sqrt(u_prime_RMS**2 + v_prime_RMS**2 + w_prime_RMS**2)) / U_mag

    return TI, U_mag, U, V, W

def cal_velocity(BarnFilePath, cal_data, SF):

    ## function to calculate velocities from Barnacle voltage data

    # Import libs
    import base64
    import numpy as np
    import scipy.io as sio
    from scipy import interpolate
    import statistics as st

    # Constants
    rho = 997

    Dynfit = cal_data['Dynfit']
    Yawfit = cal_data['Yawfit']
    LDyn1 = cal_data['Ldyn1']
    LYaw1 = cal_data['Lyaw1']
    LDyn2 = cal_data['Ldyn2']
    LYaw2 = cal_data['Lyaw2']
    LDyn_0 = cal_data['Ldyn0'][0]

    # Importing Zeroes
    zeros = {}
    # Raw data which is zero reading data from slack water
    # Taking average of zero readings for each transducer
    zeros['pr_mean'] = [st.mean(cal_data['Zero']),st.mean(cal_data['Zero1']),st.mean(cal_data['Zero2']),st.mean(cal_data['Zero3']),st.mean(cal_data['Zero4'])]

    # Evaluating yawcal for a polynomial Cal.Yawfit and dyncal
    yawcal = np.zeros((91, 2))
    yawcal[:, 0] = np.linspace(-45, 45, 91)
    yawcal[:, 1] = np.polyval(Yawfit, yawcal[:, 0])
    dyncal = np.polyval(Dynfit, yawcal[:, 0])
    dyncal = dyncal * LDyn_0

    prb = {}

    prb['raw'] = np.loadtxt(BarnFilePath, delimiter=',')
    # Calculating velocities
    prb['raw'] -= zeros['pr_mean']
    # Data analysis
    # Calculating the mean of each row of the angled probes
    prb['denom'] = np.mean(prb['raw'][:, :4], axis=1)
    # Calculating Lyaw and Lpitch
    prb['Lyaw'] = (prb['raw'][:, 1] - prb['raw'][:, 3]) / prb['denom']
    prb['Lpitch'] = (prb['raw'][:, 0] - prb['raw'][:, 2]) / prb['denom']
    # Interpolating for each yaw and pitch angle
    ayaw_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
    apitch_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
    prb['ayaw'] = ayaw_interp(prb['Lyaw'])
    prb['apitch'] = apitch_interp(prb['Lpitch'])
    # Bodge: whatever one is bigger interpolate for Ldyn
    prb['pitchbigger'] = np.abs(prb['apitch']) > np.abs(prb['ayaw'])
    prb['amax'] = prb['pitchbigger'] * prb['apitch'] + (1 - prb['pitchbigger']) * prb['ayaw']
    ldyn_interp = interpolate.interp1d(yawcal[:, 0], dyncal, kind='linear', fill_value='extrapolate')
    prb['ldyn'] = ldyn_interp(prb['amax'])

    # Splitting into velocities
    prb['U1'] = np.sqrt(2 * -prb['ldyn'] * np.mean(prb['raw'][:, :4], axis=1) / rho)
    prb['U1'][np.imag(prb['U1']) > 0] = 0
    prb['Ux'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.cos(np.deg2rad(prb['ayaw']))
    prb['Uy'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.sin(np.deg2rad(prb['ayaw']))
    prb['Uz'] = prb['U1'] * np.sin(np.deg2rad(prb['apitch']))
    prb['t'] = np.linspace(0, prb['raw'].shape[0] / SF, prb['raw'].shape[0]);

    prb_final = {
        'U1': prb['U1'],
        'Ux': prb['Ux'],
        'Uy': prb['Uy'],
        'Uz': prb['Uz'],
        't': prb['t'],
    }

    return prb_final


# Create the Dash app object
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks=True)


app.layout = dbc.Container([

# Graph row
dbc.Row([
    dbc.Col(
        dcc.Graph(id='Velocity_Graph', figure={}),
        width=12),
]),
