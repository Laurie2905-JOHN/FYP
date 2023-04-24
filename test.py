from pathlib import Path
import uuid
import dash_bootstrap_components as dbc
import dash_uploader as du
import dash
from dash import html, dash_table
from dash.dependencies import Input, Output, State

def cal_velocity(file_name, BarnFilePath):

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

    # Barnacle Data
    prb = {}
    prb['raw'] = np.loadtxt(BarnFilePath, delimiter=',')

    # Assigning dictionaries

    prb_final = {}

    # Calculating velocities
    # For loop allows calculation for multiple files if needed
    # Assigning numpy array to dictionary
    # Subtracting zero readings from the data
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
    prb['t'] = np.linspace(0, prb['raw'].shape[0] / fs, prb['raw'].shape[0]);

    # Taking data needed
    prb_final = {'Ux': {}}
    prb_final = {'Uy': {}}
    prb_final = {'Uz': {}}
    prb_final = {'t': {}}

    prb_final['Ux'] = prb['Ux']
    prb_final['Uy'] = prb['Uy']
    prb_final['Uz'] = prb['Uz']
    prb_final['t'] = prb['t']

    return prb_final


app = dash.Dash(__name__)

UPLOAD_FOLDER_ROOT = r'C:\Users\lauri\Desktop'
du.configure_upload(
    app,
    str(UPLOAD_FOLDER_ROOT),
    use_upload_id=True,
)

def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=30000,  # 1800 Mb
        filetypes=['txt', 'csv'],
        upload_id=uuid.uuid1(),  # Unique session id
    )


def get_app_layout():

    return html.Div(
        [

            html.H1('Demo'),
            html.Div(
                [
                    get_upload_component(id='dash-uploader'),
                    html.Div(id='callback-output'),
                ],
                style={  # wrapper div style
                    'textAlign': 'center',
                    'width': '600px',
                    'padding': '10px',
                    'display': 'inline-block'
                }),
        ],
        style={
            'textAlign': 'center',
        },
    )


@du.callback(
    output=Output("callback-output", "children"),
    id="dash-uploader",
)
def callback_on_completion(status):
    BarnFilePath = [str(x) for x in status.uploaded_files]
    prb = cal_velocity('test', BarnFilePath[0])
    print(prb)
    return 'done'

# get_app_layout is a function
# This way we can use unique session id's as upload_id's
app.layout = get_app_layout






if __name__ == '__main__':
    app.run_server(debug=True)

