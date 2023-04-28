# Import libs
from dash import dash, dcc, Output, Input, ctx, State, dash_table, html
import dash_bootstrap_components as dbc
from dash.dash import no_update
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import base64
import io
import os
import math
import re
import shutil

# Ignore warning of square root of negative number
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Define functions

# Function to return min and max values based on inpuuted conditions
def update_values(large_val, small_val):
    """
    Update large and small input values, ensuring large_val is not less than small_val.
    If both inputs are None, prevent update.
    """

    if large_val < small_val:
        large_val = small_val

    return large_val, small_val

# Function to calculate the turbulence intensity
def calculate_turbulence_intensity(u, v, w):
    """
    Function to calculate turbulence intensity from three velocity components.

    Args:
        u (numpy.ndarray): Array containing velocity in x-direction.
        v (numpy.ndarray): Array containing velocity in y-direction.
        w (numpy.ndarray): Array containing velocity in z-direction.

    Returns:
        TI (float): Turbulence intensity.
        U_mag (float): Magnitude of mean flow velocity.
        U (float): Mean velocity in x-direction.
        V (float): Mean velocity in y-direction.
        W (float): Mean velocity in z-direction.
    """

    # Required imports
    import numpy as np

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

# Function to calculate velocity data
def cal_velocity(BarnFilePath, cal_data, SF):
    """
    Function to calculate velocities from Barnacle voltage data.

    Args:
        BarnFilePath (str): Path to the file containing Barnacle data.
        cal_data (dict): Dictionary containing calibration data.
        SF (int/float): Sample frequency used for data acquisition.

    Returns:
        prb_final (dict): Dictionary containing calculated velocities U1, Ux, Uy, Uz, and time array t.
    """

    # Required imports
    import numpy as np
    from scipy import interpolate
    import statistics as st

    # Constants
    rho = 997

    # Calibration data
    Dynfit = cal_data['Dynfit']
    Yawfit = cal_data['Yawfit']
    LDyn1 = cal_data['Ldyn1']
    LYaw1 = cal_data['Lyaw1']
    LDyn2 = cal_data['Ldyn2']
    LYaw2 = cal_data['Lyaw2']
    LDyn_0 = cal_data['Ldyn0'][0]

    # Importing Zeroes
    zeros = {}
    # Average zero readings for each transducer
    zeros['pr_mean'] = [st.mean(cal_data['Zero']), st.mean(cal_data['Zero1']), st.mean(cal_data['Zero2']),
                        st.mean(cal_data['Zero3']), st.mean(cal_data['Zero4'])]

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

# Define the layout of the app
app.layout = dbc.Container([

    # Header row with title
    dbc.Row(
        html.Div(className='mb-4')
    ),

    # Row for title
    dbc.Row([
        dbc.Col(
            html.H1("BARNACLE DATA ANALYSIS",
                    className='text-center fw-bold mb-1'),
            width=12),
    ]),

    # Graph row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='Velocity_Graph'),
        ],
            width=12),
    ]),

    # Options row
    dbc.Row([
        # Row for horizontal rule
        dbc.Row([
            dbc.Col(
                html.Hr(),
                width=12
            ),  # Horizontal rule to separate content
        ], className='mb-2'),

        # Graph options header
        dbc.Col(
            html.H5('GRAPH OPTIONS', className="text-center fw-bold"),
            width=12),

        # Row for horizontal rule
        dbc.Row(
            dbc.Col(
                html.Hr(),
                width=12
            ),  # Horizontal rule to separate content
            className='mb-4'),

        # Alert box
        dbc.Col([
            dbc.Alert(
                id="Graph_alert",
                is_open=False,
                dismissable=True,
                duration=20000,
                className='text-center'),
        ], width=12),
    ], align='center', justify='center'),

    # Row for controls and options
    dbc.Row([

        # Column for plot buttons
        dbc.Col(
            dbc.Stack([
                # Plot button
                dbc.Button("PLOT", id="plot_bttn", size="lg", color="primary", class_name='fw-bold'),
                # Clear figure button
                dbc.Button("CLEAR FIGURE", id="plot_clear_bttn", size="lg", color="primary", class_name='fw-bold'),
            ], gap=3),
            width=3
        ),

        # Column for dropdown menus and input fields
        dbc.Col(
            dbc.Stack([
                # Row for dropdown menus
                dbc.Row([
                    # Dropdown menu for selecting file
                    dbc.Col(
                        dcc.Dropdown(
                            id="File",
                            options=[],
                            multi=True,
                            value=[],
                            placeholder="Select a Dataset"
                        )
                    ),
                    # Dropdown menu for selecting quantity
                    dbc.Col(
                        dcc.Dropdown(
                            id="Vect",
                            options=[],
                            multi=True,
                            value=[],
                            placeholder="Select a Quantity"
                        )
                    ),
                ], align='center', justify='center'),

                # Row for input fields for minimum and maximum times
                dbc.Row([
                    # Input field for minimum time
                    dbc.Col(
                        dbc.Input(id="time_small", min=0, type="number", placeholder="Min Time")
                    ),
                    # Input field for maximum time
                    dbc.Col(
                        dbc.Input(id="time_large", min=0, type="number", placeholder="Max Time")
                    ),
                ], align='center', justify='center'),

            ], gap=3),
            width=4
        ),

        # Column for radio buttons and input fields
        dbc.Col(
            dbc.Stack([
                # Row for radio buttons to control title and legend display
                dbc.Row([
                    dbc.Col(
                        dbc.Stack([
                            dbc.Label("TITLE", className="text-start, fw-bold"),
                            dbc.RadioItems(id='title_onoff', value='On', options=['On', 'Off'], inline=True)
                        ])
                    ),
                    dbc.Col(
                        dbc.Stack([
                            dbc.Label('LEGEND', className="text-start, fw-bold"),
                            dbc.RadioItems(id='legend_onoff', value='On', options=['On', 'Off'], inline=True),
                        ]),
                        className='mb-3'
                    ),
                ]),
                # Row for input fields to update title and legend
                dbc.Row(
                    dbc.Stack([
                        dbc.InputGroup([
                            # Dropdown menu to choose what to update
                            dbc.DropdownMenu([
                                dbc.DropdownMenuItem("Update Title", id="dropdown_title_update"),
                                dbc.DropdownMenuItem("Update Legend", id="dropdown_legend_update"),
                                dbc.DropdownMenuItem(divider=True),
                                dbc.DropdownMenuItem("Clear", id="dropdown_clear"),
                            ], label="Update"),
                            # Input field for new title/legend
                            dbc.Input(
                                id="New_name",
                                type='text',
                                placeholder="Enter text to update legend or title",
                            ),
                        ]),
                    ], direction="horizontal"),
                ),
            ]),
            width=5
        ),
    ], className='mb-3', justify="center", align="center"),

    dbc.Row([
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ], className='mb-2'),

    dbc.Row(
        dbc.Col(
            html.H5('TURBULENCE PARAMETERS', className='center-text, fw-bold'),
            width=12,
            className="text-center"
        ),  # Column containing the header for the download files section
    ),

    dbc.Row(
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
        className='mb-4'
    ),

    dbc.Row(
        dbc.Col([
            dbc.Alert(
                id="TI_alert",
                is_open=False,
                dismissable=True,
                duration=30000,
                className='text-center',
            ),  # Alert component to show status of file download
        ], width=12),
    ),

    dbc.Col([

        # Row containing the buttons and dropdown menu
        dbc.Row([

            # Column for buttons
            dbc.Col(

                dbc.Stack([

                    # Button for downloading data
                    dbc.Button("CALCULATE", id="TI_btn_download", size="lg", color="primary", className='fw-bold'),

                    # Button for clearing the table
                    dbc.Button("CLEAR TABLE", id="Clear_Table", size="lg", color="primary", className='fw-bold'),

                ], gap=3),

                width=3),

            # Column for dropdown menu and input fields
            dbc.Col(

                dbc.Stack([

                    # Dropdown menu for selecting dataset
                    dcc.Dropdown(
                        id="DataSet_TI",
                        options=[],
                        multi=False,
                        placeholder="Select a Dataset"
                    ),

                    # Input field for minimum time
                    dbc.Input(id="small_t_TI", type="number", placeholder="Min Time"),

                    # Input field for maximum time
                    dbc.Input(id="big_t_TI", type="number", placeholder="Max Time"),

                ], gap=3),

                width=3),

        ], align='center', justify='center'),

    ], width=12),

    # Column for displaying the table
    dbc.Col([

        dash_table.DataTable(
            id='TI_Table',
            columns=[
                {"id": 'FileName', "name": 'File Name'},
                {'id': 'CalFile', 'name': 'Cal File'},
                {'id': 'SF', 'name': 'SF'},
                {"id": 'Time_1', "name": 'Time 1'},
                {"id": 'Time_2', "name": 'Time 2'},
                {"id": 'Ux', "name": 'Average Ux'},
                {"id": 'Uy', "name": 'Average Uy'},
                {"id": 'Uz', "name": 'Average Uz'},
                {"id": 'U1', "name": 'Average U'},
                {"id": 'TI', 'name': 'Turbulence Intensity'},
            ],
            export_format='xlsx',
            export_headers='display',
            row_deletable=True,
        ),

        html.Div(className='mb-4'),

    ], width=12),

    dbc.Row([
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ], className='mb-2'),

    dbc.Row([

        # Column for "Upload/Clear Files" title
        dbc.Col(
            html.H5('WORKSPACE', className='center-text, fw-bold'),
            width=12,
            className="text-center"
        ),

        dbc.Row(
            dbc.Col(
                html.Hr(),
                width=12
            ),  # Horizontal rule to separate content
        ),

        dbc.Col([
            dbc.Alert(
                id="Workspace_alert_temp",
                is_open=False,
                class_name='text-center',
                dismissable=True,
                duration=30000,
            ),
        ], class_name='mb-3', width=10),

        # Column containing alert message for workspace (hidden by default)
        dbc.Col([
            dbc.Alert(
                id="Workspace_alert",
                is_open=False,
                class_name='text-center'
            ),
        ], class_name='mb-3', width=10),

        # Column containing input group for workspace update/clear
        dbc.Col(
            dbc.InputGroup([
                # Dropdown menu for selecting the update action
                dbc.DropdownMenu([
                    dbc.DropdownMenuItem("Update Workspace", id="Workspace_update"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Clear Workspace", id="Workspace_clear"),
                ],
                    label="UPDATE"),
                # Input element for entering the new title or legend
                dbc.Input(
                    id="Workspace",
                    type='text',
                    placeholder="Enter Workspace Filepath",
                ),

            ]),
            width=11,
            class_name='mb-3'
        ),

        # Horizontal rule to separate content
        dbc.Row([
            dbc.Col(
                html.Hr(),
                width=12
            ),
        ], className='mb-2'),

        # Column for "Upload/Clear Files" title
        dbc.Col(
            html.H5('FILE UPLOAD', className='center-text, fw-bold'),
            width=12,
            className="text-center"
        ),

        # Column for alert message (hidden by default)
        dbc.Col([
            dbc.Alert(
                id="ClearFiles_alert",
                is_open=False,
                dismissable=True,
                duration=30000,
                className='text-center',
            ),
        ], width=12),

        # Horizontal line
        dbc.Row(
            dbc.Col(
                html.Hr(),
                width=12
            ),  # Horizontal rule to separate content
            className='mb-4'
        ),

        # Column for BARNACLE file selection/upload
        dbc.Col(
            dbc.InputGroup([
                # Dropdown menu for selecting the update action
                dbc.DropdownMenu([
                    dbc.DropdownMenuItem("Add to Uploads", id="dropdown_BARN_update"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Clear Uploads", id="dropdown_BARN_clear"),
                ],
                    label="UPDATE"),
                # Input element for entering the new title or legend
                dbc.Input(
                    id="submit_files",
                    type='text',
                    placeholder="Enter BARNACLE Filepath",
                ),
            ]),
            width=11,
            class_name='mb-5'
        ),

        # Column for file selection/upload
        dbc.Col([
            dbc.Stack([
                dcc.Upload(
                    id='submit_Cal_file',
                    children=dbc.Button(
                        'CLICK TO SELECT A CALIBRATION FILE',
                        className="primary fw-bold"
                    ),
                    # Allow multiple files to be uploaded
                    multiple=False,
                ),
                dbc.Alert(
                    id='calAlert',
                    dismissable=False,
                    class_name='text-center'
                ),
            ], gap=3),
        ], width=3),

        # Column for uploading files
        dbc.Col(
            dbc.Stack([
                dbc.Button(
                    'PROCESS SELECTED FILES',
                    id='newfile',
                    color="primary",
                    className="me-1, fw-bold",
                    n_clicks=0,
                ),
                dbc.Input(
                    id="Sample_rate",
                    min=0,
                    type="number",
                    placeholder="Enter Sample Frequency",
                ),
                html.Label(
                    "SELECT FILES TO PROCESS",
                    className='center-text, fw-bold'
                ),

                # Checkbox for uploading all files
                dbc.Checklist(
                    ["All"], [], id="all_upload_file_checklist", inline=True
                ),

                # Checkbox for selecting individual files to upload
                dbc.Checklist(value=[], id="upload_file_checklist", inline=True),
            ],

                gap=3),
            width=3),

        # Column for clearing files
        dbc.Col(
            dbc.Stack([
                dbc.Button(
                    "CLEAR SELECTED FILES FROM WORKSPACE",
                    id='clear_files',
                    color="primary",
                    className="me-1, fw-bold",
                    n_clicks=0
                ),

                html.Label("SELECT FILES TO CLEAR", className='center-text, fw-bold'),

                # Checkbox for clearing all files
                dbc.Checklist(
                    ["All"], [], id="all_clear_file_checklist", inline=True
                ),

                # Checkbox for selecting individual files to clear
                dbc.Checklist(value=[], id="clear_file_checklist", inline=True),
            ],
                gap=3),
            width=3
        ),

    ], align='start', justify='evenly', className = 'mb-4'),

    dbc.Row([
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ], className='mb-2'),

    dbc.Row(
        dbc.Col(
            html.H5('DOWNLOAD DATA', className='center-text, fw-bold'),
            width=12,
            className="text-center"
        ),  # Column containing the header for the download files section
    ),

    dbc.Row(
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
        className='mb-4'
    ),

    dbc.Row(
        dbc.Col([
            dbc.Alert(
                id="Download_alert",
                is_open=False,
                dismissable=True,
                duration=30000,
                className='text-center',
            ),  # Alert component to show status of file download
        ], width=12),
    ),

dbc.Row([

    dbc.Col(

        dbc.Stack([
            # Label for selecting data file
            html.Label("CHOOSE DATASET", className ='fw-bold'),

            # Radio buttons for selecting data file
            dbc.RadioItems(id="file_checklist", inline=True),

            # Label for selecting quantity of data to download
            html.Label("CHOOSE QUANTITY", className ='fw-bold'),

            # Checkbox to select all data
            dbc.Checklist(["All"], [], id="all_vel_checklist", inline=True),

            # Checkbox to select specific data
            dbc.Checklist(value=[], options=[], id="vel_checklist", inline=True),

        ], gap=3),

    width = 4),

    dbc.Col(

        dbc.Stack([
            # Button for downloading selected data
            dbc.Button("DOWNLOAD", class_name = 'fw-bold', id="btn_download", size="lg",color="primary"),

            # Input field for file name
            dbc.Input(id="file_name_input", type="text", placeholder="Enter Filename"),

            # Row for input fields for minimum and maximum times
            dbc.Row([
                # Input field for minimum time
                dbc.Col(
                    dbc.Input(id="small_t", type="number", placeholder="Min Time")
                ),

                # Input field for maximum time
                dbc.Col(
                    dbc.Input(id="big_t", min=0, type="number", placeholder="Max Time")
                ),

            ], justify="center"),

        ], gap=3),

    width = 4),


    ], align='center', justify='center',className = 'mb-5'),

    # # Components for storing and downloading data
    dbc.Spinner(children = [dcc.Store(id='Loading_variable_Process', storage_type='memory')],color="primary",
                fullscreen = True, size = 'lg', show_initially = False, delay_hide = 80, delay_show = 80),

    # # Components for storing and downloading data
    dbc.Spinner(children=[dcc.Store(id='Loading_variable_Table', storage_type='memory')], color="primary",
                fullscreen=True, size='lg', show_initially=False, delay_hide=80, delay_show=80),

    # # Components for storing and downloading data
    dbc.Spinner(children=[dcc.Store(id='Loading_variable_Download', storage_type='memory')], color="primary",
                fullscreen=True, size='lg', show_initially=False, delay_hide=80, delay_show=80),

    # # Components for storing and downloading data
    dbc.Spinner(children=[dcc.Store(id='Loading_variable_Graph', storage_type='memory')], color="primary",
                fullscreen=True, size='lg', show_initially=False, delay_hide=80, delay_show=80),

    # Store Components
    dcc.Store(id='legend_Data', storage_type='memory'),
    dcc.Store(id='title_Data', storage_type='memory'),
    dcc.Store(id='filestorage', storage_type='session'),
    dcc.Store(id='filename_filepath', storage_type='session'),
    dcc.Store(id='Workspace_store', storage_type='local'),
    dcc.Store(id='Cal_storage', storage_type='local'),
])

# Callback 1
# Call back to update the workspace filepath.
# This callback function is triggered by a click event on the 'Workspace_update' button.
@app.callback(
    Output(component_id='Workspace_store', component_property='data'),
    Output(component_id='Workspace_alert_temp', component_property='children', allow_duplicate=True),
    Output(component_id='Workspace_alert_temp', component_property='color', allow_duplicate=True),
    Output(component_id='Workspace_alert_temp', component_property='is_open', allow_duplicate=True),
    Output(component_id='Workspace_alert', component_property='children', allow_duplicate=True),
    Output(component_id='Workspace_alert', component_property='color', allow_duplicate=True),
    Output(component_id='Workspace_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='Workspace_update', component_property='n_clicks'),
    State(component_id='Workspace', component_property='value'),
)
def update_Workspace(n_clicks, Workspace_input):

    # Try/Except, used to catch any errors not considered
    try:

        # Check if the trigger is from the 'Workspace_update' button.
        if ctx.triggered_id == 'Workspace_update':

            # Handle the case when no input is provided.
            if Workspace_input is None or Workspace_input == '':
                error_temp = 'NO FILE PATH INPUTTED'
                color_temp = 'danger'
                error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                color_perm = 'danger'
                Workspace_data = no_update

            else:
                # Normalize and validate the file path.
                Workspace_input2 = Workspace_input.replace('"', "")
                Workspace_input3 = os.path.normpath(Workspace_input2)

                # Check if the given path exists.
                if not os.path.exists(Workspace_input3):
                    error_temp = 'PLEASE CHECK FILE PATH IS VALID'
                    color_temp = 'danger'
                    error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                    color_perm = 'danger'
                    Workspace_data = no_update
                else:
                    # If the path exists, update the workspace.
                    error_temp = 'WORKSPACE UPDATED'
                    color_temp = 'success'
                    error_perm = 'CURRENT WORKSPACE: ' + Workspace_input3
                    color_perm = 'primary'
                    Workspace_data = Workspace_input3

        return Workspace_data, error_temp, color_temp, True, error_perm, color_perm, True

    except Exception as e:

        # If error there is an error print it.
        error_temp = str(e)
        color_temp = 'danger'
        error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
        color_perm = 'danger'

    return no_update, error_temp, color_temp, True, error_perm, color_perm, True

# Callback 2
# Callback function for clearing the workspace store.
@app.callback(
    Output(component_id='Workspace', component_property='value', allow_duplicate=True),
    Output(component_id='Workspace_store', component_property='clear_data', allow_duplicate=True),
    Output(component_id='filestorage', component_property='clear_data', allow_duplicate=True),
    Output(component_id='filename_filepath', component_property='clear_data', allow_duplicate=True),
    Output(component_id='Workspace_alert_temp', component_property='children', allow_duplicate=True),
    Output(component_id='Workspace_alert_temp', component_property='color', allow_duplicate=True),
    Output(component_id='Workspace_alert_temp', component_property='is_open', allow_duplicate=True),
    Output(component_id='Workspace_alert', component_property='children', allow_duplicate=True),
    Output(component_id='Workspace_alert', component_property='color', allow_duplicate=True),
    Output(component_id='Workspace_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='Workspace_clear', component_property='n_clicks'),
    State(component_id='Workspace_store', component_property='data'),
    prevent_initial_call=True
)
def clear_Workspace(n_clicks, Workspace_data):

    try:
        # If the callback was triggered by the 'Workspace_clear' button
        if ctx.triggered_id == 'Workspace_clear':
            # If there's no workspace data to clear
            if Workspace_data is None:
                error_temp = 'NO WORKSPACE TO CLEAR'
                color_temp = 'danger'
                error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                color_perm = 'danger'
                Workspace_input = None
                Workspace_Clear_data = False
                Upload_Clear_data = False
                filedata_Clear_data = False
            else:

                deleted_files = []
                error_files = []

                if not os.path.exists(Workspace_data):
                    error_temp = 'WORKSPACE NO LONGER EXISTS'
                    color_temp = 'danger'
                    error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                    color_perm = 'danger'
                    Workspace_input = None
                    Workspace_Clear_data = True
                    Upload_Clear_data = True
                    filedata_Clear_data = True

                else:

                    # Iterate through files in the workspace directory and delete them
                    for file_name in os.listdir(Workspace_data):
                        path = os.path.join(Workspace_data, file_name)
                        try:
                            if os.path.isfile(path):
                                # Delete the file
                                os.remove(path)
                                deleted_files.append(file_name)
                            elif os.path.isdir(path):
                                # Delete the folder and its contents recursively
                                shutil.rmtree(path)
                                deleted_files.append(file_name)
                        except Exception as e:
                            error_files.append(file_name)

                    # Update output values
                    Workspace_Clear_data = True
                    Upload_Clear_data = True
                    filedata_Clear_data = True
                    Workspace_input = None

                    # Prepare the success/error message
                    if deleted_files != [] and error_files == []:
                        error_temp = 'WORKSPACE DATA CLEARED. '  + ', '.join(deleted_files) + ' REMOVED.'
                        color_temp = 'success'
                    elif deleted_files == [] and error_files != []:
                        error_temp = 'WORKSPACE CLEARED. ' + 'NO FILES REMOVED. ' + 'ERROR DELETING: ' + ', '.join(error_files)
                        color_temp = 'danger'
                    elif error_files != [] and deleted_files != []:
                        error_temp = 'WORKSPACE CLEARED. ' + ', '.join(deleted_files) + ' REMOVED. ' + 'ERROR DELETING: ' +\
                                ', '.join(error_files)
                        color_temp = 'danger'
                    elif error_files == [] and deleted_files == []:
                        error_temp = 'WORKSPACE CLEARED. ' + 'NO FILES REMOVED.'
                        color_temp = 'danger'

                    error_perm = 'NO WORKSPACE SELECTED'
                    color_perm = 'danger'

            # Return updated values for UI components
            return Workspace_input, Workspace_Clear_data, Upload_Clear_data, filedata_Clear_data, error_temp,\
                color_temp, True, error_perm, color_perm, True

    except Exception as e:
        error_temp = str(e)
        color_temp = 'danger'
        error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
        color_perm = 'danger'
        # Return no_update for output components when an exception occurs
        return no_update, no_update, no_update, no_update, error_temp, color_temp, True, error_perm, color_perm, True

# Callback 3
# Callback to update workspace alert based on data
@app.callback(
    Output(component_id='Workspace_alert', component_property='children'),
    Output(component_id='Workspace_alert', component_property='color'),
    Output(component_id='Workspace_alert', component_property='is_open'),
    Input(component_id="Workspace_store", component_property='data'),
)
def update_Workspace_Alert(Workspace_data):

    try:

        if Workspace_data is None:
            alert_work = 'NO WORKSPACE SELECTED'
            color1 = 'danger'
        else:
            alert_work = 'WORKSPACE: ' + Workspace_data
            color1 = 'primary'

        return alert_work, color1, True
    except Exception as e:
        return str(e), 'danger', True

# Callback 4
# Define callback function for updating alert text based on calibration file data.
@app.callback(
    Output(component_id='calAlert', component_property='children'),
    Output(component_id='calAlert', component_property='color'),
    Output(component_id='calAlert', component_property='is_open'),
    Input(component_id="Cal_storage", component_property='data'),
)
def update_cal_text(Cal_data):

    try:
        # If there's no calibration data provided
        if Cal_data is None:
            alert_cal = 'NO CALIBRATION FILE SELECTED'
            color = 'danger'
        else:

            # Create a success message with the name of the selected calibration file
            alert_cal = Cal_data[0] + ' Selected'
            color = 'primary'

        # Return updated values for the alert component
        return alert_cal, color, True

    except Exception as e:
        # If an exception occurs, display the error message
        alert_cal = str(e)
        color = 'danger'
        return alert_cal, color, True

# Callback 5
# Define callback function for processing the uploaded calibration file
@ app.callback(
    Output(component_id="Cal_storage", component_property='data', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='submit_Cal_file', component_property='filename'),
    Input(component_id='submit_Cal_file', component_property='contents'),
        prevent_initial_call=True)

def cal_analysis(filename, contents):

    try:

        if filename == '' or contents == '':
            # If data is none or empty prevent error
            error = 'PLEASE TRY AGAIN AND CHECK THE FILE'
            color = 'danger'
            cal_data = no_update
        else:
            # Separate the content type and actual data from the contents
            content_type, content_string = contents.split(',')

            # Decode the base64 encoded content string
            decoded = base64.b64decode(content_string)
            # Check if the file is an Excel file

            if filename.split('.')[1] != 'xlsx':
                # If the file is not an Excel file, display an error message
                error = 'PLEASE UPLOAD A .xlsx FILE'
                color = 'danger'
                cal_data = no_update
            else:
                # Read the Excel file into a pandas DataFrame
                cal_data = pd.read_excel(io.BytesIO(decoded))

                # Convert the DataFrame to a dictionary and remove NaN values
                cal_data = cal_data.to_dict('list')

                if list(cal_data.keys()) == ['Dynfit', 'Yawfit', 'Ldyn1', 'Ldyn2', 'Lyaw1', 'Lyaw2','Ldyn0','Zero',\
                                             'Zero1', 'Zero2', 'Zero3', 'Zero4']:

                    cal_data = [filename, {key: [val for val in values if not math.isnan(val)] for key, values in
                                       cal_data.items()}]

                    # Prepare the success message
                    error = filename + ' UPLOADED SUCCESSFULLY'
                    color = 'success'

                else:

                    cal_data = no_update
                    # Prepare the success message
                    error = 'PLEASE TRY AGAIN AND CHECK THE FILE HAS THE CORRECT FORMAT'
                    color = 'success'

        # Return updated values for UI components
        return cal_data, error, color, True

    except Exception as e:
        # If an exception occurs, display the error message
        error = str(e)
        color = 'danger'
        return no_update, error, color, True

# Callback 6
# Define callback function for updating the file upload checklist data and verifying file format
@app.callback(
        Output(component_id='submit_files', component_property='value'),
        Output(component_id='filename_filepath', component_property='data'),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id = 'dropdown_BARN_update', component_property ='n_clicks'),
        State(component_id='submit_files', component_property='value'),
        State(component_id='filename_filepath', component_property='data'),
        prevent_initial_call = True)

def update_file_to_upload_checklist(n_clicks, filepath1, filename_filepath_data):

    try:
    # Check if the file upload update button is clicked
        if ctx.triggered_id == 'dropdown_BARN_update':
            # Verify if the file path input is empty or not
            if filepath1 is None or filepath1 == '':
                error = 'NO FILEPATH INPUTTED. PLEASE CHECK.'
                color = 'danger'
                filepath_input = None
                filename_filepath_data = no_update
            else:
                # Normalize and process the file path
                filepath2 = filepath1.replace('"', "")
                filepath = os.path.normpath(filepath2)
                filename1 = os.path.basename(filepath)
                filename = os.path.splitext(filename1)[0]

                # Check if the file exists and the format is .txt or .csv
                if not os.path.isfile(filepath):
                    error = 'PLEASE CHECK FILEPATH'
                    color = 'danger'
                    filepath_input = no_update
                    filename_filepath_data = no_update
                elif os.path.splitext(filename1)[1] not in ('.txt', '.csv'):
                    error = 'PLEASE UPLOAD .TXT OR .CSV FILES'
                    color = 'danger'
                    filepath_input = None
                    filename_filepath_data = no_update
                else:
                    # Update the file checklist data with the new file information
                    if filename_filepath_data is None:
                        filename_filepath_data = [[filename], [filepath]]
                        error = filename + ' ADDED'
                        color = 'success'
                        filepath_input = None
                    else:
                        combined_filenames = filename_filepath_data[0].copy()
                        combined_filepaths = filename_filepath_data[1].copy()
                        repeated_filename = []

                        for value in combined_filenames:
                            if filename == value:
                                repeated_filename.append(filename)

                        if repeated_filename != []:
                            error = filename + ' ALREADY EXISTS. PLEASE CHECK.'
                            color = 'danger'
                            filepath_input = no_update
                            filename_filepath_data = no_update
                        else:
                            combined_filenames.append(filename)
                            combined_filepaths.append(filepath)
                            error = filename + ' ADDED'
                            color = 'success'
                            filepath_input = None
                            filename_filepath_data = [combined_filenames, combined_filepaths]
            return filepath_input, filename_filepath_data, error, color, True

    except Exception as e:
        # If any error occurs, display the error message
        error = str(e)
        color = 'danger'
        return no_update, no_update, error, color, True

# Callback 7
# Define callback function for clearing the file upload checklist data
@app.callback(
    Output(component_id='submit_files', component_property='value', allow_duplicate=True),
    Output(component_id='filename_filepath', component_property='clear_data', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='dropdown_BARN_clear', component_property='n_clicks'),
    prevent_initial_call=True
)
def clear_upload(n_clicks):
    # Use try-except block to handle any unexpected errors
    try:
        # Check if the clear file upload button is clicked
        if ctx.triggered_id == 'dropdown_BARN_clear':
            # Clear the file upload checklist and display a success message
            filepath_input = None
            error = 'UPLOAD FILES CLEARED'
            color = 'primary'
            clear_filename_filepath_data = True

        return filepath_input, clear_filename_filepath_data, error, color, True

    except Exception as e:
        # If any error occurs, display the error message
        error = str(e)
        color = 'danger'
        return no_update, no_update, error, color, True

# Callback 7
# Callback to analyse data from uploaded files
@app.callback(
    [
        Output(component_id='filestorage', component_property='data', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='filename_filepath', component_property='data', allow_duplicate=True),
        Output(component_id="upload_file_checklist", component_property='value', allow_duplicate=True),
        Output(component_id='Loading_variable_Process', component_property='data', allow_duplicate=True),
        Output(component_id='Workspace_store', component_property='clear_data', allow_duplicate=True),
        Output(component_id='filestorage', component_property='clear_data', allow_duplicate=True),
        Output(component_id='filename_filepath', component_property='clear_data', allow_duplicate=True)
    ],
    [
        Input(component_id='newfile', component_property='n_clicks'),
        State(component_id='filename_filepath', component_property='data'),
        State(component_id="Cal_storage", component_property='data'),
        State(component_id='Sample_rate', component_property='value'),
        State(component_id='filestorage', component_property='data'),
        State(component_id="upload_file_checklist", component_property='value'),
        State(component_id="Workspace_store", component_property='data')
    ],
)
def Analyse_content(n_clicks, filename_filepath_data, cal_data, SF, file_data, filenames, Workspace_data):

    # Handle errors and exceptions
    try:
        # Check if the "newfile" button was clicked
        if "newfile" == ctx.triggered_id:
            upload_file_checklist = []

            # Check if no workspace was selected
            if Workspace_data is None:
                error_temp = 'UPDATE WORKSPACE'
                color_temp = 'danger'
                error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                color_perm = 'danger'
                open_perm = True
                filename_filepath_data = no_update
                # Return the same data if no files were uploaded
                file_data = no_update
                error_perm = no_update
                color_perm = no_update
                loading_variable = no_update
                Workspace_store_clear = True
                filestorage_clear = True
                filename_filepath_clear = True

            else:

                if os.path.exists(Workspace_data):

                    # Initialise data dictionary if it is None
                    if file_data is None:
                        file_data = [[], [], [], [], [], [], []]

                    # Check if no files were uploaded
                    if filenames is None or filenames == []:
                        error_temp = 'NO FILES SELECTED FOR UPLOAD'
                        color_temp = "danger"
                        filename_filepath_data = no_update
                        # Return the same data if no files were uploaded
                        file_data = no_update

                    # Check if no sample rate was selected
                    elif SF is None or SF == 0:
                        error_temp = 'NO SAMPLE RATE SELECTED'
                        color_temp = "danger"
                        filename_filepath_data = no_update
                        # Return the same data if no files were uploaded
                        file_data = no_update

                    else:
                        # Get existing data from file_data
                        Oldfilenames = file_data[0]
                        old_dtype_shape = file_data[1]
                        Old_calData = file_data[2]
                        Old_SF = file_data[3]
                        Old_filepath = file_data[4]
                        Old_min = file_data[5]
                        Old_max = file_data[6]

                        # Make copies of existing data
                        combined_filenames = Oldfilenames.copy()
                        combined_dtype_shape = old_dtype_shape.copy()
                        combined_CalData = Old_calData.copy()
                        combined_SF = Old_SF.copy()
                        combined_filepath = Old_filepath.copy()
                        combined_min = Old_min.copy()
                        combined_max = Old_max.copy()

                        # Initialise lists for processing files
                        new_value = []  # List of uploaded file names which aren't repeated
                        repeated_value = []  # List of repeated file names
                        error_file = []  # List of files with invalid formats

                        # Define function to save array as memmap
                        def save_array_memmap(array, filename, folder_path):
                            filepath = os.path.join(folder_path, filename)
                            dtype = str(array.dtype)
                            shape = array.shape
                            array_memmap = np.memmap(filepath, dtype=dtype, shape=shape, mode='w+')
                            array_memmap[:] = array[:]
                            del array_memmap
                            return shape, dtype

                        # Define function to get a unique path for each file
                        def get_unique_path(base_path, name):
                            counter = 1
                            new_name = name

                            while os.path.exists(os.path.join(base_path, new_name)):
                                new_name = f"{name} ({counter})"
                                counter += 1

                            return os.path.normpath(os.path.join(base_path, new_name))

                        # Loop through uploaded files and process them
                        for i, value in enumerate(filenames):
                            # Check if the file name is already in the combined list
                            if value not in combined_filenames:
                                try:
                                    # Process file data
                                    Barn_data = cal_velocity(filename_filepath_data[1][i], cal_data[1], SF)

                                    # Create workspace folder if it doesn't exist
                                    Workspace_Path = os.path.join(Workspace_data, 'Cached_Files')
                                    if not os.path.exists(Workspace_Path):
                                        os.mkdir(Workspace_Path)

                                    # Get unique file path for saving data
                                    file_path = get_unique_path(Workspace_Path, value)
                                    os.makedirs(file_path, exist_ok=True)

                                    # Update data with new processed file data
                                    combined_max.append(np.amax(Barn_data['t']))
                                    combined_min.append(np.amin(Barn_data['t']))

                                    save_array_memmap(Barn_data['Ux'], 'Ux.dat', file_path)
                                    save_array_memmap(Barn_data['Uy'], 'Uy.dat', file_path)
                                    save_array_memmap(Barn_data['Uz'], 'Uz.dat', file_path)
                                    save_array_memmap(Barn_data['U1'], 'U1.dat', file_path)
                                    shape_dtype = save_array_memmap(Barn_data['t'], 't.dat', file_path)

                                    new_value.append(value)
                                    combined_filenames.append(value)
                                    combined_dtype_shape.append(shape_dtype)
                                    combined_CalData.append(cal_data[0])
                                    combined_SF.append(SF)
                                    combined_filepath.append(file_path)

                                # If there's an error processing the file, add it to the error list
                                except Exception as e:
                                    error_file.append(value)
                            else:
                                repeated_value.append(value)

                        # Update file_data with combined data
                        file_data = [combined_filenames, combined_dtype_shape, combined_CalData, combined_SF, combined_filepath,
                                     combined_min, combined_max]

                        # Update uploaded file data
                        upload_filename = filename_filepath_data[0]
                        upload_filepath = filename_filepath_data[1]

                        # Delete selected data
                        for value in filenames:
                            i = upload_filename.index(value)
                            upload_filename.remove(value)
                            del upload_filepath[i]

                        filename_filepath_data = [upload_filename, upload_filepath]

                        # Display error messages if there are any errors
                        if repeated_value != [] or error_file != []:
                            error_list_complete = repeated_value + error_file

                            if new_value == []:
                                error_start = 'THERE WAS AN ERROR PROCESSING ALL FILES: \n ' \
                                              '(' + ', '.join(error_list_complete) + ').'
                            else:
                                error_start = 'THERE WAS AN ERROR PROCESSING FILES: \n ' \
                                              '(' + ', '.join(error_list_complete) + ').'

                            error_repeat = ' PLEASE CHECK FILES: ' \
                                           '(' + ', '.join(repeated_value) + ') ARE NOT REPEATED.'

                            error_process = ' PLEASE CHECK: \n' \
                                            '(' + ', '.join(error_file) + ') FOR ERRORS.'

                            # If there are errors in files and repeated files
                            if repeated_value != [] and error_file != []:
                                error_temp = error_start + error_repeat + error_process
                                color_temp = "danger"
                            elif error_file != [] and repeated_value == []:
                                error_temp = error_start + '\n' + error_process
                                color_temp = "danger"
                            elif error_file == [] and repeated_value != []:
                                error_temp = error_start + '\n' + error_repeat
                                color_temp = "danger"
                        else:
                            # If no errors display success message
                            error_temp = ', '.join(new_value) + ' PROCESSED'
                            color_temp = "success"

                        # Set loading variable to 'done' when processing is complete
                    loading_variable = 'done'
                    error_perm = no_update
                    color_perm = no_update
                    open_perm = no_update
                    Workspace_store_clear = False
                    filestorage_clear = False
                    filename_filepath_clear = False

                # If workspace doesnt exist display error messages
                else:
                    file_data = no_update
                    error_temp = 'ERROR IN FINDING WORKSPACE FOLDER'
                    color_temp = 'danger'
                    error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                    color_perm = 'danger'
                    open_perm = True
                    Workspace_store_clear = True
                    filestorage_clear = True
                    filename_filepath_clear = True
                    filename_filepath_data = no_update
                    loading_variable = no_update

        # Return output values
        return file_data, error_temp, color_temp, True, error_perm, color_perm, open_perm, filename_filepath_data,\
            upload_file_checklist, loading_variable, Workspace_store_clear, filestorage_clear, filename_filepath_clear

    except Exception as e:
        # If any error display message
        error_temp = str(e)
        color_temp = 'danger'
        error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
        color_perm = 'danger'
        upload_file_checklist = []
        Workspace_store_clear = True
        filestorage_clear = True
        filename_filepath_clear = True

    return no_update, error_temp, color_temp, True, error_perm, color_perm, True, no_update, upload_file_checklist,\
        no_update, Workspace_store_clear, filestorage_clear, filename_filepath_clear

# Callback 9
# Callback to clear data
@app.callback(
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='filestorage', component_property='data', allow_duplicate=True),
        Output(component_id="filestorage", component_property='clear_data', allow_duplicate=True),
        Output(component_id='filename_filepath', component_property='clear_data', allow_duplicate=True),
        Output(component_id="clear_file_checklist", component_property='value', allow_duplicate=True),
        Input(component_id='clear_files', component_property='n_clicks'),
        State(component_id='filestorage', component_property='data'),
        State(component_id="clear_file_checklist", component_property='value'),
        State(component_id='Workspace_store', component_property='data'),
        prevent_initial_call=True)

def clear_files(n_clicks, maindata, whatclear, Workspace_data):

    # # Try/Except, used to catch any errors not considered
    try:

        # If the clear files button is pressed, prevent update
        if "clear_files" == ctx.triggered_id:
            # If no workspace data
            if Workspace_data is None:
                error_temp = 'UPDATE WORKSPACE'
                color_temp = 'danger'
                error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                color_perm = 'danger'
                open_perm = True
                newmaindata = no_update
                clear_data_main = True
                clear_file_initial = True
                clear_file_checklist = []
            else:

                if not os.path.exists(Workspace_data):

                    error_temp = 'ERROR IN FINDING WORKSPACE FOLDER'
                    color_temp = 'danger'
                    error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                    color_perm = 'danger'
                    open_perm = True
                    newmaindata = no_update
                    clear_data_main = True
                    clear_file_initial = True
                    clear_file_checklist = []

                else:

                    error_perm = no_update
                    color_perm = no_update
                    open_perm = no_update

                    # If no files selected display error message
                    if len(whatclear) == 0:

                        clear_file_checklist = no_update

                        # display bad error message
                        error_temp = 'NO FILES DELETED'
                        color_temp = "danger"

                        # No update to new main data
                        newmaindata = no_update

                        clear_data_main = False

                        clear_file_initial = False

                    elif len(whatclear) > 0:

                        file_path = maindata[4]
                        deleted_files = []

                        try:

                            # Iterate through the selected files to be deleted
                            for what in whatclear:
                                i = maindata[0].index(what)
                                if os.path.isdir(file_path[i]):
                                    # delete the folder and its contents recursively
                                    shutil.rmtree(file_path[i])
                                    deleted_files.append(what)
                                    for j in range(len(maindata)):
                                        del maindata[j][i]
                            error_try = ''

                        except Exception as e:
                            error_try = ' ' + str(e)

                        # Assign new data
                        newmaindata = maindata
                        clear_file_initial = True
                        clear_data_main = False

                        if error_try != '':
                            color_temp = "danger"
                        else:
                            color_temp = "success"

                        error_temp = 'FILES CLEARED: ' + ','.join(deleted_files) + error_try

                        clear_file_checklist = []

            # Return required values
            return error_temp, color_temp, True, error_perm, color_perm, open_perm, newmaindata, clear_data_main, clear_file_initial, clear_file_checklist

    except Exception as e:

        error_temp = str(e)
        color_temp = 'danger'
        # Return required values
        return error_temp, color_temp, True, no_update, no_update, no_update, no_update, no_update, no_update, no_update

# Callback 10
# Callback which syncs the all button of the upload checklist. If all is clicked all files will be selected.
# If all files are clicked all will be selected
@app.callback(
        Output(component_id="upload_file_checklist", component_property='value'),
        Output(component_id='all_upload_file_checklist', component_property='value'),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id="upload_file_checklist", component_property='value'),
        Input(component_id='all_upload_file_checklist', component_property='value'),
        State(component_id='filename_filepath', component_property='data'),
        prevent_initial_call=True
        )

def file_upload_sync_checklist(upload_file_check, all_upload_file_check, filename_filepath_data):

    # Prevent update if there are no file names
    if filename_filepath_data is None:
        raise PreventUpdate

    # Try/Except, used to catch any errors not considered
    try:

        # Split up the triggered callback
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if input_id == "upload_file_checklist":
            # If the upload file checklist input triggered the callback, update the all upload file checklist
            all_upload_file_check = ["All"] if set(upload_file_check) == set(filename_filepath_data[0]) else []
        else:
            # If the all upload file checklist input triggered the callback, update the upload file checklist
            upload_file_check = filename_filepath_data[0] if all_upload_file_check else []

        # Return the updated upload file checklist and all upload file checklist
        return upload_file_check, all_upload_file_check, no_update, no_update, no_update

    except Exception as e:

        # If any error display message
        error = str(e)
        color = 'danger'

        return no_update, no_update, error, color, True

# Callback 11
# Callback which syncs the all button of the clear file checklist. If all is clicked all files will be selected.
# If all files are clicked all will be selected
@app.callback(
        Output(component_id="clear_file_checklist", component_property='value'),
        Output(component_id='all_clear_file_checklist', component_property='value'),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id="clear_file_checklist", component_property='value'),
        Input(component_id='all_clear_file_checklist', component_property='value'),
        Input(component_id='filestorage', component_property='data'),
        prevent_initial_call=True
        )

def file_clear_sync_checklist(clear_file_check, all_clear_check, data):

    # If stored data is none prevent update
    if data is None:
        raise PreventUpdate

    # Try/Except, used to catch any errors not considered
    try:

        # Extract the ID of the input that triggered the callback
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Extract the file options from the data dictionary
        file_options = data[0]

        if input_id == "clear_file_checklist":
            # If the clear file checklist input triggered the callback, update the all clear checklist
            all_clear_check = ["All"] if set(clear_file_check) == set(file_options) else []
        else:
            # If the all clear checklist input triggered the callback, update the clear file checklist
            clear_file_check = file_options if all_clear_check else []

        # Return the updated upload file checklist and all upload file checklist
        return clear_file_check, all_clear_check, no_update, no_update, no_update

    except Exception as e:

        # If any error display message
        error = str(e)
        color = 'danger'

    return no_update, no_update, error, color, True

# Callback 12
# Callback which syncs the all button of the vel checklist. If all is clicked all options will be selected.
# If all options are clicked all will be selected
@app.callback(
    Output(component_id="vel_checklist", component_property='value'),
    Output(component_id='all_vel_checklist', component_property='value'),
    Output(component_id='Download_alert', component_property='children', allow_duplicate=True),
    Output(component_id='Download_alert', component_property='color', allow_duplicate=True),
    Output(component_id='Download_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id="vel_checklist", component_property='value'),
    Input(component_id='all_vel_checklist', component_property='value'),
    prevent_initial_call=True
)

def vel_sync_checklist(vel_check, all_vel_checklist):

    # Try/Except, used to catch any errors not considered
    try:

        input_id = ctx.triggered[0]["prop_id"].split(".")[0]

        vel_type = ['t','U1','Ux', 'Uy', 'Uz']

        if input_id == "vel_checklist":
            # If the velocity checklist input triggered the callback, update the all velocity checklist
            all_vel_checklist = ["All"] if set(vel_check) == set(vel_type) else []
        else:
            # If the all velocity checklist input triggered the callback, update the velocity checklist
            vel_check = vel_type if all_vel_checklist else []

        # Return the updated velocity checklist and all velocity checklist
        return vel_check, all_vel_checklist, no_update, no_update, False

    except Exception as e:

        # If any error display message
        error = str(e)
        color = 'danger'

        return no_update, no_update, error, color, True

# Callback 13
# Call back which updates the download time range to prevent error
@app.callback(
        Output(component_id="big_t", component_property='value', allow_duplicate=True),
        Output(component_id="small_t", component_property='value', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id="small_t", component_property='value'),
        Input(component_id="big_t", component_property='value'),
        prevent_initial_call=True)

def update_vals(small_val, large_val):

    if large_val is None or small_val is None:
        raise PreventUpdate

    # Try/Except, used to catch any errors not considered
    try:

        large_val, small_val = update_values(large_val, small_val)

        # Return the updated large and small input values
        return large_val, small_val, no_update, no_update, no_update,

    except Exception as e:
        # If any error display message
        error = str(e)
        color = 'danger'

        return no_update, no_update, error, color, True,

# Callback 14
# Callback to download data
@app.callback(
        Output(component_id='Download_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Workspace_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='Loading_variable_Download', component_property='data'),
        Output(component_id='Workspace_store', component_property='clear_data', allow_duplicate=True),
        Output(component_id='filestorage', component_property='clear_data', allow_duplicate=True),
        Output(component_id='filename_filepath', component_property='clear_data', allow_duplicate=True),
        Input(component_id="btn_download", component_property='n_clicks'),
        State(component_id='Workspace_store', component_property='data'),
        State(component_id="file_name_input", component_property='value'),
        State(component_id="small_t", component_property='value'),
        State(component_id="big_t", component_property='value'),
        State(component_id="vel_checklist", component_property='value'),
        State(component_id="file_checklist", component_property='value'),
        State(component_id='filestorage', component_property='data'),
        prevent_initial_call=True)

def download(n_clicks, Workspace_data, selected_name, smallt, bigt, vector_value, file, file_data):

    try:
        # If download button is pressed
        if "btn_download" == ctx.triggered_id:

            if Workspace_data is None:
                error_temp = 'UPDATE WORKSPACE'
                color_temp = 'danger'
                error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                color_perm = 'danger'
                open_perm = True
                Loading_variable = 'done'
                Workspace_store_clear = True
                filestorage_clear = True
                filename_filepath_clear = True
            else:

                # If workspace doesn't exist clear files
                if not os.path.exists(Workspace_data):

                    error_temp = 'ERROR IN FINDING WORKSPACE FOLDER'
                    color_temp = 'danger'
                    error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                    color_perm = 'danger'
                    open_perm = True
                    Loading_variable = 'done'
                    Workspace_store_clear = True
                    filestorage_clear = True
                    filename_filepath_clear = True

                else:

                    Workspace_store_clear = False
                    filestorage_clear = False
                    filename_filepath_clear = False

                    error_perm = no_update
                    color_perm = no_update
                    open_perm = no_update

                    Download_Path = os.path.join(Workspace_data, 'Downloads')

                    # Check if the folder exists
                    if not os.path.exists(Download_Path):
                        # Create the folder
                        os.mkdir(Download_Path)

                    # If no file selected display error message
                    if file is None:

                        error_temp = 'NO FILE SELECTED'

                        color_temp = 'danger'

                    # If quantity is not picked
                    elif vector_value == [] or vector_value is None:

                        error_temp = 'NO QUANTITY SELECTED'

                        color_temp = 'danger'

                    else:

                        # Load requested data sets
                        i = file_data[0].index(file)

                        numpy_vect_data = []

                        def load_array_memmap(filename, folder_path, dtype, shape, row_numbers):
                            filepath = os.path.join(folder_path, filename)
                            mapped_data = np.memmap(filepath, dtype=dtype, mode='r', shape=shape)

                            if row_numbers == 'all':
                                loaded_data = mapped_data[:]
                            else:
                                loaded_data = mapped_data[row_numbers]

                            return loaded_data

                        file_path = file_data[4][i]

                        shape_dtype = file_data[1][i]

                        shape, dtype = shape_dtype

                        t = load_array_memmap('t.dat', file_path, dtype=dtype, shape=shape[0], row_numbers='all')

                        min1 = file_data[5][i]
                        max1 = file_data[6][i]

                        # Error messages
                        smallt_error = 'THE DATA HAS BEEN CUT TO THE MINIMUM TIME AS THE REQUESTED TIME IS OUTSIDE THE' \
                                       ' AVAILABLE RANGE.'+' AVAILABLE TIME RANGE FOR SELECTED DATA: ('+ str(min1) +' TO ' + str(max1) +'). '

                        bigt_error = 'THE DATA HAS BEEN CUT TO THE MAXIMUM TIME AS THE REQUESTED TIME IS OUTSIDE THE AVAILABLE' \
                                     ' RANGE.'+' AVAILABLE TIME RANGE FOR SELECTED DATA: ('+ str(min1) +' TO ' + str(max1) +').'

                        both_t_error = 'THE DATA HAS BEEN CUT TO THE MAXIMUM AND MINIMUM TIME AS THE REQUESTED TIME IS OUTSIDE' \
                                       ' THE AVAILABLE RANGE.'+' AVAILABLE TIME RANGE FOR SELECTED DATA: ' \
                                                               '(' + str(min1) + ' TO ' + str(max1) +').'

                        both_t_NO_error = 'THE DATA HAS BEEN CUT TO THE SPECIFIED LIMITS'

                        # Cut data based on conditions and assign error message
                        if smallt is None and bigt is None:
                            bigt = max1
                            smallt = min1
                            error_cut = both_t_error

                        elif smallt is None and bigt is not None:
                            smallt = min1
                            error_cut = smallt_error

                        elif bigt is None and smallt is not None:
                            bigt = max1
                            error_cut = bigt_error

                        else:

                            if smallt < min1 and bigt > max1:
                                smallt = min1
                                bigt = max1
                                error_cut = both_t_error

                            elif smallt < min1:
                                bigt = min1
                                error_cut = smallt_error


                            elif bigt > max1:
                                bigt = max1
                                error_cut = bigt_error

                            else:
                                error_cut = both_t_NO_error


                        # Assign mask based on condition
                        mask = (t >= smallt) & (t <= bigt)
                        # From mask calculated row numbers
                        row_numbers = np.where(mask)[0].tolist()
                        # Load requested data
                        for vector in vector_value:
                            if vector == 't':
                                numpy_vect_data.append(t[mask])
                            elif vector != 't':
                                numpy_vect_data.append(
                                    load_array_memmap(vector + '.dat', file_path, dtype=dtype, shape=shape[0],
                                                      row_numbers=row_numbers))

                        # Concatenate the arrays
                        concatenated_array = np.column_stack(numpy_vect_data)
                        concatenated_array1 = np.append([vector_value],concatenated_array,  axis=0)

                        # Assigning filenames
                        if selected_name is None or selected_name == '':
                            filename = file
                            error_special = ''
                        else:
                            # Remove special characters from filename
                            filename = re.sub(r'[^\w\s\-_]+', '',selected_name)
                            # Assign error message
                            if filename != selected_name:
                                error_special = ' DISALLOWED CHARACTERS HAVE BEEN REMOVED FROM THE FILENAME. ' \
                                            'THE FILE HAS BEEN SAVED AS: ' + filename + '.'
                                if filename == '':
                                    filename = file
                                    error_special = ' DISALLOWED CHARACTERS HAVE BEEN REMOVED FROM THE FILENAME. ' \
                                            'THE FILE HAS BEEN SAVED AS: ' + filename + '.'
                            else:
                                error_special = ''

                        # Function to generate a unique filename based on existing files in the directory
                        def get_unique_filename(base_path, name):
                            counter = 1
                            new_name = name

                            while os.path.isfile(os.path.join(base_path, new_name + '.csv')):
                                new_name = f"{name} ({counter})"
                                counter += 1

                            return os.path.normpath(os.path.join(base_path, new_name))

                        # Get a unique filename for the CSV file to be saved
                        new_filename_path = get_unique_filename(Download_Path, filename)

                        # Save the concatenated array as a CSV file
                        np.savetxt(new_filename_path + '.csv', concatenated_array1, delimiter=",", fmt="%s")

                        # Prepare a message indicating the file path of the saved data
                        error_filepath = ' DATA SAVED IN: ' + new_filename_path + '.csv'

                        # Combine error messages and set the alert color to 'primary'
                        error_temp = error_cut + error_special + error_filepath
                        color_temp = 'primary'

                    # Update the loading status to 'done'
                    Loading_variable = 'done'

        # Return error message, alert color, whether the alert should be open, and loading status
        return error_temp, color_temp, True, error_perm, color_perm, open_perm, Loading_variable,\
            Workspace_store_clear, filestorage_clear, filename_filepath_clear

    except Exception as e:
        # If any error occurs, display the error message and set the alert color to 'danger'
        error_temp = str(e)
        color_temp = 'danger'

        # Return error message, alert color, whether the alert should be open, and no update for the loading status
        return error_temp, color_temp, True, no_update, no_update, no_update, no_update,\
            no_update, no_update, no_update

# Callback 15
# Call back which updates the TI table time range to prevent error
@ app.callback(
    Output(component_id="big_t_TI", component_property='value', allow_duplicate=True),
    Output(component_id="small_t_TI", component_property='value', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='children', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='color', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id="small_t_TI", component_property='value'),
    Input(component_id="big_t_TI", component_property='value'),
    prevent_initial_call=True)

def update_vals2(small_val, large_val):

    if large_val is None or small_val is None:
        raise PreventUpdate

    # Try/Except, used to catch any errors not considered
    try:

        large_val, small_val = update_values(large_val, small_val)

        # Return the updated large and small input values
        return large_val, small_val, no_update, no_update, no_update,

    except Exception as e:
        # If any error display message
        error = str(e)
        color = 'danger'

        return no_update, no_update, error, color, True,

# Callback 16
# Callback to analyse data and update TI table
@app.callback(
    [Output(component_id='TI_Table', component_property='data', allow_duplicate=True),
     Output(component_id='TI_alert', component_property='children', allow_duplicate=True),
     Output(component_id='TI_alert', component_property='color', allow_duplicate=True),
     Output(component_id='TI_alert', component_property='is_open', allow_duplicate=True),
     Output(component_id='Workspace_alert', component_property='children', allow_duplicate=True),
     Output(component_id='Workspace_alert', component_property='color', allow_duplicate=True),
     Output(component_id='Workspace_alert', component_property='is_open', allow_duplicate=True),
     Output(component_id='Workspace_store', component_property='clear_data', allow_duplicate=True),
     Output(component_id='filestorage', component_property='clear_data', allow_duplicate=True),
     Output(component_id='filename_filepath', component_property='clear_data', allow_duplicate=True),
     Output(component_id='Loading_variable_Table', component_property='data', allow_duplicate=True)],
    [Input(component_id='TI_btn_download', component_property='n_clicks'),
     State(component_id='filestorage', component_property='data'),
     State(component_id='DataSet_TI', component_property='value'),
     State(component_id="small_t_TI", component_property='value'),
     State(component_id="big_t_TI", component_property='value'),
     State(component_id='TI_Table', component_property='data'),
     State(component_id='TI_Table', component_property='columns'),
     State(component_id='Workspace_store', component_property='data'),
     ])
# Define a function to calculate turbulence intensity and update the table
def TI_caluculate(n_clicks, file_data, chosen_file, small_TI, big_TI, table_data, column_data, Workspace_data):
    # Use try-except to catch any unexpected errors
    try:
        # Check if the button with ID "TI_btn_download" was clicked
        if "TI_btn_download" == ctx.triggered_id:

            if Workspace_data is None:
                error_temp = 'UPDATE WORKSPACE'
                color_temp = 'danger'
                error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                color_perm = 'danger'
                open_perm = True
                Loading_variable = 'done'
                Workspace_store_clear = True
                filestorage_clear = True
                filename_filepath_clear = True
            else:

                # If workspace doesn't exist clear files
                if not os.path.exists(Workspace_data):

                    error_temp = 'ERROR IN FINDING WORKSPACE FOLDER'
                    color_temp = 'danger'
                    error_perm = 'PLEASE UPLOAD NEW WORKSPACE'
                    color_perm = 'danger'
                    open_perm = True
                    Loading_variable = 'done'
                    Workspace_store_clear = True
                    filestorage_clear = True
                    filename_filepath_clear = True

                else:

                    error_perm = no_update
                    color_perm = no_update
                    open_perm = no_update
                    Workspace_store_clear = no_update
                    filestorage_clear = no_update
                    filename_filepath_clear = no_update

                    # If no dataset is chosen, show an error message and don't update the table data
                    if chosen_file is None:
                        error_temp = 'TURBULENCE INTENSITY NOT CALCULATED. PLEASE CHECK YOU HAVE SELECTED A DATASET'
                        color_temp = 'danger'
                        table_data = no_update
                    else:
                        # Check if the inputted time range is correct
                        if small_TI == big_TI and small_TI is not None and big_TI is not None:
                            error_temp = 'TURBULENCE INTENSITY NOT CALCULATED. PLEASE CHECK THAT THE INPUTTED TIME' \
                                         ' RANGE IS CORRECT'
                            color_temp = 'danger'
                            table_data = no_update
                        else:
                            # Define a function to load data from a memmap file
                            def load_array_memmap(filename, folder_path, dtype, shape, row_numbers):
                                filepath = os.path.join(folder_path, filename)
                                mapped_data = np.memmap(filepath, dtype=dtype, mode='r', shape=shape)

                                if row_numbers == 'all':
                                    loaded_data = mapped_data[:]
                                else:
                                    loaded_data = mapped_data[row_numbers]

                                return loaded_data

                            # Find the index of the chosen file in file_data list
                            i = file_data[0].index(chosen_file)

                            # Get shape and dtype information for the chosen file
                            shape_dtype = file_data[1][i]
                            shape, dtype = shape_dtype

                            # Get the file path, min and max time values for the chosen file
                            file_path = file_data[4][i]
                            min1 = file_data[5][i]
                            max1 = file_data[6][i]

                            # Error messages
                            smallt_error = 'TURBULENCE INTENSITY CALCULATED. THE DATA HAS BEEN CUT TO THE MINIMUM TIME' \
                                           ' AS THE REQUESTED TIME IS OUTSIDE THE AVAILABLE RANGE.'+ \
                                           ' AVAILABLE TIME RANGE FOR SELECTED DATA: (' + str(min1)+' TO '+str(max1)+')'

                            bigt_error = 'TURBULENCE INTENSITY CALCULATED. THE DATA HAS BEEN CUT TO THE MAXIMUM TIME' \
                                         ' AS THE REQUESTED TIME IS OUTSIDE THE AVAILABLE RANGE.' + \
                                         ' AVAILABLE TIME RANGE FOR SELECTED DATA: (' + str(min1) +' TO '+str(max1)+ ')'

                            both_t_error = 'TURBULENCE INTENSITY CALCULATED. THE DATA HAS BEEN CUT TO THE MAXIMUM' \
                                           ' AND MINIMUM TIME AS THE REQUESTED TIME IS OUTSIDE THE AVAILABLE RANGE.' + \
                                           ' AVAILABLE TIME RANGE FOR SELECTED DATA: (' + str(min1)+' TO '+str(max1)+ ')'

                            both_t_NO_error = 'TURBULENCE INTENSITY CALCULATED. THE DATA HAS BEEN CUT TO THE ' \
                                              'SPECIFIED LIMITS'

                            # Cut data based on conditions
                            if small_TI is None and big_TI is None:
                                big_TI = max1
                                small_TI = min1
                                color_temp = 'danger'
                                error_temp = both_t_error

                            elif small_TI is None and big_TI is not None:
                                small_TI = min1
                                color_temp = 'danger'
                                error_temp = smallt_error

                            elif big_TI is None and small_TI is not None:
                                big_TI = max1
                                color_temp = 'danger'
                                error_temp = bigt_error

                            else:

                                if small_TI < min1 and big_TI > max1:
                                    small_TI = min1
                                    big_TI = max1
                                    color_temp = 'danger'
                                    error_temp = both_t_error

                                elif small_TI < min1:
                                    small_TI = min1
                                    color_temp = 'danger'
                                    error_temp = smallt_error


                                elif big_TI > max1:
                                    big_TI = max1
                                    color_temp = 'danger'
                                    error_temp = bigt_error

                                else:
                                    error_temp = both_t_NO_error
                                    color_temp = 'success'

                            # Load time data using the memmap function
                            t = load_array_memmap('t.dat', file_path, dtype=dtype, shape=shape[0], row_numbers='all')

                            # Apply a mask based on the specified time range
                            mask = (t >= small_TI) & (t <= big_TI)

                            # Update color and row numbers based on the mask
                            row_numbers = np.where(mask)[0].tolist()

                            # Load velocity components data using the memmap function
                            ux = load_array_memmap('Ux.dat', file_path, dtype=dtype, shape=shape[0],
                                                   row_numbers=row_numbers)
                            uy = load_array_memmap('Uy.dat', file_path, dtype=dtype, shape=shape[0],
                                                   row_numbers=row_numbers)
                            uz = load_array_memmap('Uz.dat', file_path, dtype=dtype, shape=shape[0],
                                                   row_numbers=row_numbers)

                            # Calculate turbulence intensity and mean velocity components
                            TI, U1, Ux, Uy, Uz = calculate_turbulence_intensity(ux, uy, uz)

                            # If table data is empty, initialize it as an empty list
                            if table_data is None:
                                table_data = []
                            print(file_data[2][0])

                            # Create new data entry with calculated values
                            new_data = [
                                {
                                    'FileName': chosen_file,
                                    'CalFile': file_data[2][i],
                                    'SF': file_data[3][i],
                                    'Time_1': round(small_TI, 2),
                                    'Time_2': round(big_TI, 2),
                                    'Ux': round(Ux, 6),
                                    'Uy': round(Uy, 6),
                                    'Uz': round(Uz, 6),
                                    'U1': round(U1, 6),
                                    'TI': round(TI, 6),
                                }
                            ]

                            # Append new data to the table_data
                            table_data.append({c['id']: new_data[0].get(c['id'], None) for c in column_data})

                    # Set loading variable to 'done'
                    Loading_variable = 'done'

        # Return updated table data, error message, color, and loading variable
        return table_data, error_temp, color_temp, True, error_perm, color_perm, open_perm, Workspace_store_clear,\
            filestorage_clear, filename_filepath_clear, Loading_variable

    except Exception as e:
        # If any error occurs, display the error message
        error_temp = str(e)
        color_temp = 'danger'

        return no_update, error_temp, color_temp, True, no_update, no_update, no_update, no_update, no_update,\
            no_update, no_update

# Callback 17
# Callback to clear parameters table
@app.callback(
    Output(component_id='TI_Table', component_property='data', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='children', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='color', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='Clear_Table', component_property='n_clicks'),
    prevent_initial_call=True)

def clear_table(n_clicks):

    # Try/Except, used to catch any errors not considered
    try:

        # If clear table pressed clear data and display error message
        if "Clear_Table" == ctx.triggered_id:

            error = 'TABLE CLEARED'

            color = 'success'

            table_data = []

        return table_data, error, color, True

    except Exception as e:

        # If any error display message
        error = str(e)
        color = 'danger'

        return no_update, error, color, True

# Callback 18
# Callback which updates the graph based on graph options
@app.callback(
        Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
        Output(component_id='Graph_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Graph_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Graph_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='Loading_variable_Graph', component_property='data', allow_duplicate=True),
        Input(component_id='plot_bttn', component_property='n_clicks'),
        State(component_id = 'filestorage', component_property = 'data'),
        State(component_id = 'File', component_property = 'value'),
        State(component_id = 'Vect', component_property = 'value'),
        State(component_id='time_small', component_property='value'),
        State(component_id='time_large', component_property='value'),
        State(component_id='legend_Data', component_property='data'),
        State(component_id='title_Data', component_property='data'),
        State(component_id='legend_onoff', component_property='value'),
        State(component_id='title_onoff', component_property='value'), prevent_initial_call = True)

def update_graph(n_clicks, file_data, file_inputs, vector_inputs1, smallt, bigt, legend_data, title_data, leg, title):
    # Try/Except, used to catch any errors not considered
    try:
        if ctx.triggered_id == 'plot_bttn':
            # If no input do not plot graphs
            if file_inputs == [] or vector_inputs1 == []:
                fig = no_update
                error = 'PLEASE CHECK INPUTS'
                color = 'danger'
                Loading_Variable = 'done'
            else:
                fig = go.Figure()
                current_names = []
                min2 = []
                max2 = []

                # Function to load data from memmap file
                def load_array_memmap(filename, folder_path, dtype, shape, row_numbers):
                    filepath = os.path.join(folder_path, filename)
                    mapped_data = np.memmap(filepath, dtype=dtype, mode='r', shape=shape)

                    if row_numbers == 'all':
                        loaded_data = mapped_data[:]
                    else:
                        loaded_data = mapped_data[row_numbers]

                    return loaded_data

                # Get min and max time values for each file
                for file in file_inputs:
                    i = file_data[0].index(file)
                    min2.append(file_data[5][i])
                    max2.append(file_data[6][i])

                min1 = min(min2)
                max1 = max(max2)

                # Error messages
                smallt_error = 'THE DATA HAS BEEN CUT TO THE MINIMUM TIME AS THE REQUESTED TIME IS OUTSIDE THE' \
                               ' AVAILABLE RANGE.'+'AVAILABLE TIME RANGE FOR SELECTED DATA: ('+ str(min1) +' TO ' + max1 +')'
                bigt_error = 'THE DATA HAS BEEN CUT TO THE MAXIMUM TIME AS THE REQUESTED TIME IS OUTSIDE THE AVAILABLE' \
                             ' RANGE.'+'AVAILABLE TIME RANGE FOR SELECTED DATA: ('+ min1 +' TO ' + max1 +')'
                both_t_error = 'THE DATA HAS BEEN CUT TO THE MAXIMUM AND MINIMUM TIME AS THE REQUESTED TIME IS OUTSIDE' \
                               ' THE AVAILABLE RANGE.'+'AVAILABLE TIME RANGE FOR SELECTED DATA: ' \
                                                       '(' + min1 + ' TO ' + max1 +')'
                both_t_NO_error = 'THE DATA HAS BEEN CUT TO THE SPECIFIED LIMITS'

                # Cut data based on conditions
                if smallt is None and bigt is None:
                    bigt = max1
                    smallt = min1
                    error_cut = both_t_error
                    error_cut_good = ''

                elif smallt is None and bigt is not None:
                    smallt = min1
                    error_cut = smallt_error
                    error_cut_good = ''

                elif bigt is None and smallt is not None:
                    bigt = max1
                    error_cut = bigt_error
                    error_cut_good = ''

                else:

                    if smallt < min1 and bigt > max1:
                        smallt = min1
                        bigt = max1
                        error_cut = both_t_error
                        error_cut_good = ''

                    elif smallt < min1:
                        bigt = min1
                        error_cut = smallt_error
                        error_cut_good = ''


                    elif bigt > max1:
                        bigt = max1
                        error_cut = bigt_error
                        error_cut_good = ''
                    else:
                        error_cut_good = both_t_NO_error
                        error_cut = ''

                # Loop through the files and vectors to create the graph
                for file in file_inputs:
                    i = file_data[0].index(file)
                    file_path = file_data[4][i]
                    shape_dtype = file_data[1][i]
                    shape, dtype = shape_dtype

                    t = load_array_memmap('t.dat', file_path, dtype=dtype, shape=shape[0], row_numbers='all')
                    mask = (t >= smallt) & (t <= bigt)

                    numpy_vect_data = {file: {'t': t[mask]}}
                    row_numbers = np.where(mask)[0].tolist()

                    for vector in vector_inputs1:
                        numpy_vect_data[file][vector] = load_array_memmap(vector + '.dat', file_path,
                                                                          dtype=dtype,
                                                                          shape=shape[0],
                                                                          row_numbers=row_numbers)

                        # Creating a list of current legend names
                        current_names.append(f"{file}{' '}{vector}")

                        # Plotting data
                        fig.add_trace(
                            go.Scattergl(
                                name=f"{file}{' '}{vector}",
                                showlegend=True,
                                x=numpy_vect_data[file]['t'],
                                y=numpy_vect_data[file][vector])
                        )

                    # Update x and y axes labels
                    fig.update_layout(
                        xaxis_title="Time (s)",
                        yaxis_title="Velocity (m/s)",
                        legend=dict(
                            y=1,
                            x=0.5,
                            orientation="h",
                            yanchor="bottom",
                            xanchor="center"),
                    )

                    # Update legend and title based on user input
                    if legend_data is None:
                        if leg == 'Off':
                            fig.layout.update(showlegend=False)
                            error_leg = ''
                        elif leg == 'On':
                            fig.layout.update(showlegend=True)
                            error_leg = ''
                    else:
                        if leg == 'Off':
                            fig.layout.update(showlegend=False)
                            error_leg = ''
                        elif leg == 'On':
                            legend_name_list = legend_data.split(',')
                            newname_result = {}

                            if len(current_names) == len(legend_name_list):
                                for i, current_name in enumerate(current_names):
                                    newnames = {current_name: legend_name_list[i]}
                                    newname_result.update(newnames)

                                fig.for_each_trace(lambda t: t.update(name=newname_result[t.name],
                                                                      legendgroup=newname_result[t.name],
                                                                      hovertemplate=t.hovertemplate.replace(
                                                                      t.name,
                                                                      newname_result[
                                                                      t.name]) if t.hovertemplate is not None else None)
                                                   )

                                fig.layout.update(showlegend=True)

                                error_leg  = ''

                            else:

                                # If legend entries do not match display error message
                                error_leg = ' NUMBER OF LEGEND ENTRIES DO NOT MATCH'

                    # Update graph title based on user input
                    if title_data is None:
                        if title == 'Off':
                            fig.layout.update(title='')
                        elif title == 'On':
                            fig.layout.update(title='Plot of ' + ', '.join(file_inputs) + ' Data')
                    else:
                        if title == 'Off':
                            fig.layout.update(title='')
                        elif title == 'On':
                            fig.layout.update(title=title_data)

                # Return figure, error message, alert color, and loading status
                if error_leg == '' or error_cut == '':
                    color = 'success'
                else:
                    color = 'danger'

                error = 'GRAPH PLOTTED. ' + error_cut + error_cut_good + error_leg
                Loading_Variable = 'done'

        return fig, error, color, True, Loading_Variable

    except Exception as e:
        # If any error display message
        error = str(e)
        color = 'danger'

    return no_update, error, color, True, no_update

# Callback 19
# Callback to clear graph through the click of a button
@app.callback(
        Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
        Output(component_id='File', component_property='value'),
        Output(component_id='Vect', component_property='value'),
        Output(component_id='time_small', component_property='value'),
        Output(component_id='time_large', component_property='value'),
        Output(component_id='Graph_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Graph_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Graph_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id='plot_clear_bttn', component_property='n_clicks'),
    prevent_initial_call=True)

def clear_graph(n_clicks):

    # Try/Except, used to catch any errors not considered
    try:
        if ctx.triggered_id == 'plot_clear_bttn':
            fig = {}
            file = None
            Vect = None
            tsmall = None
            tbig = None
            return fig, file, Vect, tsmall, tbig, no_update, no_update, False,

    except Exception as e:
        # If any error display message
        error = str(e)
        color = 'danger'

        return no_update, no_update, no_update, no_update, no_update, error, color, True,

# Callback 20
# Callback to update legend or title data
@app.callback(
     Output(component_id='New_name', component_property='value'),
     Output(component_id='legend_Data', component_property='data'),
     Output(component_id='title_Data', component_property='data'),
     Output(component_id='Graph_alert', component_property='children', allow_duplicate=True),
     Output(component_id='Graph_alert', component_property='color', allow_duplicate=True),
     Output(component_id='Graph_alert', component_property='is_open', allow_duplicate=True),
     Input(component_id="dropdown_legend_update", component_property='n_clicks'),
     Input(component_id="dropdown_title_update", component_property='n_clicks'),
     Input(component_id="dropdown_clear", component_property='n_clicks'),
     State(component_id='New_name', component_property='value'),
     prevent_initial_call = True)

def update_leg_title_data(n_click, n_clicks1, n_clicks2,  name_input):

    # Try/Except, used to catch any errors not considered
    try:

        # If legend update button is pressed
        if ctx.triggered_id == 'dropdown_legend_update':
            error = 'LEGEND DATA UPDATE'
            color = 'success'
            # Update legend data
            legend_data = name_input
            # No update to title data or name input
            title_data = no_update
            name_input = no_update
            open1 = True
        # If title update button is pressed
        elif ctx.triggered_id == 'dropdown_title_update':
            error = 'LEGEND DATA UPDATE'
            color = 'success'
            # Update title data
            title_data = name_input
            # No update to legend data or name input
            name_input = no_update
            legend_data = no_update
            open1 = True
            error = 'TITLE DATA UPDATED'

        # If clear dropdown pressed clear input box
        elif ctx.triggered_id == 'dropdown_clear':
            # Clear title and legend data
            title_data = None
            legend_data = None
            # Clear input box
            name_input = ''
            open1 = True
            color = 'success'
            error = 'LEGEND AND TITLE DATA CLEARED'

        else:
            # Else no update to any values
            title_data = no_update
            name_input = no_update
            legend_data = no_update
            error = no_update
            color = no_update
            open1 = False

        # Return name, legend and title data
        return name_input, legend_data, title_data, error, color, open1

    except Exception as e:

        # If any error display message
        error = str(e)
        color = 'danger'

        return no_update, no_update, no_update, error, color, True

# Callback 21
# Callback which updates dropdowns
@app.callback(
    Output(component_id="File", component_property='options'),
    Output(component_id='Vect', component_property='options'),
    Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="vel_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="clear_file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="upload_file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id='DataSet_TI', component_property='options'),
    Output(component_id='Graph_alert', component_property='children', allow_duplicate=True),
    Output(component_id='Graph_alert', component_property='color', allow_duplicate=True),
    Output(component_id='Graph_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='filestorage', component_property='data'),
    Input(component_id='filename_filepath', component_property='data'),
    prevent_initial_call=True)

def update_dropdowns1(data, filename_filepath_upload_data):

    # Try/Except, used to catch any errors not considered
    try:
        if filename_filepath_upload_data is not None:
            upload_file_checklist = filename_filepath_upload_data[0]
        else:
            upload_file_checklist = []


        if data is None:
            # If the data is None, set all dropdown options to empty lists
            vect_options = []
            file_dropdown_options = []
            file_checklist = []
            clear_file_check = []
            vel_checklist = []
            DataDrop_TI = []
        else:
            # If the data is not None, set the dropdown options and checklists accordingly
            vect_options = ['U1','Ux', 'Uy', 'Uz']
            vel_checklist = ['t','U1','Ux', 'Uy', 'Uz']
            file_dropdown_options = data[0]
            file_checklist = data[0]
            DataDrop_TI = data[0]
            clear_file_check = data[0]

        # Return the updated dropdown options and checklists
        return file_dropdown_options, vect_options, file_checklist, vel_checklist, clear_file_check,upload_file_checklist, DataDrop_TI, no_update, no_update, False,

    except Exception as e:

        # If any error display message
        error = str(e)
        color = 'danger'

        # Return the updated dropdown options and checklists
        return no_update, no_update, no_update, no_update, no_update,no_update, no_update, error, color, False,

# Run app
if __name__== '__main__':
    app.run_server(debug=True)

