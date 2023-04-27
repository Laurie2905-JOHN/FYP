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
import shutil



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
Barn_data = {}

# Define the layout of the app
app.layout = dbc.Container([

    # Header row with title
    dbc.Row([
        dbc.Col(
            html.H1("BARNACLE SENSOR ANALYSIS DASHBOARD",
                    className='text-center font-weight-bolder, mb-1'),
            width=12),
    ]),

    # Graph row
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='Velocity_Graph', figure={}),
            width=12),
    ]),

    # Options row
    dbc.Row([
        # Graph options header
        dbc.Col(
            html.H5('Graph Options', className="text-center"),
            width = 12),

        # Horizontal line
        dbc.Col(
            html.Hr(),
            width = 12),

        # Alert box
        dbc.Col([
            dbc.Alert(
                id="alert",
                is_open=False,
                dismissable=True,
                duration=20000),
        ], width=12),

        # Time slider row
        dbc.Row([
            # Time slider label
            dbc.Col(
                dbc.Row(
                    dbc.Label("Time Slider", className="text-center mb-1"),
                ),
            ),

            # Time slider component
            dbc.Col(
                dcc.RangeSlider(
                    id='time-range',
                    min=1,
                    max=10,
                    value=[1, 10],
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                ),
                width = 12,
                className="mb-2"
            ),
        ]),
    ]),

    # Create a row with one column
    dbc.Row([
        dbc.Col([
            # Stack two elements vertically
            dbc.Stack([
                # Label for the dropdowns
                dbc.Label("Data Options", className='text-center'),
                # Stack two dropdowns vertically
                dbc.Stack([
                    dcc.Dropdown(
                        id="File",
                        options=[],
                        multi=True,
                        value=[],
                        placeholder="Select a dataset"),
                    dcc.Dropdown(
                        id="Vect",
                        options=[],
                        multi=True,
                        value=[],
                        placeholder="Select a quantity"),
                ], gap=3)
            ]),
        ]),

        dbc.Col([
            dbc.Stack([
                dbc.Label('Line Thickness', className="text-center"),
                dcc.Slider(
                    min=0.5,
                    max=5,
                    value=1,
                    step=0.1,
                    id="line_thick",
                    marks={0.5: {'label': 'Thin'}, 5: {'label': 'Thick'}},
                    updatemode='drag'
                ),
                html.Hr(),
                dbc.Row([
                    dbc.Col(
                        dbc.Stack([
                            dbc.Label("Title", className="text-start"),
                            dbc.RadioItems(id='title_onoff', value='On', options=['On', 'Off'], inline=True)
                        ])
                    ),
                    dbc.Col(
                        dbc.Stack([
                            dbc.Label('Legend', className="text-start"),
                            dbc.RadioItems(id='legend_onoff', value='On', options=['On', 'Off'], inline=True),
                        ]),
                        className='mb-3'
                    ),
                ]),

                dbc.Row(
                    # Stack of input elements for updating the title or legend
                    dbc.Stack([
                        dbc.InputGroup([
                            # Dropdown menu for selecting the update action
                            dbc.DropdownMenu([
                                dbc.DropdownMenuItem("Update Title", id="dropdown_title_update"),
                                dbc.DropdownMenuItem("Update Legend", id="dropdown_legend_update"),
                                dbc.DropdownMenuItem(divider=True),
                                dbc.DropdownMenuItem("Clear", id="dropdown_clear"),
                            ],
                                label="Update"),
                            # Input element for entering the new title or legend
                            dbc.Input(
                                id="New_name",
                                type='text',
                                placeholder="Enter text to update legend or title",
                                debounce=True
                            ),
                        ]),
                        # Horizontal direction of the stack
                    ], direction="horizontal"),
                ),

            ]),
        ])], className = 'mb-2', justify="center", align="center"),


    dbc.Row([
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ], className='mb-2'),

    dbc.Row(
        dbc.Col(
            html.H5('Turbulence Parameters', className='center-text'),
            width=12,
            className="text-center"
        ),  # Column containing the header for the download files section
    ),

    dbc.Row(
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ),

    dbc.Row(
        dbc.Col([
            dbc.Alert(
                id="TI_alert",
                is_open=False,
                dismissable=True,
                duration=30000
            ),  # Alert component to show status of file download
        ], width=12),
    ),


    dbc.Col([

        dbc.Row([

            dbc.Col(

                dbc.Stack([

                dbc.Button("Calculate", id="TI_btn_download", size="lg"),

                dbc.Button("Clear Table", id="Clear_Table", size="lg"),

                ], gap = 2),


            width = 3),

            dbc.Col(

            dbc.Stack([

                dcc.Dropdown(
                    id="DataSet_TI",
                    options=[],
                    multi=False,
                    placeholder="Select a Dataset"),

            dbc.Input(id="small_t_TI", type="number", placeholder="Min Time", debounce=True),

            dbc.Input(id="big_t_TI", type="number", placeholder="Max Time", debounce=True),



            ], gap =2 ),

            width = 3),

            ], align='center', justify='center')

        ], width = 12),



  dbc.Col(

dash_table.DataTable(id = 'TI_Table',
                     columns =
                     [
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
                     row_deletable=True
),

  width = 12),

    # Empty row with a horizontal line
    dbc.Row([
        dbc.Col(
            html.Hr(),
            width=12
        ),
    ], className='mb-2'),

    dbc.Row([

        # Column for "Upload/Clear Files" title
        dbc.Col(
            html.H5('File Upload', className='center-text'),
            width=12,
            className="text-center"
        ),

        # Column for alert message (hidden by default)
        dbc.Col([
            dbc.Alert(
                id="ClearFiles_alert",
                is_open=False,
                dismissable=True,
                duration=30000
            ),
        ], width=12),

        # Horizontal line
        dbc.Col(
            html.Hr(),
            width=12
        ),


        # Column for "Upload/Clear Files" title
        dbc.Col(
            html.H5('Workspace', className='center-text'),
            width=12,
            className="text-center"
        ),

        dbc.Col([
            dbc.Alert(
                id="Workspace_alert",
                is_open=False,
                class_name='text-center'
            ),
        ], width=10),

        dbc.Col(
        dbc.InputGroup([
            # Dropdown menu for selecting the update action
            dbc.DropdownMenu([
                dbc.DropdownMenuItem("Update Workspace", id="Workspace_update"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("Clear Workspace", id="Workspace_clear"),
            ],
                label="Update"),
            # Input element for entering the new title or legend
            dbc.Input(
                id="Workspace",
                type='text',
                placeholder="Enter Workspace Filepath",
                debounce=True
            ),

        ]),
    width =11, class_name = 'mb-3'),

        # Horizontal line
        dbc.Col(
            html.Hr(),
            width=12
        ),

        dbc.Col(

            dbc.InputGroup([
                # Dropdown menu for selecting the update action
                dbc.DropdownMenu([
                    dbc.DropdownMenuItem("Add to Uploads", id="dropdown_BARN_update"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Clear Uploads", id="dropdown_BARN_clear"),
                ],
                    label="Update"),
                # Input element for entering the new title or legend
                dbc.Input(
                    id="submit_files",
                    type='text',
                    placeholder="Enter BARNACLE Filepath",
                    debounce=True
                ),
            ]),
            # Horizontal direction of the stack

    width = 11, class_name = 'mb-3'),


        # Column for file selection/upload
        dbc.Col([
            dbc.Stack([

                dcc.Upload(
                    id='submit_Cal_file',
                    children=html.Div([
                        html.A(id = 'Cal_select_text',children = 'Select a Calibration File')
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
                    multiple=True
                ),


                dbc.Alert(id = 'calAlert', dismissable=False, class_name = 'text-center'),

                ], gap = 3)


        ], width=3),

        # Column for uploading files
        dbc.Col(
            dbc.Stack([

                dbc.Button(
                    'Process Selected Files',
                    id='newfile',
                    outline=True,
                    color="primary",
                    className="me-1",
                    n_clicks=0,
                ),

                dbc.Input(id="Sample_rate", min=0, type="number", placeholder="Enter Sample Frequency", debounce=True),

                html.Label("Select files to analyse"),

                # Checkbox for uploading all files
                dbc.Checklist(
                    ["All"], [], id="all_upload_file_checklist", inline=True
                ),

                # Checkbox for selecting individual files to upload
                dbc.Checklist(value=[], id="upload_file_checklist", inline=True),
            ], gap=3),
            width=3
        ),

        # Column for clearing files
        dbc.Col(
            dbc.Stack([
                dbc.Button(
                    "Clear Selected Files",
                    id='clear_files',
                    outline=True,
                    color="primary",
                    className="me-1",
                    n_clicks=0
                ),

                html.Label("Select files to clear", className='center-text'),

                # Checkbox for clearing all files
                dbc.Checklist(
                    ["All"], [], id="all_clear_file_checklist", inline=True
                ),

                # Checkbox for selecting individual files to clear
                dbc.Checklist(value=[], id="clear_file_checklist", inline=True),
            ], gap=3),
            width=3
        ),


    ], align='start', justify='evenly'),

    dbc.Row([
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ], className='mb-2'),

    dbc.Row(
        dbc.Col(
            html.H5('Download Data', className='center-text'),
            width=12,
            className="text-center"
        ),  # Column containing the header for the download files section
    ),

    dbc.Row(
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ),

    dbc.Row(
        dbc.Col([
            dbc.Alert(
                id="Download_alert",
                is_open=False,
                dismissable=True,
                duration=30000
            ),  # Alert component to show status of file download
        ], width=12),
    ),

dbc.Col([

    dbc.Row([

        dbc.Col(

            dbc.Stack([
                html.Label("Choose Data File"),  # Label for selecting data file

                dbc.RadioItems(id="file_checklist", inline=True),  # Radio buttons for selecting data file

                html.Label("Choose Quantity"),  # Label for selecting quantity of data to download

                dbc.Checklist(["All"], [], id="all_vel_checklist", inline=True),  # Checkbox to select all data

                dbc.Checklist(value=[], options=[], id="vel_checklist", inline=True),
                # Checkbox to select specific data


            ], gap=2),

        width = 4),

        dbc.Col(

            dbc.Stack([
                dbc.Button("Download", id="btn_download", size="lg",color="primary", outline=True),  # Button for downloading selected data

                dbc.Input(id="file_name_input", type="text", placeholder="Enter Filename"),  # Input field for file name

                dbc.Row([
                    dbc.Col(
                        dbc.Input(id="small_t", type="number", placeholder="Min Time", debounce=True)
                    ),  # Input field for minimum time

                    dbc.Col(
                        dbc.Input(id="big_t", min=0, type="number", placeholder="Max Time", debounce=True)
                    ),  # Input field for maximum time

                ], justify="center"),  # Row for input fields for minimum and maximum times

            ], gap=2),

        width = 4),


        ], align='center', justify='center'),



    ], width = 12),




    # # Components for storing and downloading data
    dbc.Spinner(children = [dcc.Store(id='Loading_variable_Process', storage_type='memory')],color="primary",
                fullscreen = True, size = 'lg', show_initially = False, delay_hide = 500, delay_show = 500),

    # # Components for storing and downloading data
    dbc.Spinner(children=[dcc.Store(id='Loading_variable_Table', storage_type='memory')], color="primary",
                fullscreen=True, size='lg', show_initially=False, delay_hide=500, delay_show=500),

    # # Components for storing and downloading data
    dbc.Spinner(children=[dcc.Store(id='Loading_variable_Download', storage_type='memory')], color="primary",
                fullscreen=True, size='lg', show_initially=False, delay_hide=500, delay_show=500),

    # # Components for storing and downloading data
    dbc.Spinner(children=[dcc.Store(id='Loading_variable_Graph', storage_type='memory')], color="primary",
                fullscreen=True, size='lg', show_initially=False, delay_hide=500, delay_show=500),

    dcc.Download(id="download"),
    dcc.Store(id='legend_Data', storage_type='memory'),
    dcc.Store(id='title_Data', storage_type='memory'),
    dcc.Store(id='filestorage', storage_type='session'),
    dcc.Store(id='filename_filepath', storage_type='session'),
    dcc.Store(id='Workspace_store', storage_type='local'),
    dcc.Store(id='Cal_storage', storage_type='local'),
])

@ app.callback(
    Output(component_id='Workspace_store', component_property='data'),
    Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='Workspace_update', component_property='n_clicks'),
    State(component_id='Workspace', component_property='value'),
)

def update_Workspace(n_clicks, Workspace_input):

    if ctx.triggered_id == 'Workspace_update':

        if Workspace_input is None or Workspace_input == '':
            error = 'No Filepath Inputted. Please Check'
            color1 = 'danger'
            open1 = True
            Workspace_data = no_update

        else:

            Workspace_input2 = Workspace_input.replace('"', "")
            Workspace_input3 = os.path.normpath(Workspace_input2)

            if not os.path.exists(Workspace_input3):
                error = 'Please Check Filepath'
                color1 = 'danger'
                open1 = True
                Workspace_data = no_update
            else:
                error = 'Workspace Updated'
                color1 = 'success'
                open1 = True
                Workspace_data = Workspace_input3

        return Workspace_data, error, color1, open1


@app.callback(
        Output(component_id='Workspace', component_property='value', allow_duplicate=True),
        Output(component_id='Workspace_store', component_property='clear_data', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id='Workspace_clear', component_property='n_clicks'),
        State(component_id='Workspace_store', component_property='data'),
        prevent_initial_call = True)

def clear_Workspace(n_clicks,Workspace_data):

    if ctx.triggered_id == 'Workspace_clear':
        if Workspace_data is None:
            error = 'No Workspace Selected to Clear'
            color1 = 'danger'
            Workspace_input = ''
            Workspace_Clear_data = False

        else:

            color1 = 'success'
            deleted_files = []

            for file_name in os.listdir(Workspace_data):
                path = os.path.join(Workspace_data,file_name)
                try:
                    if os.path.isfile(path):
                        # delete the file
                        os.remove(path)
                        deleted_files.append(file_name)
                    elif os.path.isdir(path):
                        # delete the folder and its contents recursively
                        shutil.rmtree(path)
                        deleted_files.append(file_name)
                except Exception as e:
                    print(f"Error deleting {path}: {e}")

            Workspace_Clear_data = True
            Workspace_input = ''

            if deleted_files == []:

                error = 'Workspace Data Cleared'

            else:

                error = 'Workspace Cleared. ' + ', '.join(deleted_files) + ' removed.'

        return Workspace_input, Workspace_Clear_data, error, color1, True
    else:
        raise PreventUpdate


@app.callback(
    Output(component_id='Workspace_alert', component_property='children'),
    Output(component_id='Workspace_alert', component_property='color'),
    Output(component_id='Workspace_alert', component_property='is_open'),
    Input(component_id="Workspace_store", component_property='data'),
)
def update_Workspace_Alert(Workspace_data):

    if Workspace_data is None:
        alert_work = 'No Workspace Selected'
        color1 = 'danger'
    else:
        alert_work = Workspace_data
        color1 = 'primary'

    return alert_work, color1, True

@ app.callback(
    Output(component_id='calAlert', component_property='children'),
    Output(component_id='calAlert', component_property='color'),
    Output(component_id='calAlert', component_property='is_open'),
    Input(component_id="Cal_storage", component_property='data'),
)

def update_cal_text(Cal_data):

    if Cal_data is None:
        alert_cal = 'No Calibration File Selected'
        color1 = 'danger'
    else:
        alert_cal = Cal_data[0][0] + ' Selected'
        color1 = 'primary'


    return alert_cal, color1, True

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
        content_type, content_string = contents[0].split(',')

        decoded = base64.b64decode(content_string)

        if 'xlsx' or 'xlx' in filename:

            cal_data = pd.read_excel(io.BytesIO(decoded))

            cal_data = cal_data.to_dict('list')

            # Remove NaN values from the lists in the dictionary
            cal_data = [filename, {key: [val for val in values if not math.isnan(val)] for key, values in
                                  cal_data.items()}]

            error1 = filename[0] + ' Uploaded Successfully'

            color1 = 'success'

        else:

            error1 = 'Please Upload an Excel File'

            color1 = 'danger'

            cal_data = no_update

    except Exception as e:

        print(e)

        error1 = 'There was an error processing this file'

        color1 = 'danger'

        cal_data = no_update


    return cal_data, error1, color1, True



# Call back to update upload file checklist once files are selected
@app.callback(
        Output(component_id='submit_files', component_property='value'),
        Output(component_id='filename_filepath', component_property='data'),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id = 'dropdown_BARN_update', component_property ='n_clicks'),
        Input(component_id='dropdown_BARN_clear', component_property='n_clicks'),
        State(component_id='submit_files', component_property='value'),
        State(component_id='filename_filepath', component_property='data'),
        prevent_initial_call = True)

def update_file_to_upload_checklist(n_clicks, n_clicks2, filepath1, filename_filepath_data):


    if ctx.triggered_id == 'dropdown_BARN_update':

        if filepath1 is None or filepath1 == '':
            error = 'No Filepath Inputted. Please Check'
            color1 = 'danger'
            open1 = True
            filepath_input = ''
            filename_filepath_data = no_update

        else:

            filepath2 = filepath1.replace('"', "")
            filepath = os.path.normpath(filepath2)

            filename1 = os.path.basename(filepath)
            filename = os.path.splitext(filename1)[0]

            if os.path.isfile(filepath)==False:
                error = 'Please Check Filepath'
                color1 = 'danger'
                open1 = True
                filepath_input = no_update
                filename_filepath_data = no_update

            elif os.path.splitext(filename1)[1] != '.txt':
                    error = ' Please upload .txt files'
                    color1 = 'danger'
                    open1 = True
                    filepath_input = ''
                    filename_filepath_data = no_update

            else:

                if filename_filepath_data is None:
                    filename_filepath_data = [[filename],[filepath]]
                    error = filename + ' added'
                    color1 = 'success'
                    open1 = True
                    filepath_input = ''

                else:

                    combined_filenames = filename_filepath_data[0].copy()
                    combined_filepaths = filename_filepath_data[1].copy()
                    repeated_filename = []

                    for value in combined_filenames:
                        if filename == value:
                            repeated_filename.append(filename)

                    if repeated_filename != []:
                        error = filename + ' already exists. Please check'
                        color1 = 'danger'
                        open1 = True
                        filepath_input = no_update
                        filename_filepath_data = no_update
                        repeated_filename = []

                    else:
                        combined_filenames.append(filename)
                        combined_filepaths.append(filepath)
                        error = filename + ' added'
                        color1 = 'success'
                        open1 = True
                        filepath_input = ''
                        filename_filepath_data = [combined_filenames, combined_filepaths]

        return filepath_input, filename_filepath_data, error, color1, open1

    else:
        raise PreventUpdate



@app.callback(
        Output(component_id='submit_files', component_property='value', allow_duplicate=True),
        Output(component_id='filename_filepath', component_property='clear_data', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id='dropdown_BARN_clear', component_property='n_clicks'),
        prevent_initial_call = True)

def clear_upload(n_clicks):

    if ctx.triggered_id == 'dropdown_BARN_clear':

        filepath_input = ''
        error = 'Upload Files Cleared'
        color1 = 'primary'
        open1 = True
        clear_filename_filepath_data = True

        return filepath_input, clear_filename_filepath_data, error, color1, open1


@ app.callback(
    [
    Output(component_id='filestorage', component_property='data', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
    Output(component_id='filename_filepath', component_property='data', allow_duplicate=True),
    Output(component_id='Loading_variable_Process', component_property='data', allow_duplicate=True)],
    [Input(component_id='newfile', component_property='n_clicks'),
    State(component_id='filename_filepath', component_property='data'),
    State(component_id="Cal_storage", component_property='data'),
    State(component_id='Sample_rate', component_property='value'),
    State(component_id='filestorage', component_property='data'),
    State(component_id="upload_file_checklist", component_property='value'),
    State(component_id="Workspace_store", component_property='data')
     ],
)

def Analyse_content(n_clicks,filename_filepath_data, cal_data, SF, file_data, filenames, Workspace_data):

    # Check if the "newfile" button was clicked
    if "newfile" == ctx.triggered_id:

        # Initialise data dictionary if it is None
        if file_data is None:
            file_data = [[],[],[],[],[]]

        # Check if no files were uploaded
        if filenames is None or filenames == []:

            error = 'No Files Selected For Upload'
            color = "danger"
            open1 = True
            filename_filepath_data = no_update
            # Return the same data if no files were uploaded
            file_data = no_update

        elif SF is None or SF == 0:

            error = 'No Sample Rate Selected'
            color = "danger"
            open1 = True
            filename_filepath_data = no_update
            # Return the same data if no files were uploaded
            file_data = no_update

        elif Workspace_data is None:
            error = 'No Workspace Selected'
            color = "danger"
            open1 = True
            filename_filepath_data = no_update
            # Return the same data if no files were uploaded
            file_data = no_update

        else:

            Oldfilenames = file_data[0] # Get existing file names
            old_dtype_shape = file_data[1]
            Old_calData = file_data[2]
            Old_SF = file_data[3]
            Old_filepath = file_data[4]

            combined_filenames = Oldfilenames.copy() # Make a copy of existing file names
            combined_dtype_shape = old_dtype_shape.copy()
            combined_CalData = Old_calData.copy() # Make a copy of existing file names
            combined_SF = Old_SF.copy()
            combined_filepath = Old_filepath.copy() # Make a copy of existing file names



            new_value = [] # List of uploaded file names which aren't repeated
            repeated_value = [] # List of repeated file names
            error_file = [] # List of files with invalid formats

            # Loop through uploaded files and process them

            def save_array_memmap(array, filename, folder_path):
                filepath = os.path.join(folder_path, filename)
                dtype = str(array.dtype)
                shape = array.shape
                array_memmap = np.memmap(filepath, dtype=dtype, shape = shape, mode='w+')
                array_memmap[:] = array[:]
                del array_memmap
                return shape, dtype

            def get_unique_path(base_path, name):
                counter = 1
                new_name = name

                while os.path.exists(os.path.join(base_path, new_name)):
                    new_name = f"{name} ({counter})"
                    counter += 1

                return os.path.normpath(os.path.join(base_path, new_name))


            for i, value in enumerate(filenames):
                # Check if the file name is already in the combined list
                if value not in combined_filenames:
                    try:

                        Barn_data = cal_velocity(filename_filepath_data[1][i], cal_data[1], SF)

                        Workspace_Path = os.path.join(Workspace_data, 'Cached_Files')

                        # Check if the folder exists
                        if not os.path.exists(Workspace_Path):
                            # Create the folder
                            os.mkdir(Workspace_Path)

                        file_path = get_unique_path(Workspace_Path, value)

                        os.makedirs(file_path, exist_ok=True)


                        save_array_memmap(Barn_data['Ux'], 'Ux.dat', file_path)
                        save_array_memmap(Barn_data['Uy'], 'Uy.dat', file_path)
                        save_array_memmap(Barn_data['Uz'], 'Uz.dat', file_path)
                        save_array_memmap(Barn_data['U1'], 'U1.dat', file_path)
                        shape_dtype = save_array_memmap(Barn_data['t'], 't.dat', file_path)

                        new_value.append(value)
                        combined_filenames.append(value)
                        combined_dtype_shape.append(shape_dtype)
                        combined_CalData.append(cal_data[0][0])
                        combined_SF.append(SF)
                        combined_filepath.append(file_path)

                    # If there's an error processing the file, add it to the error list
                    except Exception as e:
                        print('cal' + e)
                        error_file.append(value)
                else:

                    repeated_value.append(value)

            file_data = [combined_filenames, combined_dtype_shape, combined_CalData, combined_SF, combined_filepath]


            upload_filename = filename_filepath_data[0]
            upload_filepath = filename_filepath_data[1]

            # Delete selected data
            for value in filenames:
                i = upload_filename.index(value)
                upload_filename.remove(value)
                del upload_filepath[i]

            filename_filepath_data = [upload_filename, upload_filepath]


            # If there are errors, return error messages
            if repeated_value != [] or error_file != []:

                error_list_complete = repeated_value + error_file

                if new_value == []:

                    error_start = 'There was an error processing all files: \n ' \
                    '(' + ', '.join(error_list_complete) + ').'

                else:

                    error_start = 'There was an error processing files: \n ' \
                                   '(' + ', '.join(error_list_complete) + ').'

                error_repeat = ' Please check files: ' \
                               '(' + ', '.join(repeated_value) + ') are not repeated.'

                error_process = ' Please check: \n' \
                            '(' + ', '.join(error_file) + ') for errors.'



                # If there are errors in files and repeated files
                if repeated_value != [] and error_file != []:
                    error = error_start + error_repeat + error_process

                # If there are errors in files
                elif error_file != [] and repeated_value == []:
                    error = error_start + '\n' + error_process

                # If there are errors in files
                elif error_file == [] and repeated_value != []:
                    error = error_start + '\n' + error_repeat

                color = "danger"
                open1 = True

            else:

                # If no errors display success message
                error = ', '.join(new_value) + ' processed'

                color = "success"

                open1 = True

        loading_variable = 'done'

        return file_data, error, color, open1, filename_filepath_data, loading_variable

@ app.callback(
    Output(component_id="big_t_TI", component_property='value', allow_duplicate=True),
    Output(component_id="small_t_TI", component_property='value', allow_duplicate=True),
    Input(component_id="small_t_TI", component_property='value'),
    Input(component_id="big_t_TI", component_property='value'),
        prevent_initial_call=True)

def update_In(small_val, large_val):
    # If both inputs are None, prevent update
    if large_val is None or small_val is None:
        raise PreventUpdate

    # If large input is less than small input, set large input equal to small input
    if large_val < small_val:
        large_val = small_val

    # Return the updated large and small input values
    return large_val, small_val

@app.callback(
    Output(component_id='TI_Table', component_property='data', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='children', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='color', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='Clear_Table', component_property='n_clicks'),
    prevent_initial_call=True)

def clear_table(n_clicks):

    if "Clear_Table" == ctx.triggered_id:

        error = 'TABLE CLEARED'

        error_col = 'success'

        table_data = []

    return table_data, error, error_col, True

# Callback to analyse and update TI table
@ app.callback(
    [Output(component_id='TI_Table', component_property='data', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='children', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='color', allow_duplicate=True),
    Output(component_id='TI_alert', component_property='is_open', allow_duplicate=True),
    Output(component_id='Loading_variable_Table', component_property='data', allow_duplicate=True)],
    [Input(component_id='TI_btn_download', component_property='n_clicks'),
    State(component_id='filestorage', component_property='data'),
    State(component_id='DataSet_TI', component_property='value'),
    State(component_id="small_t_TI", component_property='value'),
    State(component_id="big_t_TI", component_property='value'),
    State(component_id='TI_Table', component_property='data'),
    State(component_id='TI_Table', component_property='columns')])


def TI_caluculate(n_clicks, file_data, chosen_file, small_TI, big_TI, table_data, column_data):

    if "TI_btn_download" == ctx.triggered_id:

        if chosen_file is None:

            error = 'TURBULENCE INTENSITY NOT CALCULATED. Please check you have selected a dataset'

            error_col = 'danger'

            table_data = no_update

        else:

            if small_TI == big_TI and small_TI is not None and big_TI is not None:

                error = 'TURBULENCE INTENSITY NOT CALCULATED. Please check that the inputted time range is correct'

                error_col = 'danger'

                table_data = no_update

            else:


                def load_array_memmap(filename, folder_path, dtype, shape, row_numbers):
                    filepath = os.path.join(folder_path, filename)
                    mapped_data = np.memmap(filepath, dtype=dtype, mode='r', shape=shape)

                    if row_numbers == 'all':
                        loaded_data = mapped_data[:]
                    else:
                        loaded_data = mapped_data[row_numbers]

                    return loaded_data

                i = file_data[0].index(chosen_file)


                shape_dtype = file_data[1][i]

                shape, dtype = shape_dtype

                file_path = file_data[4][i]

                t = load_array_memmap('t.dat',file_path, dtype= dtype, shape= shape[0], row_numbers = 'all')

                max1 = np.amax(t)
                min1 = np.amin(t)

                # Error messages
                smallt_error = 'TURBULENCE INTENSITY CALCULATED. The data has been cut to the minimum time limit because the inputted time ' \
                                                                    'is outside the available range. Please adjust your time limit accordingly.'


                bigt_error = 'TURBULENCE INTENSITY CALCULATED. The data has been cut to the maximum time limit because the inputted time ' \
                                                                    'is outside the available range. Please adjust your time limit accordingly.'

                both_t_error ='TURBULENCE INTENSITY CALCULATED. The data has been cut to the minimum and maximum time limit because the inputted times ' \
                                                                   'are outside the available range. Please adjust your time limit accordingly.'

                both_t_NO_error = 'TURBULENCE INTENSITY CALCULATED'

                # Cut data based on conditions
                if small_TI is None and big_TI is None:
                    big_TI = max1
                    small_TI = min1
                    error = both_t_error

                elif small_TI is None and big_TI is not None:
                    small_TI = min1
                    error = smallt_error

                elif big_TI is None and small_TI is not None:
                    big_TI = max1
                    error = bigt_error

                else:

                    if small_TI < min1 and big_TI > max1:
                        small_TI = min1
                        big_TI = max1
                        error = both_t_error

                    elif small_TI < min1:
                        small_TI = min1
                        error = smallt_error


                    elif big_TI > max1:
                        big_TI = max1
                        error = bigt_error

                    else:
                        error = both_t_NO_error

                mask = (t >= small_TI) & (t <= big_TI)
                error_col = 'primary'
                row_numbers = np.where(mask)[0].tolist()

                ux = load_array_memmap('Ux.dat',file_path, dtype= dtype, shape= shape[0], row_numbers = row_numbers)
                uy = load_array_memmap('Uy.dat',file_path, dtype= dtype, shape= shape[0], row_numbers = row_numbers)
                uz = load_array_memmap('Uz.dat',file_path, dtype= dtype, shape= shape[0], row_numbers = row_numbers)

                TI, U1, Ux, Uy, Uz = calculate_turbulence_intensity(ux, uy, uz)


                if table_data is None:
                    table_data = []

                i = file_data[0].index(chosen_file)

                new_data = [
                    {
                    'FileName': chosen_file,
                    'CalFile': file_data[2][i],
                    'SF': file_data[3][i],
                    'Time_1': round(small_TI,2),
                    'Time_2': round(big_TI,2),
                    'Ux': round(Ux,6),
                    'Uy': round(Uy,6),
                    'Uz': round(Uz,6),
                    'U1': round(U1,6),
                    'TI': round(TI,6),
                }

                ]

                table_data.append({c['id']: new_data[0].get(c['id'], None) for c in column_data})

        Loading_variable = 'done'

        return table_data, error, error_col, True, Loading_variable

#
# data = [combined_filenames, combined_dtype_shape, Workspace_data, SF, cal_data[0][0]]

# # Callback which updates the graph based on graph options
# @app.callback(
#         Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
#         Input(component_id = 'filestorage', component_property = 'data'),
#         Input(component_id = 'File', component_property = 'value'),
#         Input(component_id = 'Vect', component_property = 'value'),
#         Input(component_id='line_thick', component_property='value'),
#         prevent_initial_call = True)
#
# def update_graph(data, user_inputs, user_inputs1, line_thick):
#
#
#     # If no input do not plot graphs
#     if user_inputs == [] or user_inputs1 == []:
#
#         error1 = no_update
#
#         color = no_update
#
#         open1 = False
#
#         fig = {}
#
#         min_sl = 1
#
#         max_sl = 10
#
#         value =[1, 10]
#
#     else:
#
#         folder_path = r'C:\Users\lauri\OneDrive\Documents (1)\University\Year 3\Semester 2\BARNACLE\Example Data\Workspace'
#         # Plotting graphs
#
#         max1 = []
#
#         min1 = []
#
#         fig = go.Figure()
#
#         current_names = []
#
#         # If user inputs are changed reset time slider and plot full data
#         if "File" == ctx.triggered_id or "Vect" == ctx.triggered_id:
#
#             for user_input in user_inputs:
#                 for user_input1 in user_inputs1:
#
#                     # Create a new file name with the current 'value' and the '.npz' extension
#                     filename2 = user_input + '.npz'
#
#                     # Join the folder path with the file name
#                     file_path = os.path.join(folder_path, filename2)
#
#                     filename2 = user_input +'.npz'
#
#                     file_path = os.path.join(folder_path, filename2)
#
#                     Barn_data = np.load(file_path)
#
#
#                     V = Barn_data[user_input1]
#                     t = Barn_data['t']
#
#                     # Plotting data
#                     fig.add_trace(go.Scatter(x=t, y=V, mode='lines',
#                                             line=dict(
#                                             width=line_thick),
#                                             name=f"{user_input}{' '}{user_input1}"))
#                     # Creating a list of current legend names
#                     current_names.append(f"{user_input}{' '}{user_input1}")
#
#
#
#         return fig,

# Callback to update legend or title data
@app.callback(
     Output(component_id='New_name', component_property='value'),
     Output(component_id='legend_Data', component_property='data'),
     Output(component_id='title_Data', component_property='data'),
     Input(component_id="dropdown_legend_update", component_property='n_clicks'),
     Input(component_id="dropdown_title_update", component_property='n_clicks'),
     Input(component_id="dropdown_clear", component_property='n_clicks'),
     State(component_id='New_name', component_property='value'),
     prevent_initial_call = True)

def update_leg_title_data(n_click, n_clicks1, n_clicks2,  name_input):

    # If legend update button is pressed
    if ctx.triggered_id == 'dropdown_legend_update':
        # Update legend data
        legend_data = name_input
        # No update to title data or name input
        title_data = no_update
        name_input = no_update
    # If title update button is pressed
    elif ctx.triggered_id == 'dropdown_title_update':
        # Update title data
        title_data = name_input
        # No update to legend data or name input
        name_input = no_update
        legend_data = no_update

    # If clear dropdown pressed clear input box
    elif ctx.triggered_id == 'dropdown_clear':
        # Clear title and legend data
        title_data = None
        legend_data = None
        # Clear input box
        name_input = ''

    else:
        # Else no update to any values
        title_data = no_update
        name_input = no_update
        legend_data = no_update

    # Return name, legend and title data
    return name_input, legend_data, title_data

# Callback which syncs the all button of the upload checklist. If all is clicked all files will be selected.
# If all files are clicked all will be selected
@app.callback(
        Output(component_id="upload_file_checklist", component_property='value'),
        Output(component_id='all_upload_file_checklist', component_property='value'),
        Input(component_id="upload_file_checklist", component_property='value'),
        Input(component_id='all_upload_file_checklist', component_property='value'),
        Input(component_id='filename_filepath', component_property='data'),
        prevent_initial_call=True
        )

def file_upload_sync_checklist(upload_file_check, all_upload_file_check, filename_filepath_data):
    # Prevent update if there are no file names
    if filename_filepath_data is None:
        raise PreventUpdate

    # Split up the triggered callback
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if input_id == "upload_file_checklist":
        # If the upload file checklist input triggered the callback, update the all upload file checklist
        all_upload_file_check = ["All"] if set(upload_file_check) == set(filename_filepath_data[0]) else []
    else:
        # If the all upload file checklist input triggered the callback, update the upload file checklist
        upload_file_check = filename_filepath_data[0] if all_upload_file_check else []

    # Return the updated upload file checklist and all upload file checklist
    return upload_file_check, all_upload_file_check


# Callback which syncs the all button of the clear file checklist. If all is clicked all files will be selected.
# If all files are clicked all will be selected
@app.callback(
        Output(component_id="clear_file_checklist", component_property='value'),
        Output(component_id='all_clear_file_checklist', component_property='value'),
        Input(component_id="clear_file_checklist", component_property='value'),
        Input(component_id='all_clear_file_checklist', component_property='value'),
        Input(component_id='filestorage', component_property='data'),
        prevent_initial_call=True
        )

def file_clear_sync_checklist(clear_file_check, all_clear_check, data):
    # If stored data is none prevent update
    if data is None:
        raise PreventUpdate

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

    # Return the updated clear file checklist and all clear checklist
    return clear_file_check, all_clear_check

# Callback which syncs the all button of the vel checklist. If all is clicked all options will be selected.
# If all options are clicked all will be selected
@app.callback(
    Output(component_id="vel_checklist", component_property='value'),
    Output(component_id='all_vel_checklist', component_property='value'),
    Input(component_id="vel_checklist", component_property='value'),
    Input(component_id='all_vel_checklist', component_property='value'),
    prevent_initial_call=True
)

def vel_sync_checklist(vel_check, all_vel_checklist):
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    vel_type = ['t','U1','Ux', 'Uy', 'Uz']

    if input_id == "vel_checklist":
        # If the velocity checklist input triggered the callback, update the all velocity checklist
        all_vel_checklist = ["All"] if set(vel_check) == set(vel_type) else []
    else:
        # If the all velocity checklist input triggered the callback, update the velocity checklist
        vel_check = vel_type if all_vel_checklist else []

    # Return the updated velocity checklist and all velocity checklist
    return vel_check, all_vel_checklist


# Callback which updates dropdowns
@app.callback(
    Output(component_id="File", component_property='options'),
    Output(component_id='Vect', component_property='options'),
    Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="vel_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="clear_file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="upload_file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id='DataSet_TI', component_property='options'),
    Input(component_id='filestorage', component_property='data'),
    Input(component_id='filename_filepath', component_property='data'),
    prevent_initial_call=True)

def update_dropdowns1(data, filename_filepath_upload_data):

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
    return file_dropdown_options, vect_options, file_checklist, vel_checklist, clear_file_check,upload_file_checklist, DataDrop_TI


# Call back which updates the download time range to prevent error
@app.callback(
        Output(component_id="big_t", component_property='value', allow_duplicate=True),
        Output(component_id="small_t", component_property='value', allow_duplicate=True),
        Input(component_id="small_t", component_property='value'),
        Input(component_id="big_t", component_property='value'),
        prevent_initial_call=True)

def update_In(small_val, large_val):
    # If both inputs are None, prevent update
    if large_val is None and small_val is None:
        raise PreventUpdate

    # If large input is None, set it to 0
    if large_val is None:
        large_val = 0

    # If small input is None, set it to 0
    if small_val is None:
        small_val = 0

    # If large input is less than small input, set large input equal to small input
    if large_val < small_val:
        large_val = small_val

    # Return the updated large and small input values
    return large_val, small_val

# Callback for download button
@app.callback(
        Output(component_id='Download_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='Loading_variable_Download', component_property='data'),
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

    # If download button is pressed
    if "btn_download" == ctx.triggered_id:

        try:

            Download_Path = os.path.join(Workspace_data, 'Downloads')

            # Check if the folder exists
            if not os.path.exists(Download_Path):
                # Create the folder
                os.mkdir(Download_Path)

            # If no file selected display error message
            if file is None:

                error = 'No file selected'

                error_col = 'danger'

            # If quantity is not picked
            elif vector_value == [] or vector_value is None:

                error = 'No data selected'

                error_col = 'danger'

            else:

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

                max1 = np.amax(t)
                min1 = np.amin(t)

                # Error messages
                smallt_error = 'DATA DOWNLOADED. The data has been cut to the minimum time limit because the inputted time ' \
                               'is outside the available range. Please adjust your time limit accordingly.'

                bigt_error = 'DATA DOWNLOADED. The data has been cut to the maximum time limit because the inputted time ' \
                             'is outside the available range. Please adjust your time limit accordingly.'

                both_t_error = 'DATA DOWNLOADED. The data has been cut to the minimum and maximum time limit because the inputted times ' \
                               'are outside the available range. Please adjust your time limit accordingly.'

                both_t_NO_error = 'DATA DOWNLOADED'

                # Cut data based on conditions
                if smallt is None and bigt is None:
                    bigt = max1
                    smallt = min1
                    error = both_t_error

                elif smallt is None and bigt is not None:
                    smallt = min1
                    error = smallt_error

                elif bigt is None and smallt is not None:
                    bigt = max1
                    error = bigt_error

                else:

                    if smallt < min1 and bigt > max1:
                        smallt = min1
                        bigt = max1
                        error = both_t_error

                    elif smallt < min1:
                        bigt = min1
                        error = smallt_error


                    elif bigt > max1:
                        bigt = max1
                        error = bigt_error

                    else:
                        error = both_t_NO_error

                mask = (t >= smallt) & (t <= bigt)
                error_col = 'primary'
                row_numbers = np.where(mask)[0].tolist()

                for vector in vector_value:
                    if vector == 't':
                        numpy_vect_data.append(t)
                    elif vector != 't':
                        numpy_vect_data.append(
                            load_array_memmap(vector + '.dat', file_path, dtype=dtype, shape=shape[0],
                                              row_numbers=row_numbers))

                # Concatenate the arrays vertically
                concatenated_array = np.column_stack(numpy_vect_data)

                concatenated_array1 = np.append([vector_value],concatenated_array,  axis=0)

                # Assigning filenames
                if selected_name is None or selected_name == '':
                    filename = file
                else:
                    filename = selected_name

                def get_unique_filename(base_path, name):
                    counter = 1
                    new_name = name

                    while os.path.isfile(os.path.join(base_path, new_name + '.csv')):
                        new_name = f"{name} ({counter})"
                        counter += 1

                    return os.path.normpath(os.path.join(base_path, new_name))

                new_filename_path = get_unique_filename(Download_Path, filename)

                # Save the concatenated array as a CSV file
                np.savetxt(new_filename_path + '.csv', concatenated_array1, delimiter=",", fmt="%s")

        except Exception as e:

            print(e)

            # If any error display message
            error = 'ERROR'
            error_col = 'danger'

        Loading_variable = 'done'

        return error, error_col, True, Loading_variable



# Callback to clear data
@app.callback(
        Output(component_id='Velocity_Graph', component_property='figure', allow_duplicate=True),
        Output(component_id="File", component_property='options', allow_duplicate=True),
        Output(component_id='Vect', component_property='options', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
        Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='filestorage', component_property='data', allow_duplicate=True),
        Output(component_id="filestorage", component_property='clear_data', allow_duplicate=True),
        Output(component_id='filename_filepath', component_property='clear_data'),
        Output(component_id="clear_file_checklist", component_property='value', allow_duplicate=True),
        Output(component_id="File", component_property='value', allow_duplicate=True),
        Output(component_id='Vect', component_property='value', allow_duplicate=True),
        Output(component_id="upload_file_checklist", component_property='value', allow_duplicate=True),
        Input(component_id='clear_files', component_property='n_clicks'),
        State(component_id='filestorage', component_property='data'),
        State(component_id="clear_file_checklist", component_property='value'),
        State(component_id="all_clear_file_checklist", component_property='value'),
        prevent_initial_call=True)

def clear_files( n_clicks, maindata, whatclear, allclear):

    # If the clear files button is pressed, prevent update
    if "clear_files" != ctx.triggered_id:
        raise PreventUpdate

    # Clear figure
    fig = {}

    # Clear upload data
    submit_val_check = []
    upload_filename = []
    upload_contents = []
    # Clear selected values
    clear_val = []
    file_drop_val = []
    vect_drop_val = []

    # If no files selected display error message
    if allclear == ['All'] and len(whatclear) == 0:

        # display bad error message
        error = 'No files deleted'
        color = "danger"

        # No update to new main data
        newmaindata = no_update

        # Clear all data
        clear_data_main = True

        # Make the file drop options and quantity options empty
        file_drop_opt = []
        vect_opt = []

        # Open error message
        open1 = True

    elif allclear == [] and len(whatclear) == 0:

        # display bad error message
        error = 'No files deleted'
        color = "danger"

        # No update to new main data
        newmaindata = no_update

        # Clear all data
        clear_data_main = True
        clear_data_file = True
        # Make the file drop options and quantity options empty
        file_drop_opt = no_update
        vect_opt = no_update

        # Open error message
        open1 = True

    elif allclear == ['All'] and len(whatclear) > 0:

        # display good error message
        error = 'All files cleared'
        color = "success"

        # No update to new main data
        newmaindata = no_update
        clear_data_file = True
        # Clear all data
        clear_data_main = True

        # Make the file drop options and quantity options empty
        file_drop_opt = []
        vect_opt = []

        # Open error message
        open1 = True


    # If 1 or more files being deleted
    elif len(whatclear) >= 1 and allclear != ['All']:


        # store = maindata[0]
        #
        # # Delete selected data
        # for what in whatclear:
        #     del prb[what]
        #     del df1[what]
        #     df2.remove(what)

        # Assign new data
        newmaindata = [df1, df2]

        # Display error message
        error = ', '.join(whatclear) + ' deleted'
        color = "success"

        # Do not clear main data
        clear_data_main = False
        clear_data_file = True
        # No option to graph options
        file_drop_opt = no_update
        vect_opt = no_update

        # Open error message
        open1 = True



    # Return required values
    return fig, file_drop_opt, vect_opt, error, color, open1, newmaindata, clear_data_main , clear_data_file, clear_val, file_drop_val, vect_drop_val, submit_val_check




# Run app
if __name__== '__main__':
    app.run_server(debug=True)

