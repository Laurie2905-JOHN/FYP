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

def cal_velocity(contents, file_names, cal_data, SF):

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
        prb[file_name]['t'] = np.linspace(0, prb[file_name]['raw'].shape[0] / SF, prb[file_name]['raw'].shape[0]);

        # Taking data needed
        prb_final = {'Ux': {}}
        prb_final = {'Uy': {}}
        prb_final = {'Uz': {}}
        prb_final = {'U1': {}}
        prb_final = {'t': {}}

        prb_final['U1'] = prb[file_name]['U1']
        prb_final['Ux'] = prb[file_name]['Ux']
        prb_final['Uy'] = prb[file_name]['Uy']
        prb_final['Uz'] = prb[file_name]['Uz']
        prb_final['t'] = prb[file_name]['t']

    return prb_final

cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))

# Create the Dash app object
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
                ])
            ])
        ]),


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
            html.H5('Upload BARNACLE Files', className='center-text'),
            width=12,
            className="text-center"
        ),

        # Horizontal line
        dbc.Col(
            html.Hr(),
            width=12
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

        # Column for file selection/upload
        dbc.Col([
            dbc.Stack([

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
                multiple=True
            ),

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
                    "Upload Selected Files",
                    id='newfile',
                    outline=True,
                    color="primary",
                    className="me-1",
                    n_clicks=0
                ),

                dbc.Input(id="Sample_rate", min=0, type="number", placeholder="Enter Sample Frequency", debounce=True),

                html.Label("Select files to upload"),

                # Checkbox for uploading all files
                dbc.Checklist(
                    ["All"], [], id="all_upload_file_checklist", inline=True
                ),

                # Checkbox for selecting individual files to upload
                dbc.Checklist(value=[], id="upload_file_checklist", inline=True),
            ], gap=2),
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
            ], gap=2),
            width=3
        ),

    ], align='center', justify='evenly'),

    dbc.Row([
        dbc.Col(
            html.Hr(),
            width=12
        ),  # Horizontal rule to separate content
    ], className='mb-2'),






    # # Components for storing and downloading data
    dcc.Download(id="download"),
    dcc.Store(id='legend_Data', storage_type='memory'),
    dcc.Store(id='title_Data', storage_type='memory'),
    dcc.Store(id='filestorage', storage_type='session'),
    dcc.Store(id='Cal_storage', storage_type='local'),

])

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

            error1 = 'File Uploaded Successfully'

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


@app.callback(
        Output(component_id="upload_file_checklist", component_property='options', allow_duplicate=True),
        Input(component_id = 'submit_files', component_property ='filename'),
        prevent_initial_call = True)

def file_checklist(file_names):

    upload_file_checklist = file_names

    return upload_file_checklist



# Callback to analyse and update data
@app.callback(
    Output(component_id='filestorage', component_property='data'),
    Output(component_id='ClearFiles_alert', component_property='children', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='color', allow_duplicate=True),
    Output(component_id='ClearFiles_alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='newfile', component_property='n_clicks'),
    State(component_id="Cal_storage", component_property='data'),
    State(component_id='Sample_rate', component_property='value'),
    State(component_id='filestorage', component_property='data'),
    State(component_id = 'submit_files',component_property = 'contents'),
    State(component_id="upload_file_checklist", component_property='value'),
    prevent_initial_call = True)


def content(n_clicks,cal_data, SF, data, contents, filenames):

    # Check if the "newfile" button was clicked
    if "newfile" == ctx.triggered_id:

        # Initialise data dictionary if it is None
        if data is None:
            data = [{}, []]

        # Check if no files were uploaded
        if filenames is None or filenames == []:

            error = 'No Files Selected For Upload'
            color = "danger"
            open1 = True

            # Return the same data if no files were uploaded
            data = no_update

        elif SF is None or SF == 0:

            error = 'No Sample Rate Selected'
            color = "danger"
            open1 = True
            # Return the same data if no files were uploaded
            data = no_update


        else:

            prb = data[0] # Get existing data dictionary
            Oldfilenames = data[1] # Get existing file names

            combined_filenames = Oldfilenames.copy() # Make a copy of existing file names

            new_value = [] # List of uploaded file names which aren't repeated
            repeated_value = [] # List of repeated file names
            contain_text = [] # List of file names that don't have 'txt' in them
            error_file = [] # List of files with invalid formats

            # Loop through uploaded files and process them
            for i, value in enumerate(filenames):

                # Check for repeated file names
                if value in combined_filenames:
                    repeated_value.append(value)

                # Check for file names without 'txt' in them
                if 'txt' not in value:
                    contain_text.append(value)

                # Check if the file name is already in the combined list
                if value not in combined_filenames:
                    # Check if file name is not in the list of names that don't have 'txt' in them
                    if value not in contain_text:

                        # Try to process the file
                        try:
                            prb[value] = {value: {}}
                            prb[value] = cal_velocity(contents[i], filenames[i], cal_data[1], SF)
                            new_value.append(value)
                            combined_filenames.append(value)

                        # If there's an error processing the file, add it to the error list
                        except Exception as e:
                            print('cal' + e)
                            error_file.append(value)

            # If there are errors, return error messages
            if contain_text != [] or repeated_value != [] or error_file != []:

                data = [prb, combined_filenames, SF, cal_data[0][0]]

                color = "danger"
                open1 = True

                error_list_complete = contain_text + repeated_value + error_file

                if new_value != []:

                    error_start = 'Files (' + ', '.join(new_value) + ') uploaded.\n'
                    'There was an error processing files: \n ' \
                    '(' + ', '.join(error_list_complete) + ').'

                else:

                    error_start = 'There was an error processing files: \n ' \
                                   '(' + ', '.join(error_list_complete) + ').'

                error_repeat = ' Please check : ' \
                               '(' + ', '.join(repeated_value) + ') are not repeated.'

                error_txt = ' Please check the file type of: \n' \
                            '(' + ', '.join(contain_text) + '). '

                error_process = ' Please check: \n' \
                            '(' + ', '.join(error_file) + ') for errors.'

                # If all three errors are present
                if contain_text != [] and repeated_value != [] and error_file != []:
                    error = error_start + error_repeat + error_txt + error_process

                # If there are invalid file types and errors in files
                elif contain_text != [] and repeated_value == [] and error_file != []:
                    error = error_start + error_txt + error_process

                # If there are invalid file types and repeated files
                elif contain_text != [] and repeated_value != [] and error_file == []:
                    error = error_start + error_repeat + error_txt

                # If there are errors in files and repeated files
                elif contain_text == [] and repeated_value != [] and error_file != []:
                    error = error_start + error_repeat + error_process

                # If there are invalid file types
                elif contain_text != [] and repeated_value == [] and error_file == []:
                    error = error_start + '\n' + error_txt

                # If there are repeated files
                elif contain_text == [] and repeated_value != [] and error_file == []:
                    error = error_start + '\n' + error_repeat

                # If there are errors in files
                elif contain_text == [] and repeated_value == [] and error_file != []:
                    error = error_start + '\n' + error_process

            else:

                # If no errors display success message
                error = ', '.join(new_value) + ' uploaded'

                color = "success"

                open1 = True

                data = [prb, combined_filenames, SF, cal_data[0][0]]


        return data, error, color, open1




# Callback which updates dropdowns
@app.callback(
    Output(component_id="File", component_property='options'),
    Output(component_id='Vect', component_property='options'),
    Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="vel_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="clear_file_checklist", component_property='options', allow_duplicate=True),
    Input(component_id='filestorage', component_property='data'),
    prevent_initial_call=True)

def update_dropdowns(data):

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
        vect_options = ['Ux', 'Uy', 'Uz']
        file_dropdown_options = data[1]
        file_checklist = file_dropdown_options
        DataDrop_TI = file_dropdown_options
        clear_file_check = file_checklist
        vel_checklist = ['Ux', 'Uy', 'Uz', 't']

    # Return the updated dropdown options and checklists
    return file_dropdown_options, vect_options, file_checklist, vel_checklist, clear_file_check, DataDrop_TI



@app.callback(
        Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
        Output(component_id = 'time-range', component_property = 'min', allow_duplicate=True),
        Output(component_id = 'time-range', component_property = 'max', allow_duplicate=True),
        Output(component_id = 'time-range', component_property = 'value', allow_duplicate=True),
        Output(component_id='alert', component_property='children', allow_duplicate=True),
        Output(component_id='alert', component_property='color', allow_duplicate=True),
        Output(component_id='alert', component_property='is_open', allow_duplicate=True),
        Input(component_id = 'filestorage', component_property = 'data'),
        Input(component_id = 'File', component_property = 'value'),
        Input(component_id = 'Vect', component_property = 'value'),


def update_dropdowns(data, user_inputs, user_inputs1, time_input, line_thick, legend_data, title_data, leg, title ):

    # Check if data is not empty or None
    if data is None or data == []:
        raise PreventUpdate

    # If no input do not plot graphs
    if user_inputs == [] or user_inputs1 == []:

        error1 = no_update

        color = no_update

        open1 = False

        fig = {}

        min_sl = 1

        max_sl = 10

        value =[1, 10]

    else:

        # Plotting graphs
        df = data[0]

        max1 = []

        min1 = []

        fig = go.Figure()

        current_names = []

        # If user inputs are changed reset time slider and plot full data
        if "File" == ctx.triggered_id or "Vect" == ctx.triggered_id:

            for user_input in user_inputs:
                for user_input1 in user_inputs1:
                    V = df[user_input][user_input1]
                    t = df[user_input]['t']
                    # Calculating max and min of the datasets
                    max1.append(np.round(np.amax(t)))
                    min1.append(np.round(np.amin(t)))
                    # Plotting data
                    fig.add_trace(go.Scatter(x=t, y=V, mode='lines',
                                            line=dict(
                                            width=line_thick),
                                            name=f"{user_input}{' '}{user_input1}"))
                    # Creating a list of current legend names
                    current_names.append(f"{user_input}{' '}{user_input1}")

            # Calculating min and max of data
            min_sl = min(min1)
            max_sl = max(max1)
            # Value of slider being put at min and max positions
            value = [min_sl, max_sl]

        else:

            # If user inputs haven't changed
            for user_input in user_inputs:
                for user_input1 in user_inputs1:
                    V = np.array(df[user_input][user_input1])
                    t = np.array(df[user_input]['t'])
                    # Calculating max and min of data set
                    max1.append(np.round(np.amax(t)))
                    min1.append(np.round(np.amin(t)))
                    # Creating a mask to trim data for the time slider
                    mask = (t >= time_input[0]) & (t < time_input[1])
                    t2 = t[mask]
                    V2 = V[mask]
                    # Plotting data
                    fig.add_trace(go.Scatter(x=t2, y=V2, mode='lines',
                                            line=dict(
                                            width=line_thick),
                                            name=f"{user_input}{' '}{user_input1}"))
                    current_names.append(f"{user_input}{' '}{user_input1}")

            # No change in slider values
            value = time_input
            # Calculate max and min of data
            min_sl = min(min1)
            max_sl = max(max1)

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

        # Update legend

        if legend_data is None:
            # If no legend data

            # Turn legend off
            if leg == 'Off':
                fig.layout.update(showlegend=False)

                # No update to error messages
                error1 = no_update

                color = no_update

                open1 = False

            # Turn legend on
            elif leg == 'On':
                fig.layout.update(showlegend=True)

                # No update to error messages
                error1 = no_update

                color = no_update

                open1 = False

        else:
            # If there is legend data

            # If legend is off
            if leg == 'Off':
                fig.layout.update(showlegend=False)

                # No error message
                error1 = no_update

                color = no_update

                open1 = False

            elif leg == 'On':
                # Split string when there is a comma
                legend_name_list = legend_data.split(',')
                # Create empty dictionary for new legend name
                newname_result = {}

                # If the length of the list of new legend entries is eqaul to the number of lines
                if len(current_names) == len(legend_name_list):
                    # For each legend name create a dictionary of the current names and their replacements
                    for i, current_name in enumerate(current_names):
                        newnames = {current_name: legend_name_list[i]}
                        newname_result.update(newnames)
                    # Update legend names
                    fig.for_each_trace(lambda t: t.update(name=newname_result[t.name],
                                                          legendgroup=newname_result[t.name],
                                                          hovertemplate=t.hovertemplate.replace(t.name, newname_result[
                                                              t.name]) if t.hovertemplate is not None else None)
                                       )

                    fig.layout.update(showlegend=True)

                    # No error message
                    error1 = no_update

                    color = no_update

                    open1 = False

                else:

                    # If legend entries do not match display error message
                    error1 = 'Number of legend entries do not match'

                    color = 'danger'

                    open1 = True
        # If title data is empty
        if title_data is None:

            # Turn graph title off
            if title == 'Off':
                fig.layout.update(title='')
            # If title is on display original title
            elif title == 'On':
                fig.layout.update(title='Plot of ' + ', '.join(user_inputs) + ' Data')

        # If title data isn't empty
        else:

            # Turn graph title off
            if title == 'Off':
                fig.layout.update(title='')
            # Update title with legend data if legend is on
            elif title == 'On':
                fig.layout.update(title=title_data)
    # return figure, min and max values for slider, and error message
    return fig, min_sl, max_sl, value, error1, color, open1





# Run app
if __name__== '__main__':
    app.run_server(debug=True)

