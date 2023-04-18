# Import libs
from dash import Dash, dcc, Output, Input, ctx, State
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
# Ignore warning of square root of negative number
warnings.simplefilter(action='ignore', category=RuntimeWarning)


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
            html.H5('Upload/Clear Files', className='center-text'),
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
            dcc.Upload(
                id='submit_files',
                children=html.Div([
                    html.A('Select Files')
                ]),
                style={
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'solid',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '20px',
                    'width': '90%',
                },
                className="text-primary",
                # Allow multiple files to be uploaded
                multiple=True
            )
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

    dbc.Row(
        dbc.Col(
            html.H5('Download Files', className='center-text'),
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

    dbc.Row([
        dbc.Col([
            dbc.Stack([
                html.Label("Choose Data File"),  # Label for selecting data file

                dbc.RadioItems(id="file_checklist", inline=True),  # Radio buttons for selecting data file

                html.Label("Choose Quantity"),  # Label for selecting quantity of data to download

                dbc.Checklist(["All"], [], id="all_vel_checklist", inline=True),  # Checkbox to select all data

                dbc.Checklist(value=[], options=[], id="vel_checklist", inline=True),  # Checkbox to select specific data

                html.Label("Choose File Type"),  # Label for selecting file type

                dbc.RadioItems(
                    options=['CSV', 'Excel', '.txt'],
                    value='CSV',
                    id="type_checklist",
                    inline=True
                ),  # Radio buttons for selecting file type
            ], gap=2),
        ], width=6),

        dbc.Col([
            dbc.Stack([
                dbc.Button("Download", id="btn_download", size="lg"),  # Button for downloading selected data

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
        ], width=6),
    ], align='center', justify='evenly'),  # Row containing columns for selecting and downloading data files

    # # Components for storing and downloading data
    dcc.Download(id="download"),
    dcc.Store(id='legend_Data', storage_type='memory'),
    dcc.Store(id='title_Data', storage_type='memory'),
    dcc.Store(id='current_names', storage_type='memory'),
    dcc.Store(id='filestorage', storage_type='session'),

])


# Call back to update upload file checklist once files are selected
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
    State(component_id='filestorage', component_property='data'),
    State(component_id = 'submit_files',component_property = 'contents'),
    State(component_id="upload_file_checklist", component_property='value'),
    prevent_initial_call=True)

def content(n_clicks, data, contents, filenames):

    # Check if the function was triggered by a button click
    if n_clicks is None:
        raise PreventUpdate

    # Check if the "newfile" button was clicked
    if "newfile" == ctx.triggered_id:

        # Initialize data dictionary if it is None
        if data is None:
            data = [{}, []]

        # Check if no files were uploaded
        if filenames is None or filenames == [] or contents is None or contents == []:

            error = 'No files selected for upload'
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
                            prb[value] = cal_velocity(contents[i], filenames[i])
                            new_value.append(value)
                            combined_filenames.append(value)

                        # If there's an error processing the file, add it to the error list
                        except Exception:
                            error_file.append(value)

            # If there are errors, return error messages
            if contain_text != [] or repeated_value != [] or error_file != []:

                data = [prb, combined_filenames]

                color = "danger"
                open1 = True

                error_list_complete = contain_text + repeated_value + error_file

                error_start = 'There was an error processing files: \n ' \
                               '(' + ', '.join(error_list_complete) + ').'

                error_repeat = ' Please check that files are not repeated: \n ' \
                               '(' + ', '.join(repeated_value) + ').'

                error_txt = ' Please check the file type of: \n' \
                            '(' + ', '.join(contain_text) + '). '

                error_process = ' Please check the file for errors: \n' \
                            '(' + ', '.join(error_file) + '). '


                if contain_text != [] and repeated_value != [] and error_file != []:

                    error = error_start + '\n' + error_repeat + '\n' + error_txt + '\n' + error_process

                elif contain_text != [] and error_file != []:

                    error = error_start + '\n' + error_txt + '\n' + error_process

                elif error_file != [] and repeated_value != []:

                    error = error_start + '\n' + error_repeat + '\n' + error_txt

                elif contain_text != [] and repeated_value != []:

                    error = error_start + '\n' + error_repeat + '\n' + error_txt

                elif error_file != []:

                    error = error_start + '\n' + error_process

                elif contain_text != []:

                    error = error_start + '\n' + error_txt

                elif repeated_value != []:

                    error = error_start + '\n' + error_repeat

            else:

                # If no errors display success message
                error = ', '.join(new_value) + ' uploaded'

                color = "success"

                open1 = True

                data = [prb, combined_filenames]


        return data, error, color, open1

# Callback which syncs the all button of the upload checklist. If all is clicked all files will be selected.
# If all files are clicked all will be selected
@app.callback(
        Output(component_id="upload_file_checklist", component_property='value'),
        Output(component_id='all_upload_file_checklist', component_property='value'),
        Input(component_id="upload_file_checklist", component_property='value'),
        Input(component_id='all_upload_file_checklist', component_property='value'),
        State(component_id = 'submit_files', component_property ='filename'),
        prevent_initial_call=True
        )

def file_upload_sync_checklist(upload_file_check, all_upload_file_check, Uploaded_filenames):
    # Prevent update if there are no file names
    if Uploaded_filenames is None:
        raise PreventUpdate

    # Split up the triggered callback
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if input_id == "upload_file_checklist":
        # If the upload file checklist input triggered the callback, update the all upload file checklist
        all_upload_file_check = ["All"] if set(upload_file_check) == set(Uploaded_filenames) else []
    else:
        # If the all upload file checklist input triggered the callback, update the upload file checklist
        upload_file_check = Uploaded_filenames if all_upload_file_check else []

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
    file_options = data[1]

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

    vel_type = ['Ux', 'Uy', 'Uz', 't']

    if input_id == "vel_checklist":
        # If the velocity checklist input triggered the callback, update the all velocity checklist
        all_vel_checklist = ["All"] if set(vel_check) == set(vel_type) else []
    else:
        # If the all velocity checklist input triggered the callback, update the velocity checklist
        vel_check = vel_type if all_vel_checklist else []

    # Return the updated velocity checklist and all velocity checklist
    return vel_check, all_vel_checklist


# Callback which updates dropdowns of graph
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
    else:
        # If the data is not None, set the dropdown options and checklists accordingly
        vect_options = ['Ux', 'Uy', 'Uz']
        file_dropdown_options = data[1]
        file_checklist = file_dropdown_options
        clear_file_check = file_checklist
        vel_checklist = ['Ux', 'Uy', 'Uz', 't']

    # Return the updated dropdown options and checklists
    return file_dropdown_options, vect_options, file_checklist, vel_checklist, clear_file_check,


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

# Callback which updates the graph based on graph options
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
        Input(component_id = 'time-range', component_property = 'value'),
        Input(component_id='line_thick', component_property='value'),
        Input(component_id='legend_Data', component_property='data'),
        Input(component_id='title_Data', component_property='data'),
        Input(component_id='legend_onoff', component_property='value'),
        Input(component_id='title_onoff', component_property='value'), prevent_initial_call = True)

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

# Callback to update legend or title data
@app.callback(
     Output(component_id='New_name', component_property='value'),
     Output(component_id='legend_Data', component_property='data'),
     Output(component_id='title_Data', component_property='data'),
     Input(component_id='filestorage', component_property='data'),
     Input(component_id="dropdown_legend_update", component_property='n_clicks'),
     Input(component_id="dropdown_title_update", component_property='n_clicks'),
     Input(component_id="dropdown_clear", component_property='n_clicks'),
     State(component_id='New_name', component_property='value'),
     prevent_initial_call = True)

def update_leg_title_data(data, n_click, n_clicks1, n_clicks2,  name_input):

    # If data is none prevent update of app
    if data is None:
        raise PreventUpdate

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

# Callback for download button
@app.callback(
        Output(component_id="download", component_property='data', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='children', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='color', allow_duplicate=True),
        Output(component_id='Download_alert', component_property='is_open', allow_duplicate=True),
        Input(component_id="btn_download", component_property='n_clicks'),
        State(component_id="file_name_input", component_property='value'),
        State(component_id="small_t", component_property='value'),
        State(component_id="big_t", component_property='value'),
        State(component_id="vel_checklist", component_property='value'),
        State(component_id="vel_checklist", component_property='options'),
        State(component_id="file_checklist", component_property='value'),
        State(component_id="type_checklist", component_property='value'),
        State(component_id='filestorage', component_property='data'),
        prevent_initial_call=True)

def download(n_clicks, selected_name, smallt, bigt, vels, vel_opts, file, file_type, data):

    # If download button is pressed
    if "btn_download" == ctx.triggered_id:

        # If no file selected display error message
        if file is None:

            # Display error message and don't download anything
            text = no_update

            error1 = ['No data to download', 'danger']

        # If quantity is not picked
        if vels == [] or vels is None:

            # Display error message and don't download anything
            text = no_update

            error1 = ['No data to download', "danger"]

        # If required values are selected
        else:

            # Assign values
            prb = data[0]

            # Create a dictionary of all options
            dff = {file: {vel_opt: np.array(prb[file][vel_opt]) for vel_opt in vel_opts}}

            # Create a dictionary of options selected
            df = {file: {vel: [] for vel in vels}}

            # If a time range has been selected
            if smallt is not None or bigt is not None:
                # Assign variables and convert to numpy array
                t = np.array(dff[file]['t'])
                # Calculate min and  max
                max1 = np.amax(t)
                min1 = np.amin(t)

                # If both smallt and bigt isn't empty
                if smallt is not None and bigt is not None:

                    # Error messages
                    smallt_error = [
                    file_type + ' File Downloaded.\n' + 'The data has been cut to the minimum time limit because it\n'
                    'is outside the available range of raw time data. Please adjust your time limit accordingly.', 'danger']

                    bigt_error = [
                        file_type + ' File Downloaded.\n' + 'The data has been cut to the maximum time limit because it\n'
                        'is outside the available range of raw time data. Please adjust your time limit accordingly.', 'danger']

                    both_t_error = [
                        file_type + ' File Downloaded. Warning: Both minimum and maximum time limits are outside the range\n'
                                    'of available raw time data. The data has been trimmed to the minimum and maximum\n'
                                    'time limits. Please adjust your time limits accordingly.', 'primary']

                    both_t_NO_error = [file_type + ' File Downloaded.\n' + 'Data has been cut to the specified limits', 'primary']

                    # Cut data based on conditions
                    if smallt < min1 and bigt > max1:
                        error1 = both_t_error
                        mask = (t >= min1) & (t <= max1)

                    elif smallt < min1:
                        error1 = smallt_error
                        mask = (t >= min1)

                    elif bigt > max1:
                        error1 = bigt_error
                        mask = (t <= max1)

                    else:
                        mask = (t >= smallt) & (t < bigt)
                        error1 = both_t_NO_error

                # For selected options assign data
                for vel in vels:
                    df[file][vel] = dff[file][vel][mask]

            # If no time range selected
            else:

                # Assign data
                for vel in vels:
                    df[file][vel] = dff[file][vel]

                # Error message
                error1 = [file_type + ' File Downloaded', 'primary']

                open1 = True

            # If .txt is not in file name
            if file_type == '.txt':

                # Create an empty list
                list_all = []

                # Convert data to strings based on initial conditions
                if len(vels) == 1:
                    stacked = df[file][vels[0]]
                    list_all.append(stacked)
                    list_all = list_all[0]
                    str_all = np.array2string(list_all, separator=',\n', threshold=sys.maxsize)

                else:

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
                    list_all = np.concatenate(list_all, axis=1)
                    str_all = np.array2string(list_all, separator=',', threshold=sys.maxsize)

                # Filter data so it is in the correct format
                vels_str = ','.join(vels)
                str_all = vels_str + '\n' + str_all
                str_all = str_all.replace(' ', '')
                str_all = str_all.replace('],', '')
                str_all = str_all.replace(']]', '')
                str_all = str_all.replace('[[', '')
                str_all = str_all.replace('[', '')
                str_all = str_all.replace(']', '')

                # If no filename selected, assign filename
                if selected_name is None or selected_name == '':
                    value = file.split('.')
                    filenameTXT = value[0] + ".txt"

                # If filename is chosen, create filename string
                else:
                    filenameTXT = selected_name + ".txt"

                # Assign text file to save
                text = dict(content=str_all, filename = filenameTXT )

            # If chosen file type is excel or CSV
            if file_type == 'Excel' or 'CSV':

                # create an empty list to store dataframes
                pandaData = []

                # loop through each file and convert to dataframe
                for file, df in df.items():
                    dfff = pd.DataFrame(df)
                    pandaData.append(dfff)
                # concatenate all dataframes in the list
                PDdata = pd.concat(pandaData)

                # Assigning filenames
                if selected_name is None or selected_name == '':
                    value = file.split('.')
                    filename = value[0]
                else:
                    filename = selected_name

                if file_type == 'Excel':
                    ty = '.xlsx'
                    text = dcc.send_data_frame(PDdata.to_excel, filename + ty)

                if file_type == 'CSV':
                    ty = '.csv'
                    text = dcc.send_data_frame(PDdata.to_csv, filename + ty)

        return text, error1[0], error1[1], True,

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
        Output(component_id='submit_files', component_property='filename'),
        Output(component_id='submit_files', component_property='contents'),
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
    upload_filename = []
    upload_contents = []

    # If the all checklist is checked
    if allclear == ['All']:

        # display good error message
        error = 'All files cleared'
        color = "success"

        # No update to new main data
        newmaindata = no_update

        # Clear all data
        clear_data_main = True

        # Make the file drop options and quantity options empty
        file_drop_opt = []
        vect_opt = []

        # If no files selected display error message
        if len(whatclear) == 0:

            # display good error message
            error = 'No files deleted'
            color = "danger"

            # No update to new main data
            newmaindata = no_update

            # Clear all data
            clear_data_main = True

            # Make the file drop options and quantity options empty
            file_drop_opt = []
            vect_opt = []

    # If 1 or more files being deleted
    elif len(whatclear) >= 1:

        # Assign data
        df1 = maindata[0]
        df2 = maindata[1]

        # Delete selected data
        for what in whatclear:
            del df1[what]
            df2.remove(what)

        # Assign new data
        newmaindata = [df1, df2]

        # Display error message
        error = ', '.join(whatclear) + ' deleted'
        color = "success"

        # Do not clear main data
        clear_data_main = False

        # No option to graph options
        file_drop_opt = no_update
        vect_opt = no_update

    else:
        # No uodate to new data
        newmaindata = no_update

        # Display error message
        error = 'No files deleted as none were selected'
        color = "danger"

        # Do not clear main data
        clear_data_main = False

        # No option to graph options
        file_drop_opt = no_update
        vect_opt = no_update

    # Open error message
    open1 = True

    # Return required values
    return fig, file_drop_opt, vect_opt, error, color, open1, newmaindata, clear_data_main, upload_filename, upload_contents


# Run app
if __name__== '__main__':
    app.run_server(debug=True)

