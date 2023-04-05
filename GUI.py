#file_paths = ['C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 1.txt', 'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 2.txt']

#file_names = ['Example 1.txt', 'Example 2.txt']
from dash import Dash, dcc, Output, Input, ctx, State
from dash.dash import no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import sys
import dash_bootstrap_components as dbc
from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import numpy as np
import scipy.io as sio
from pathlib import Path, PureWindowsPath
import plotly.graph_objects as go
import base64
import datetime
import io
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import dash_mantine_components as dmc


def cal_velocity(contents, file_names):
    import base64
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
    prb_final = {}

    for i, file_name in enumerate(file_names):
        prb[file_name] = {'raw': {}}
        content_string = contents[i]
        decoded = base64.b64decode(content_string)
        decoded_str = decoded.removeprefix(b'u\xabZ\xb5\xecm\xfe\x99Z\x8av\xda\xb1\xee\xb8')
        lines = decoded_str.decode().split('\r\n')[:-1]
        prb[file_name]['raw'] = np.array([list(map(float, line.split(','))) for line in lines])
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

        # Taking data needed
        # Taking data needed
        prb_final[file_name] = {'Ux': {}}
        prb_final[file_name] = {'Uy': {}}
        prb_final[file_name] = {'Uz': {}}
        prb_final[file_name] = {'t': {}}
        prb_final[file_name]['Ux'] = prb[file_name]['Ux']
        prb_final[file_name]['Uy'] = prb[file_name]['Uy']
        prb_final[file_name]['Uz'] = prb[file_name]['Uz']
        prb_final[file_name]['t'] = prb[file_name]['t']


    return prb_final



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout of the app
app.layout = dbc.Container([


    dbc.Row([
        dbc.Col(
            html.H1("BARNACLE SENSOR ANALYSIS DASHBOARD",
                    className='text-center font-weight-bolder, mb-1'),
            width=12),
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='Velocity_Graph', figure={}),
            width=12),
    ]),

    dbc.Row([

        dbc.Col(
            html.H5('Graph Options', className="text-center"),
        width = 12),

        dbc.Col(
            html.Hr(),
        width = 12),

        dbc.Col(
        dbc.Label("Time Slider", className="text-center mb-1"),
            width = 12,),

        dbc.Col(
            dcc.RangeSlider(
            id='time-range',
            min=1,
            max=10,
            value=[1, 10],
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode='drag'
        ), width = 12, className="mb-2"),



]),

    dbc.Row([

    dbc.Col([

        dbc.Stack([

            dbc.Label("Data Options",
            className='text-center'),

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
        ], gap = 3)

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
            updatemode='drag'),

        html.Hr(),


dbc.Row([

    dbc.Col(

    dbc.Stack([

    dbc.Label("Title", className="text-start"),

    dbc.RadioItems(id='title_onoff', value='On', options=['On', 'Off'], inline=True)

    ]),

    ),
        dbc.Col(

        dbc.Stack([

            dbc.Label('Legend', className="text-start"),

            dbc.RadioItems(id='legend_onoff', value='On', options=['On', 'Off'], inline=True),

        ]),


        ),
    ])

        ]),
]),


    dbc.Col([

        dbc.Stack([

            dbc.Label("Update Title or Legend", class_name="center-text"),

        dbc.Stack([

        dbc.Stack([
            dbc.InputGroup([
                    dbc.DropdownMenu([
                    dbc.DropdownMenuItem("Update", id="dropdown_title_update"),
                    dbc.DropdownMenuItem("Clear", id="dropdown_title_clear"),
        ],
            label="Generate"),
            dcc.Input(
            id="New_Titlename",
            type='text',
            placeholder="Enter new title",
            debounce=True),
    ]),
            ], direction="horizontal"),

            ])

        ])
        ])




])
    ])







#
#

#

#         html.Div(className="mb-3"),
#     ]),
#

# ]),
# ],fluid=True),
#
#
# dbc.Row([
#     dbc.Col(
#     html.Hr(),
#     width = 12),
#     ], className = 'mb-2'),
#
# dbc.Tabs(
#     [
#         dbc.Tab(label="Upload", tab_id="Upload"),
#         dbc.Tab(label="Download", tab_id="Download"),
#     ],
#     id="tabs",
#     active_tab="Upload",
# ),
# html.Div(id="tab-content", className="p-4"),






    # Create a button for downloading data
#     html.Button("Update Title", id="btn_title_update", n_clicks=0),
#
# ]),
#
#     html.Div("", className="mb-2"),
#
#     dbc.Row([
#         html.Label("Legend"),
#
#     dcc.RadioItems(id='legend_onoff', value='On', options=['On', 'Off'], inline=True),


#
# dbc.Row(
#
#     dbc.Alert(
#         id="alert",
#         is_open=False,
#         dismissable=True,
#         duration=20000,
#     ),
# ),




# # Create a component for downloading data
# dcc.Download(id="download"),
# dcc.Store(id='newfilestorage', storage_type='memory'),
# dcc.Store(id='filestorage', storage_type='session'),

@app.callback(
    Output(component_id = "tab-content", component_property = "children"),
    [Input(component_id ="tabs", component_property = "active_tab")]
)

def render_tab_content(active_tab):

    if active_tab is not None:
        if active_tab == "Upload":
            return dbc.Row([

                dbc.Col(

                    html.H5('Upload/Clear Files'),

                width = 12, className ="text-center" ),

        dbc.Col(

            dcc.Upload(
                id='submit_files',
                children=html.Div([
                    'Drag/Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '20px',
                    'width': '100%',
                },
                # Allow multiple files to be uploaded
                multiple=True), className = 'ms-2'),

        dbc.Col([



            dbc.Row(

                dbc.Button(
                    "Clear Selected Files",
                    id='clear_files',
                    outline=True,
                    color="secondary",
                    className="me-1",
                    n_clicks=0),

            className = 'ms-2'),

            html.Div(className = 'mb-3'),

            dbc.Row([
                # Create a checklist for selecting a velocity
                html.Label("Select files to clear"),
                dcc.Checklist(["All"], [], id="all_clear_file_checklist", inline=True),
                dcc.Checklist(value=[], id="clear_file_checklist", inline=True),
            ]),

        ]),

        dbc.Col([

            dbc.Row(

                dbc.Button(
                    "Upload Selected Files",
                    id='newfile',
                    outline=True,
                    color="primary",
                    className="me-1",
                    n_clicks=0), className = 'ms-2'
            ),


            dbc.Row([
                # Create a checklist for selecting a velocity
                html.Label("Select files to upload"),
                dcc.Checklist(["All"], [], id="all_upload_file_checklist", inline=True),
                dcc.Checklist(value=[], id="upload_file_checklist", inline=True),
            ]),
        ]),


    ], align ='center'),


        elif active_tab == "Download":

            return dbc.Col([

                dbc.Row(

                    # Create a label for downloading data
                    html.Label("Download Options"),

                ),

                dbc.Row(

                    # Create a label for downloading data
                    html.Label("Instructions"),

                ),


                dbc.Col([

                    dbc.Row(

                        html.Label("Choose data file"),

                    ),


                    dbc.Row(

                        # Create a label for selecting a data file
                        dcc.RadioItems(value='', id="file_checklist", inline=True),

                    ),

                    dbc.Row(

                        # Create a label for selecting a data file
                        html.Label("Filename"),

                    ),

                    dbc.Row(

                        # Create a label for selecting a data file
                        dcc.Input(id="file_name_input", type="text", placeholder="Enter Filename"),

                    ),

                    dbc.Row(

                        # Create a button for downloading data
                        html.Button("Download", id="btn_download"),

                    ),

                ]),

                dbc.Col([

                    dbc.Row(

                        html.Label("Choose quantity"),

                    ),

                    dbc.Row([

                        # Create a checklist for selecting a velocity
                        dcc.Checklist(["All"], [], id="all_vel_checklist", inline=True),

                        dcc.Checklist(value=[], id="vel_checklist", inline=True),

                    ]),

                    dbc.Row(

                        html.Label("Choose file type"),

                    ),

                    dbc.Row(

                        # Create a label for selecting a data file
                        dcc.RadioItems(options=['CSV', 'Excel', '.txt'], value='CSV', id="type_checklist", inline=True),

                    ),

                    dbc.Row(

                        html.Label("Time Range"),

                    ),

                    dbc.Row([

                        dbc.Col([

                            dbc.Row(

                                html.Label("Max"),

                            ),

                            dbc.Row(

                                dcc.Input(id="big_t", min=0, type="number", placeholder="Maximum Time", debounce=True, ),

                            ),

                        ]),

                        dbc.Col([

                            dbc.Row(

                                html.Label("Min"),

                            ),

                            dbc.Row(

                                dcc.Input(id="small_t", type="number", placeholder="Minimum Time", debounce=True,
                                          style={'marginRight': '10px'}),
                            ),

                        ]),

                    ]),

                    dbc.Row(

                        html.Label('error'),

                    ),

                ]),
            ])
    return "No tab selected"


@app.callback(
        Output(component_id = 'newfilestorage', component_property = 'data'),
        Output(component_id='alert', component_property='children'),
        Output(component_id='alert', component_property='color'),
        Output(component_id='alert', component_property='is_open'),
        Input(component_id="upload_file_checklist", component_property='value'),
        Input(component_id = 'submit_files',component_property = 'contents'),
        State(component_id = 'submit_files', component_property ='filename'),
        prevent_initial_call = True)

def new_contents(filenames, contents, ALL_filenames):

    if "submit_files" == ctx.triggered_id:

        try:

            contain_text = []

            for name1 in filenames:
                if 'txt' not in name1:
                    contain_text.append(name1)

            if contain_text != []:

                error = 'There was an error processing files: (' + ', '.join(contain_text) + ') ' + '\nPlease check file type'

                color = "danger"

                open1 = True

                newdata = []

            else:

                if filenames and contents is not None:

                    prb = cal_velocity(contents, filenames)

                    newdata = [prb, filenames]

                    filename_str = ', '.join(filenames)

                    error = filename_str + ' selected, please upload for analysis'

                    color = "primary"

                    open1 = True

                else:

                    newdata = []

                    error = 'There was an error processing the files, please try again.'

                    color = "danger"

                    open1 = True

        except Exception:

            newdata = []

            error = 'There was an error processing the files, please try again.'

            color = "danger"

            open1 = True

        return newdata, error, color, open1

    else:

        contain_text = []

        for name in ALL_filenames:
            if 'txt' not in name:
                contain_text.append(name)

        error = 'WARNING: Files: (' + ', '.join(
            contain_text) + ') ' + '\n  unsupported file type. \n Please upload .txt files'

        color = "danger"

        open1 = True

        return no_update, error, color, open1



@app.callback(
    Output(component_id='filestorage', component_property='data'),
    Output(component_id='newfilestorage', component_property='clear_data', allow_duplicate=True),
    Output(component_id='alert', component_property='children', allow_duplicate=True),
    Output(component_id='alert', component_property='color', allow_duplicate=True),
    Output(component_id='alert', component_property='is_open', allow_duplicate=True),
    Input(component_id='newfile', component_property='n_clicks'),
    State(component_id='newfilestorage', component_property='data'),
    State(component_id='filestorage', component_property='data'),
    prevent_initial_call=True)

def content(n_clicks, newData, data):


    if n_clicks is None:
        raise PreventUpdate

    if "newfile" == ctx.triggered_id:

        if newData is not None and newData != []:

            if data is None:

                new_prb = newData[0]

                new_filenames = newData[1]

                data = [new_prb, new_filenames]

                error = ', '.join(new_filenames) + ' uploaded'

                color = "primary"

            else:

                new_prb = newData[0]

                new_filenames = newData[1]

                prb = data[0]

                filenames = data[1]

                # Create a new list to hold the combined values
                combined_filenames = filenames.copy()
                new_value = []
                repeated_value = []

                for i, value in enumerate(new_filenames):
                    # Check if the value is already in the combined list
                    if value not in combined_filenames:
                        new_value.append(value)
                        prb[value] = {value: {}}
                        prb[value] = new_prb[value]
                        combined_filenames.append(value)
                    if value in combined_filenames:
                        repeated_value.append(value)

                if len(new_value) != len(repeated_value):

                    if len(new_value) == 0:

                        error = ', '.join(repeated_value) + ' not uploaded as repeated filenames were found'
                    else:
                        error = ', '.join(new_value) + ' uploaded successfully ,but' + ', '.join(repeated_value) +\
                                ' not uploaded as repeated filenames were found'

                    color = "danger"

                else:

                    error = ', '.join(new_value) + ' uploaded'

                    color = "primary"

                data = [prb, combined_filenames]

        else:

            error = 'No files selected to upload'

            color = "danger"

            data = data

        newData = True

        open1 = True

        return data, newData, error, color, open1

@app.callback(
        Output(component_id="upload_file_checklist", component_property='value'),
        Output(component_id='all_upload_file_checklist', component_property='value'),
        Input(component_id="upload_file_checklist", component_property='value'),
        Input(component_id='all_upload_file_checklist', component_property='value'),
        Input(component_id = 'submit_files', component_property ='filename'),
        prevent_initial_call=True
        )

def file_upload_sync_checklist(upload_file_check, all_upload_file_check, Uploaded_filenames):

    if Uploaded_filenames is None:
        raise PreventUpdate

    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if input_id == "upload_file_check":

        all_upload_file_check = ["All"] if set(upload_file_check) == set(Uploaded_filenames) else []

    else:

        upload_file_check = Uploaded_filenames if all_upload_file_check else []

    return upload_file_check, all_upload_file_check

@app.callback(
        Output(component_id="clear_file_checklist", component_property='value'),
        Output(component_id='all_clear_file_checklist', component_property='value'),
        Input(component_id="clear_file_checklist", component_property='value'),
        Input(component_id='all_clear_file_checklist', component_property='value'),
        Input(component_id='filestorage', component_property='data'),
        prevent_initial_call=True
        )

def file_clear_sync_checklist(clear_file_check, all_clear_check, data):

    if data is None:
        raise PreventUpdate

    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    file_options = data[1]

    if input_id == "clear_file_checklist":

        all_clear_check = ["All"] if set(clear_file_check) == set(file_options) else []

    else:

        clear_file_check = file_options if all_clear_check else []

    return clear_file_check, all_clear_check

@app.callback(
    Output(component_id="File", component_property='options'),
    Output(component_id='Vect', component_property='options'),
    Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="vel_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="clear_file_checklist", component_property='options', allow_duplicate=True),
    Output(component_id="file_checklist", component_property='value', allow_duplicate=True),
    Input(component_id='filestorage', component_property='data'),
    prevent_initial_call=True)

def update_dropdowns(data):

    if data is None:
        raise PreventUpdate

    vect_options = ['Ux', 'Uy', 'Uz']

    file_dropdown_options = data[1]

    file_checklist = file_dropdown_options

    clear_file_check = file_checklist

    vel_checklist = ['Ux', 'Uy', 'Uz', 't']

    file_val = file_checklist[0]

    return file_dropdown_options, vect_options, file_checklist, vel_checklist, clear_file_check, file_val



@app.callback(
        Output(component_id="vel_checklist", component_property='value'),
        Output(component_id='all_vel_checklist', component_property='value'),
        Input(component_id="vel_checklist", component_property='value'),
        Input(component_id='all_vel_checklist', component_property='value'),
        prevent_initial_call=True
        )


def vel_sync_checklist(vel_check, all_vel_checklist):

    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    vel_type = ['Ux','Uy','Uz','t']

    if input_id == "vel_checklist":

        all_vel_checklist = ["All"] if set(vel_check) == set(vel_type) else []

    else:

        vel_check = vel_type if all_vel_checklist else []

    return vel_check, all_vel_checklist


@app.callback(
        Output(component_id="big_t", component_property='value', allow_duplicate=True),
        Output(component_id="small_t", component_property='value', allow_duplicate=True),
        Input(component_id="small_t", component_property='value'),
        Input(component_id="big_t", component_property='value'),
        prevent_initial_call=True)


def update_In(Sin_val, Lin_val):

    if Lin_val is None and Sin_val is None:
         raise PreventUpdate

    if Lin_val is None:
        Lin_val = 0

    if Sin_val is None:
        Sin_val = 0

    if Lin_val < Sin_val:
        Lin_val = Sin_val

    return Lin_val, Sin_val,


@app.callback(
        [Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
        Output(component_id = 'time-range', component_property = 'min', allow_duplicate=True),
        Output(component_id = 'time-range', component_property = 'max', allow_duplicate=True),
        Output(component_id = 'time-range', component_property = 'value', allow_duplicate=True),
        Output(component_id='btn_title_update', component_property='n_clicks'),
        Output(component_id='btn_leg_update', component_property='n_clicks'),
        Output(component_id='alert', component_property='children', allow_duplicate=True),
        Output(component_id='alert', component_property='color', allow_duplicate=True),
        Output(component_id='alert', component_property='is_open', allow_duplicate=True)],
        [Input(component_id = 'filestorage', component_property = 'data'),
        Input(component_id = 'File', component_property = 'value'),
        Input(component_id = 'Vect', component_property = 'value'),
        Input(component_id = 'time-range', component_property = 'value'),
        Input(component_id='line_thick', component_property='value'),
        Input(component_id='legend_onoff', component_property='value'),
        Input(component_id='title_onoff', component_property='value'),
        Input(component_id='btn_title_update', component_property='n_clicks'),
        Input(component_id='btn_leg_update', component_property='n_clicks'),
        State(component_id='alert', component_property='children'),
        State(component_id='alert', component_property='color'),
        State(component_id='alert', component_property='is_open'),
        State(component_id='New_Titlename', component_property='value'),
        State(component_id='New_LegName', component_property='value')],
        prevent_initial_call = True)

def update_dropdowns(data, user_inputs, user_inputs1,time_input,line_thick, leg, title, n_clicks, n_clicks1, error,
                     color, open1, NewTit_name, NewLeg_name):

    if data is None or {}:
        raise PreventUpdate


    if user_inputs == [] or user_inputs1 == []:

        error = error

        color = color

        open1 = open1

        fig = {}

        min_sl = 1

        max_sl = 10

        value =[1, 10]

    else:

        df = data[0]

        max1 = []

        min1 = []

        fig = go.Figure()

        current_names = []


        if "File" == ctx.triggered_id or "Vect" == ctx.triggered_id:

            for user_input in user_inputs:
                for user_input1 in user_inputs1:
                    V = df[user_input][user_input1]
                    t = df[user_input]['t']
                    max1.append(np.round(np.amax(t)))
                    min1.append(np.round(np.amin(t)))
                    fig.add_trace(go.Scatter(x=t, y=V, mode='lines',
                                            line=dict(
                                            width=line_thick),
                                            name=f"{user_input}{' '}{user_input1}"))
                    current_names.append(f"{user_input}{' '}{user_input1}")

            min_sl = min(min1)
            max_sl = max(max1)

            value = [min_sl, max_sl]

        else:

            for user_input in user_inputs:
                for user_input1 in user_inputs1:
                    V = np.array(df[user_input][user_input1])
                    t = np.array(df[user_input]['t'])
                    max1.append(np.round(np.amax(t)))
                    min1.append(np.round(np.amin(t)))
                    mask = (t >= time_input[0]) & (t < time_input[1])
                    t2 = t[mask]
                    V2 = V[mask]
                    fig.add_trace(go.Scatter(x=t2, y=V2, mode='lines',
                                            line=dict(
                                            width=line_thick),
                                            name=f"{user_input}{' '}{user_input1}"))
                    current_names.append(f"{user_input}{' '}{user_input1}")


            value = time_input
            min_sl = min(min1)
            max_sl = max(max1)

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


        if title == 'Off':
            fig.layout.update(title='')

        elif title =='On' and n_clicks >= 1  and NewTit_name !='' and NewTit_name !=None:
            fig.layout.update(title=NewTit_name)

        elif title == 'On':
            fig.layout.update(title='Barnacle Data')


        if leg == 'Off':
            fig.layout.update(showlegend=False)

        elif leg =='On' and n_clicks1 >= 1 and NewLeg_name !='' and NewLeg_name !=None:

            NewLeg_name_list = NewLeg_name.split(',')

            newname_result = {}

            if len(current_names) == len(NewLeg_name_list):

                error = 'Legend Updated'

                color = "success"

                for i, current_name in enumerate(current_names):
                    newnames = {current_name: NewLeg_name_list[i]}
                    newname_result.update(newnames)

                fig.for_each_trace(lambda t: t.update(name=newname_result[t.name],
                                                      legendgroup=newname_result[t.name],
                                                      hovertemplate=t.hovertemplate.replace(t.name, newname_result[
                                                          t.name]) if t.hovertemplate is not None else None)
                                   )

            else:

                error = 'Number of legend entries do not match'

                color = "danger"


            open1 = True

            fig.layout.update(showlegend=True)

        elif leg == 'On':

            fig.layout.update(showlegend=True)

    return fig, min_sl, max_sl, value, n_clicks, n_clicks1, error, color, open1,


@app.callback(
        Output(component_id="download", component_property='data', allow_duplicate=True),
        Output(component_id='alert', component_property='children', allow_duplicate=True),
        Output(component_id='alert', component_property='color', allow_duplicate=True),
        Output(component_id='alert', component_property='is_open', allow_duplicate=True),
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

    if "btn_download" == ctx.triggered_id:

        if file == '':

            error = 'No data to download'

            color = 'danger'

            open1 = True

            return no_update, error, open1, color

        if vels == [] or vels is None:

            error = 'No data selected'

            color = 'danger'

            open1 = True

            return no_update, error, open1, color

        else:

            prb = data[0]

            dff = {file: {vel_opt: np.array(prb[file][vel_opt]) for vel_opt in vel_opts}}

            df = {file: {vel: [] for vel in vels}}

            if smallt is not None or bigt is not None:

                t = np.array(dff[file]['t'])
                max1 = np.amax(t)
                min1 = np.amin(t)

                if smallt is not None and bigt is not None:

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


                for vel in vels:
                    df[file][vel] = dff[file][vel][mask]

            else:

                for vel in vels:
                    df[file][vel] = dff[file][vel]

                error1 = [file_type + ' File Downloaded', 'primary']

                open1 = True

            if file_type == '.txt':

                list_all = []

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

                vels_str = ','.join(vels)
                str_all = vels_str + '\n' + str_all
                str_all = str_all.replace(' ', '')
                str_all = str_all.replace('],', '')
                str_all = str_all.replace(']]', '')
                str_all = str_all.replace('[[', '')
                str_all = str_all.replace('[', '')
                str_all = str_all.replace(']', '')

                if selected_name is None or selected_name == '':
                    value = file.split('.')
                    filenameTXT = value[0] + ".txt"

                else:
                    filenameTXT = selected_name + ".txt"

                text = dict(content=str_all, filename = filenameTXT )

            if file_type == 'Excel' or 'CSV':

                # create an empty list to store dataframes
                pandaData = []

                # loop through each file and convert to dataframe
                for file, df in df.items():
                    dfff = pd.DataFrame(df)
                    pandaData.append(dfff)
                # concatenate all dataframes in the list
                PDdata = pd.concat(pandaData)

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

            return text, error1[0], True, error1[1],

@app.callback(
        Output(component_id="File", component_property='value', allow_duplicate=True),
        Output(component_id='Vect', component_property='value', allow_duplicate=True),
        Output(component_id="File", component_property='options', allow_duplicate=True),
        Output(component_id='Vect', component_property='options', allow_duplicate=True),
        Output(component_id='Velocity_Graph', component_property='figure', allow_duplicate=True),
        Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
        Output(component_id="vel_checklist", component_property='options', allow_duplicate=True),
        Output(component_id='all_vel_checklist', component_property='value', allow_duplicate=True),
        Output(component_id='New_Titlename', component_property='value', allow_duplicate=True),
        Output(component_id='New_LegName', component_property='value', allow_duplicate=True),
        Output(component_id='file_name_input', component_property='value', allow_duplicate=True),
        Output(component_id="small_t", component_property='value', allow_duplicate=True),
        Output(component_id="big_t", component_property='value', allow_duplicate=True),
        Output(component_id="line_thick", component_property='value', allow_duplicate=True),
        Output(component_id="filestorage", component_property='clear_data', allow_duplicate=True),
        Output(component_id='newfilestorage', component_property='clear_data', allow_duplicate=True),
        Output(component_id="clear_file_checklist", component_property='options', allow_duplicate=True),
        Output(component_id='alert', component_property='children', allow_duplicate=True),
        Output(component_id='alert', component_property='color', allow_duplicate=True),
        Output(component_id='alert', component_property='is_open', allow_duplicate=True),
        Output(component_id='filestorage', component_property='data', allow_duplicate=True),
        Output(component_id="file_checklist", component_property='value'),
        Input(component_id='clear_files', component_property='n_clicks'),
        State(component_id='filestorage', component_property='data'),
        State(component_id="clear_file_checklist", component_property='value'),
        State(component_id="all_clear_file_checklist", component_property='value'),
        prevent_initial_call=True)

def clear_files(n_clicks, maindata, whatclear, allclear):

    if "clear_files" == ctx.triggered_id:

        if allclear == ['All']:

            newmaindata = []

            clear_data_main = True

            clear_data = True

            error = 'All files cleared'

        elif len(whatclear) >= 1:

            df1 = maindata[0]
            df2 = maindata[1]

            for what in whatclear:
                del df1[what]
                df2.remove(what)

            newmaindata = [df1, df2]

            error = ', '.join(whatclear) + ' deleted'

            clear_data_main = False

            clear_data = True

        else:

            newmaindata = maindata

            error = 'No files deleted as none were selected'

            clear_data_main = False

            clear_data = True

        file_download_val = ''

        vect_val = []

        file_val = []

        file_dropdown_options = []

        vect_options = []

        fig = {}

        file_checklist = []

        vel_checklist = []

        all_vel_checklist = []

        in_val_S = 0

        in_val_L = 1

        title_name = ''

        new_legname = ''

        file_name_inp = ''

        line_thickness = 1

        clear_opt = []

        color = "success"

        open1 = True

    return file_val, vect_val, file_dropdown_options, vect_options, fig, file_checklist,\
        vel_checklist, all_vel_checklist, title_name, new_legname, file_name_inp, in_val_S, in_val_L, line_thickness,\
        clear_data_main, clear_data, clear_opt, error, color, open1, newmaindata, file_download_val


# Run app
if __name__== '__main__':
    app.run_server(debug=True)

