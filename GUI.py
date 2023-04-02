from dash import Dash, dcc, Output, Input, ctx, State
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import sys

from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import numpy as np
import scipy.io as sio
from pathlib import Path, PureWindowsPath
import plotly.graph_objects as go


# Create an instance of tkinter frame or window
def file_chooser():

    win = Tk()
    # Set the geometry of tkinter frame
    win.geometry("800x350")
    # Make the window jump above all
    win.attributes('-topmost', 1)
    # Add a Label widget
    label = Label(win, text="Select the Button to Open the File", font=('Aerial 11'))
    label.pack(pady=30)
    # Add a Treeview widget to display the selected file names
    tree = ttk.Treeview(win, columns=('Filename'))
    tree.heading('#0', text='Index')
    tree.heading('Filename', text='Filename')
    tree.pack(side=LEFT, padx=30, pady=30)

    def open_file():
        files = fd.askopenfilenames(parent=win, title='Choose a File')
        global file_paths
        file_paths = list(win.splitlist(files))
        # Clear the Treeview widget before inserting new file names
        tree.delete(*tree.get_children())
        # Update the table with the selected file names
        global file_names
        file_names = []
        for i, file_path in enumerate(file_paths):
            file_name = file_path.split("/")[-1]
            file_names.append(file_name)
            tree.insert('', 'end', text=str(i + 1), values=(file_name,))
        return file_paths, file_names

    def close_window():
        win.destroy()

    # Add a Button Widget
    ttk.Button(win, text="Select a File", command=open_file).pack()
    # Add a Close Button Widget

    # Add a Label widget for close button
    label = Label(win, text="Close window once files are added", font=('Aerial 11'))
    label.pack(pady=30)
    ttk.Button(win, text="Close", command=close_window).pack()

    win.mainloop()

    return file_paths, file_names

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

# print(file_names)
# print(file_paths)

#file_paths = ['C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 1.txt', 'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 2.txt']

#file_names = ['Example 1.txt', 'Example 2.txt']

#file_names = {}

# Import necessary modules

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# Define layout of the app
app.layout = html.Div([

    # Create a heading
    html.H1("BARNACLE SENSOR ANALYSIS", style={'text-align': 'center'}),

    # Create a graph component
    dcc.Graph(id='Velocity_Graph', figure={}),

    # Create a range slider component
    html.Label("Choose Time", style={'text-align': 'center'}),

    dcc.RangeSlider(
        id='time-range',
        min=1,
        max=10,
        value=[1, 10],
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag',
    ),

    # Create a div component
    html.Div(children=[

        # Create a button to select files
        html.Br(),
        html.Button("Select Files", id='submit_files', n_clicks=0, type='button'),
        html.Br(),
        html.Br(),


        # Create a button to clear files
        html.Button("Clear Files", id='clear_files', n_clicks=0),
        html.Br(),
        html.Br(),

        # Create a dropdown for selecting a dataset
        html.Label("Choose DataSet"),
        dcc.Dropdown(
            id="File",
            options=[],
            multi=True,
            value=['Example 1.txt'],
            placeholder="Select a dataset",
            style={'width': "50%"}
        ),

        # Create a dropdown for selecting a velocity
        html.Label("Choose Velocity"),
        dcc.Dropdown(
            id="Vect",
            options=[],
            multi=True,
            value=[],
            placeholder="Select a velocity",
            style={'width': "50%"}
        ),

        html.Br(),

        html.Label("Graph Options", style={'text-align': 'left'}),

        html.Label("Line Thickness", style={'text-align': 'left'}),

        daq.Slider(min = 0.5,max = 5,
                   value = 1,
                   step=0.1,
            id="line_thick",
            marks={0.5: {'label': 'Thin'}, 5: {'label': 'Thick'}},
            updatemode = 'drag'),

        html.Br(),


        html.Div(children = [

        html.Label("Legend", style={'text-align': 'left'}),

        dcc.RadioItems(id ='legend_onoff', value = 'On', options = ['On', 'Off']),

        dcc.Input(
                id="New_LegName",
                type='text',
                placeholder="Enter New Legend (seperate values with comma)",
                debounce=True,
            ),

            # Create a button for downloading data
        html.Button("Update Legend", id="btn_leg_update", n_clicks=0, ),

        ]),


        html.Div(children=[

            html.Label("Title", style={'text-align': 'left'}),

            dcc.RadioItems(id = 'title_onoff', value = 'On', options = ['On', 'Off']),

            dcc.Input(
                id="New_Titlename",
                type='text',
                placeholder="Enter New Title",
                debounce=True,
            ),

            # Create a button for downloading data
            html.Button("Update Title", id="btn_title_update", n_clicks=0, ),

        ]),

        html.Div([

            html.Label("Custom Colours", style={'text-align': 'left'}),

            html.Label("Select ", style={'text-align': 'left'}),

            # Create a checklist for selecting a data file
            dcc.Checklist(["All"], [], id="all_leg_checklist", inline=True),
            dcc.Checklist(id="leg_list", inline=True),

            daq.ColorPicker(

                id='color_picker',
                label='Color Picker',
                value=dict(hex='#119DFF')
            ),

            html.Div(id='color-picker-output-1'),
        ]),



        html.Br(),

        html.Br(),

        # Create a label for downloading data
        html.Label("Download Data"),

        # Create a label for selecting a data file
        html.Label("Select Data File"),

        # Create a checklist for selecting a data file
        dcc.RadioItems(value='', id="file_checklist", inline=True),

        # Create a checklist for selecting a file type
        html.Label("Select File Type"),
        dcc.RadioItems(['CSV', 'Excel', '.txt'], id="type_checklist", inline=True),


        # Create a label for selecting a velocity
        html.Label("Select Velocity"),

        # Create a checklist for selecting a velocity
            dcc.Checklist(["All"], [], id="all_vel_checklist", inline=True),
            dcc.Checklist(value=[], id="vel_checklist", inline=True),


            html.I(
                "Input the maximum and minimum time values for download"),
            html.Br(),
            dcc.Input(id="small_t", min = 0, type="number", placeholder = "Minimum Time", debounce = True, style={'marginRight': '10px'}),
            dcc.Input(id="big_t", min = 0, type="number", placeholder="Maximum Time", debounce = True,),
            html.Br(),
            html.Br(),
            dcc.Input(id="file_name_input", type="text", placeholder="Enter Filename", debounce=True, ),

        html.Br(),

        html.Br(),

        # Create a button for downloading data
        html.Button("Download", id="btn_download"),

        # Create a component for downloading data
        dcc.Download(id="download"),

    ])
])



# @app.callback(
#      [Output(component_id="File", component_property='options'),
#      Output(component_id='Vect', component_property='options'),
#      Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
#      Output(component_id="vel_checklist", component_property='options', allow_duplicate=True)],
#     [Input(component_id="submit_files", component_property='n_clicks'),
#     Input(component_id="File", component_property='options'),
#      Input(component_id='Vect', component_property='options'),
#      Input(component_id="file_checklist", component_property='options'),
#      Input(component_id="vel_checklist", component_property='options')
#      ], prevent_initial_call=True)
#
# def upload_data(n_clicks, file_dropdown_options, vect_options, file_checklist, vel_checklist ):
#
#     file_paths = ['C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 1.txt', 'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 2.txt']
#
#     file_names = ['Example 1.txt', 'Example 2.txt']
#
#     file_checklist = file_checklist
#
#     vel_checklist = vel_checklist
#
#     file_dropdown_options = file_dropdown_options
#
#     vect_options = vect_options
#
#     if n_clicks <= 1:
#
#         if file_dropdown_options == [] and vect_options == []:
#
#             if "submit_files" == ctx.triggered_id:
#
#                 # This block of code will run when the user clicks the submit button
#                 #file_chooser()
#
#                 vect_options = ['Ux', 'Uy', 'Uz']
#
#                 file_dropdown_options = file_names
#
#                 global prb
#
#                 prb = cal_velocity(file_paths)
#
#                 file_checklist = file_dropdown_options
#
#                 vel_checklist = ['Ux','Uy','Uz','t']
#
#     return file_dropdown_options, vect_options, file_checklist, vel_checklist

@app.callback(
    [Output(component_id="File", component_property='options'),
     Output(component_id='Vect', component_property='options'),
     Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
     Output(component_id="vel_checklist", component_property='options', allow_duplicate=True)],
    [Input(component_id="submit_files", component_property='n_clicks'),
     Input(component_id="File", component_property='options'),
     Input(component_id='Vect', component_property='options'),
     Input(component_id="file_checklist", component_property='options'),
     Input(component_id="vel_checklist", component_property='options')
     ], prevent_initial_call=True)

def upload_data(n_clicks, file_dropdown_options, vect_options, file_checklist, vel_checklist):

    file_paths = [

        'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 1.txt',
        'C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/Example 2.txt']

    file_names = ['Example 1.txt', 'Example 2.txt']

    file_dropdown_options = ['Example 1.txt', 'Example 2.txt']

    vect_options = ['Ux', 'Uy', 'Uz']

    global prb

    prb = cal_velocity(file_paths)

    file_checklist = file_dropdown_options

    vel_checklist = ['Ux', 'Uy', 'Uz', 't']

    return file_dropdown_options, vect_options, file_checklist, vel_checklist


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
        Input(component_id='clear_files', component_property='n_clicks'),
        prevent_initial_call=True)


def update_In(Sin_val, Lin_val, n_clicks):

    if "clear_files" != ctx.triggered_id:

        if Lin_val == None:

            Lin_val = Sin_val + 1

        if Sin_val == None:

            Sin_val = Lin_val - 1

        if Lin_val - Sin_val < 1:

            Lin_val = Sin_val + 1

    return Lin_val, Sin_val

# @app.callback(
#     Output(component_id='leg_list', component_property='options'),
#     Input(component_id = 'Velocity_Graph', component_property = 'figure'),
#     Input(component_id='color_picker', component_property='value')
#     , prevent_initial_call=True)
#

@app.callback(
    Output(component_id="leg_list", component_property='value'),
    Output(component_id='all_leg_checklist', component_property='value'),
    Input(component_id="leg_list", component_property='value'),
    Input(component_id='all_leg_checklist', component_property='value'),
    Input(component_id="leg_list", component_property='options'),
     prevent_initial_call=True)

def leg_sync_checklist(leg_list, all_leg_checklist, leg_act ):

    input_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if leg_act != []:

        if input_id == "leg_list":

            all_leg_checklist = ["All"] if set(leg_list) == set(leg_act) else []

        else:

            leg_list = leg_act if all_leg_checklist else []

    else:
        leg_list = []

        all_leg_checklist = []


    return leg_list, all_leg_checklist

@app.callback(
    [Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
    Output(component_id = 'time-range', component_property = 'min', allow_duplicate=True),
    Output(component_id = 'time-range', component_property = 'max', allow_duplicate=True),
    Output(component_id = 'time-range', component_property = 'value', allow_duplicate=True),
    Output(component_id="small_t", component_property ='min'),
     Output(component_id="small_t", component_property='max'),
    Output(component_id="big_t", component_property='min'),
     Output(component_id="big_t", component_property='max'),
     Output(component_id='leg_list', component_property='options'),
     Output(component_id='btn_title_update', component_property='n_clicks'),
     Output(component_id='btn_leg_update', component_property='n_clicks')],
    [Input(component_id = 'File', component_property = 'value'),
    Input(component_id = 'Vect', component_property = 'value'),
    Input(component_id = 'time-range', component_property = 'value'),
     Input(component_id='line_thick', component_property='value'),
     Input(component_id='legend_onoff', component_property='value'),
     Input(component_id='title_onoff', component_property='value'),
     Input(component_id='btn_title_update', component_property='n_clicks'),
     Input(component_id='btn_leg_update', component_property='n_clicks'),
     Input(component_id='submit_files', component_property='n_clicks'),
     State(component_id='New_Titlename', component_property='value'),
     State(component_id='New_LegName', component_property='value'),
     ],
    prevent_initial_call = True)

def update_dropdowns(user_inputs, user_inputs1,time_input,line_thick, leg, title, n_clicks, n_clicks1, n_clicks2, NewTit_name, NewLeg_name):

    if user_inputs == [] or user_inputs1 == []:

        fig = go.Figure()

        fig = {}

        min_sl = 1

        max_sl = 10

        value =[1, 10]

        Smax_in = max_sl-1

        Smin_in = min_sl

        Lmax_in = max_sl

        Lmin_in = min_sl+1

    else:

        df = prb

        max1 = []

        min1 = []

        fig = go.Figure()

        current_names = []

# 0.5 to
        color1 = 'rgb(0,0,0)'


        if "File" == ctx.triggered_id or "Vect" == ctx.triggered_id and n_clicks1 >= 1:

            for user_input in user_inputs:
                for user_input1 in user_inputs1:
                    V = df[user_input][user_input1]
                    t = df[user_input]['t']
                    max1.append(np.round(np.amax(t)))
                    min1.append(np.round(np.amin(V)))
                    fig.add_trace(go.Scatter(x=t, y=V, mode='lines',
                                            line=dict(
                                            color = color1,
                                            width=line_thick),
                                            name=f"{user_input}{' '}{user_input1}"))
                    current_names.append(f"{user_input}{' '}{user_input1}")

            min_sl = min(min1)
            max_sl = max(max1)

            value = [min_sl, max_sl]

        else:

            for user_input in user_inputs:
                for user_input1 in user_inputs1:
                    V = df[user_input][user_input1]
                    t = df[user_input]['t']
                    max1.append(np.round(np.amax(t)))
                    min1.append(np.round(np.amin(V)))
                    mask = (t >= time_input[0]) & (t < time_input[1])
                    t2 = t[mask]
                    V2 = V[mask]
                    fig.add_trace(go.Scatter(x=t2, y=V2, mode='lines',
                                            line=dict(
                                            color=color1,
                                            width=line_thick),
                                            name=f"{user_input}{' '}{user_input1}"))
                    current_names.append(f"{user_input}{' '}{user_input1}")



            value = time_input
            min_sl = min(min1)
            max_sl = max(max1)

        Smax_in = max_sl-1
        Smin_in = min_sl
        Lmax_in = max_sl
        Lmin_in = min_sl+1

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

            print(NewLeg_name)

            NewLeg_name_list = NewLeg_name.split(',')

            newname_result = {}
            for i, current_name in enumerate(current_names):
                newnames = {current_name: NewLeg_name_list[i]}
                newname_result.update(newnames)

            fig.for_each_trace(lambda t: t.update(name=newname_result[t.name],
                                                  legendgroup=newname_result[t.name],
                                                  hovertemplate=t.hovertemplate.replace(t.name, newname_result[
                                                      t.name]) if t.hovertemplate is not None else None)
                               )

            fig.layout.update(showlegend=True)

        elif leg == 'On':
            fig.layout.update(showlegend=True)



    if fig == {}:

        names = []

    elif "newname_result" in locals():


        names = newname_result


    else:

        names = current_names




    return fig, min_sl, max_sl, value, Smin_in, Smax_in, Lmin_in, Lmax_in, names, n_clicks, n_clicks1


@app.callback(
            [Output(component_id="File", component_property='value', allow_duplicate=True),
            Output(component_id='Vect', component_property='value', allow_duplicate=True),
            Output(component_id="File", component_property='options', allow_duplicate=True),
            Output(component_id='Vect', component_property='options', allow_duplicate=True),
            Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
            Output(component_id="file_checklist", component_property='options', allow_duplicate=True),
            Output(component_id="vel_checklist", component_property='options', allow_duplicate=True),
            Output(component_id='all_vel_checklist', component_property='value', allow_duplicate=True),
            Output(component_id="small_t", component_property='value'),
            Output(component_id="big_t", component_property='value'),
            Output(component_id="submit_files", component_property='n_clicks')],
            Input(component_id='clear_files', component_property='n_clicks'),
            prevent_initial_call=True)

def clear_files(n_clicks):

    if "clear_files" == ctx.triggered_id:

        vect_val = []

        file_val = []

        file_dropdown_options = []

        vect_options = []

        fig = {}

        file_checklist = []

        vel_checklist = []

        all_vel_checklist = []

        in_val_S = "Minimum Time"
        #
        in_val_L = "Maximum Time"

        n_clicks = 0

        return file_val, vect_val, file_dropdown_options, vect_options, fig, file_checklist, vel_checklist, all_vel_checklist, in_val_S, in_val_L, n_clicks


# @app.callback(
#         Output(component_id="CSV_download", component_property='data'),
#         Input(component_id="btn_download", component_property='n_clicks'),
#         Input(component_id="small_t", component_property='value'),
#         Input(component_id="big_t", component_property='value'),
#         Input(component_id="vel_checklist", component_property='value'),
#         Input(component_id="file_checklist", component_property='value'),
#         Input(component_id="type_checklist", component_property='value'),
#         prevent_initial_call=True)
#
# def download_CSV(n_clicks,smallt, bigt, vels, file, type):
#
@app.callback(
        Output(component_id="download", component_property='data', allow_duplicate=True),
        Input(component_id="btn_download", component_property='n_clicks'),
        Input(component_id="small_t", component_property='value'),
        Input(component_id="big_t", component_property='value'),
        Input(component_id="vel_checklist", component_property='value'),
        Input(component_id="vel_checklist", component_property='options'),
        Input(component_id="file_checklist", component_property='value'),
        Input(component_id="type_checklist", component_property='value'),
        Input(component_id="file_name_input", component_property='value'),
        prevent_initial_call=True)

def download_EXCEL(n_clicks,smallt, bigt, vels, vel_opts, file, type, name):

    if "btn_download" == ctx.triggered_id and [type == 'Excel' or 'CSV']:

        dff = {file: {vel_opt: prb[file][vel_opt] for vel_opt in vel_opts}}

        df = {file: {vel: [] for vel in vels}}

        if smallt != None or bigt != None:

            t = dff[file]['t']

            if smallt != None and bigt != None:
                mask = (t >= smallt) & (t < bigt)

            if smallt != None and bigt == None:
                mask = (t >= smallt)

            if bigt != None and smallt == None:
                mask = (t < bigt)

            df[file]['t'] = t[mask]
            for vel in vels:
                df[file][vel] = dff[file][vel][mask]

        else:

            df = dff

        # create an empty list to store dataframes
        pandaData = []

        # loop through each file and convert to dataframe
        for file, df in df.items():
            dfff = pd.DataFrame(df)
            pandaData.append(dfff)

        # concatenate all dataframes in the list
        data = pd.concat(pandaData)

        if name == None:
            value = file.split('.')
            filename = value[0]
        else:
            filename = name

        if type == 'Excel':

            if name == None:
                value = file.split('.')
                filename = value[0] + '.xlsx'
            else:
                filename = name + '.xlsx'

            return dcc.send_data_frame(data.to_excel, filename)

        if type == 'CSV':

            if name == None:
                value = file.split('.')
                filename = value[0] + '.csv'
            else:
                filename = name + '.csv'

            return dcc.send_data_frame(data.to_csv, filename)

@app.callback(
        Output(component_id="download", component_property='data', allow_duplicate=True),
        Input(component_id="btn_download", component_property='n_clicks'),
        Input(component_id="small_t", component_property='value'),
        Input(component_id="big_t", component_property='value'),
        Input(component_id="vel_checklist", component_property='value'),
        Input(component_id="vel_checklist", component_property='options'),
        Input(component_id="file_checklist", component_property='value'),
        Input(component_id="type_checklist", component_property='value'),
        Input(component_id="file_name_input", component_property='value'),
        prevent_initial_call=True)


def download_TXT(n_clicks,smallt, bigt, vels, vel_opts, file, type, name):

    if "btn_download" == ctx.triggered_id and type =='.txt':

        dff = {file: {vel_opt: prb[file][vel_opt] for vel_opt in vel_opts}}

        df = {file: {vel_opt: [] for vel_opt in vel_opts}}

        if smallt != None or bigt != None:

            t = dff[file]['t']

            if smallt != None and bigt != None:
                mask = (t >= smallt) & (t < bigt)

            if smallt != None and bigt == None:
                mask = (t >= smallt)

            if bigt != None and smallt == None:
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

        if name == None:
            value = file.split('.')
            filename = value[0] + ".txt"
        else:
            filename = name + ".txt"

        text = dict(content=str_all, filename=filename)

        return text

# Run app
if __name__== '__main__':
    app.run_server(debug=True)

