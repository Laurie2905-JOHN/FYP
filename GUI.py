from dash import Dash, dcc, Output, Input, ctx, State
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

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

def cal_velocity(BarnFilePath):
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

app = Dash(__name__)



app.layout = html.Div([

    html.H1("BARNACLE SENSOR ANALYSIS", style={'text-align': 'center'}),

    dcc.Graph(id='Velocity_Graph', figure={}),

    html.Label("Choose Time",
               style={'text-align': 'center'}
),

        dcc.RangeSlider(
            id='time-range',
            min= 1,
            max= 10,
            value= [1, 10],
            tooltip={"placement": "bottom", "always_visible": True},
            updatemode = 'drag',
            ),

    html.Div(children=[

        html.Br(),

        html.Button("Select Files", id='submit_files',
                    n_clicks=0,
                    type='button'),

        html.Br(),

        html.Button("Clear Files", id='clear_files', n_clicks=0),

        html.Br(),

        html.Label("Choose DataSet"),

        dcc.Dropdown(id="File",
                     options=[],
                     multi=True,
                     value=[],
                     placeholder="Select a dataset",
                     style={'width': "50%"}),

        html.Label("Choose Velocity"),

        dcc.Dropdown(id="Vect",
                     options=[],
                     multi=True,
                     value=[],
                     placeholder="Select a velocity",
                     style={'width': "50%"})
    ]),
],
)


@app.callback(
     [Output(component_id="File", component_property='options'),
     Output(component_id='Vect', component_property='options'),
      ],
    [Input(component_id='submit_files', component_property='n_clicks'),
     Input(component_id="File", component_property='options'),
     Input(component_id='Vect', component_property='options')],
    prevent_initial_call=True

)

def upload_data(n_clicks, file_dropdown_options, vect_options ):

    if file_dropdown_options == [] or vect_options == []:

        if "submit_files" == ctx.triggered_id:

            # This block of code will run when the user clicks the submit button
            data1 = file_chooser()

            file_paths1 = data1[0]

            file_names1 = data1[1]

            vect_options = ['Ux', 'Uy', 'Uz']

            file_dropdown_options = file_names

            global prb
            prb = cal_velocity(file_paths)

            # While data is the same
            prb['Example 2.txt'].update({'Ux': prb['Example 2.txt']['Ux'] * 0.3,
                                         'Uy': prb['Example 2.txt']['Uy'] * 0.3,
                                         'Uz': prb['Example 2.txt']['Uz'] * 0.3})

            prb['Example 2.txt']['t'] -= 50


    return file_dropdown_options, vect_options



@app.callback(
    [Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True),
    Output(component_id = 'time-range', component_property = 'min', allow_duplicate=True),
    Output(component_id = 'time-range', component_property = 'max', allow_duplicate=True),
    Output(component_id = 'time-range', component_property = 'value', allow_duplicate=True),
     ],
    [
    Input(component_id = 'File', component_property = 'value'),
    Input(component_id = 'Vect', component_property = 'value'),
    Input(component_id = 'time-range', component_property = 'value'),
    Input(component_id='time-range', component_property='min'),
    Input(component_id='time-range', component_property='max')
    ],
    prevent_initial_call=True
)


def update_dropdowns(user_inputs, user_inputs1,time_input, time_min, time_max):


    if user_inputs == [] or user_inputs1 == []:

        fig = go.Figure()

        fig = {}

        min_sl = 1

        max_sl = 10

        value =[1, 10]

    else:

            df = {}

            max1 = []

            min1 = []

            fig = go.Figure()

            if "File" == ctx.triggered_id or "Vect" == ctx.triggered_id:

                for user_input in user_inputs:
                    for user_input1 in user_inputs1:
                        df[user_input] = {}  # Create a nested dictionary for each user_input
                        df[user_input][user_input1] = prb[user_input][user_input1]
                        df[user_input]['t'] = prb[user_input]['t']
                        max1.append(np.round(np.amax(df[user_input]['t'])))
                        min1.append(np.round(np.amin(df[user_input]['t'])))
                        t = df[user_input]['t']
                        V = prb[user_input][user_input1]
                        fig.add_trace(go.Scatter(x=t, y=V, mode='lines',
                                                 name=f"{user_input}{' '}{user_input1}"))

                min_sl = min(min1)
                max_sl = max(max1)
                value = [min_sl, max_sl]

            else:

                for user_input in user_inputs:
                    for user_input1 in user_inputs1:
                        df[user_input] = {}  # Create a nested dictionary for each user_input
                        df[user_input][user_input1] = prb[user_input][user_input1]
                        df[user_input]['t'] = prb[user_input]['t']
                        max1.append(np.round(np.amax(df[user_input]['t'])))
                        min1.append(np.round(np.amin(df[user_input]['t'])))
                        t = df[user_input]['t']
                        V = prb[user_input][user_input1]
                        mask = t < time_input[0]
                        t1 = np.delete(t, np.where(mask))
                        V1 = np.delete(V, np.where(mask))
                        mask = t1 > time_input[1]
                        t2 = np.delete(t1, np.where(mask))
                        V2 = np.delete(V1, np.where(mask))
                        fig.add_trace(go.Scatter(x=t2, y=V2, mode='lines',
                                                 name=f"{user_input}{' '}{user_input1}"))



                value = time_input
            #fig.update_xaxes(rangeslider_visible=True)

            min_sl = min(min1)
            max_sl = max(max1)

            if len(user_inputs) == 1 and len(user_inputs1) == 1:

                fig.update_layout(
                        title=(user_input + " " + user_input1 + " Data"),
                        xaxis_title="Time (s) ",
                        yaxis_title="Velocity (m/s)")
            else:

                fig.update_layout(legend=dict(
                    y = 1,
                    x = 0.5,
                    orientation="h",
                    yanchor="bottom",
                    xanchor="center",
                ))




    return fig, min_sl, max_sl, value


# newnames = {'col1':'hello', 'col2': 'hi'}
# fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
#                                       legendgroup = newnames[t.name],
#                                       hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
#                                      )
#                   )


@app.callback(
     [Output(component_id="File", component_property='value', allow_duplicate=True),
     Output(component_id='Vect', component_property='value', allow_duplicate=True),
     Output(component_id="File", component_property='options', allow_duplicate=True),
     Output(component_id='Vect', component_property='options', allow_duplicate=True),
    Output(component_id = 'Velocity_Graph', component_property = 'figure', allow_duplicate=True)],
    [Input(component_id='clear_files', component_property='n_clicks')],
    prevent_initial_call=True
)



def clear_files(n_clicks):

    if "clear_files" == ctx.triggered_id:

        vect_val = []

        file_val = []

        file_dropdown_options = []

        vect_options = []

        fig = {}


        return file_val, vect_val, file_dropdown_options, vect_options, fig


# Run app
if __name__== '__main__':
    app.run_server(debug=True)

