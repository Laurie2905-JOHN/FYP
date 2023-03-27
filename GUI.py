from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
import numpy as np
import scipy.io as sio
from pathlib import Path, PureWindowsPath



# # Create an instance of tkinter frame or window
# def file_chooser():
#
#     win = Tk()
#     # Set the geometry of tkinter frame
#     win.geometry("800x350")
#     # Add a Label widget
#     label = Label(win, text="Select the Button to Open the File", font=('Aerial 11'))
#     label.pack(pady=30)
#     # Add a Treeview widget to display the selected file names
#     tree = ttk.Treeview(win, columns=('Filename'))
#     tree.heading('#0', text='Index')
#     tree.heading('Filename', text='Filename')
#     tree.pack(side=LEFT, padx=30, pady=30)
#
#     def open_file():
#         files = fd.askopenfilenames(parent=win, title='Choose a File')
#         global file_paths
#         file_paths = list(win.splitlist(files))
#         # Clear the Treeview widget before inserting new file names
#         tree.delete(*tree.get_children())
#         # Update the table with the selected file names
#         global file_names
#         file_names = []
#         for i, file_path in enumerate(file_paths):
#             file_name = file_path.split("/")[-1]
#             file_names.append(file_name)
#             tree.insert('', 'end', text=str(i + 1), values=(file_name,))
#         return file_paths, file_names
#
#     # Add a Button Widget
#     ttk.Button(win, text="Select a File", command=open_file).pack()
#     # Add a Close Button Widget
#
#     def close_window():
#         win.destroy()
#
#     # Add a Label widget for close button
#     label = Label(win, text="Close window once files are added", font=('Aerial 11'))
#     label.pack(pady=30)
#     ttk.Button(win, text="Close", command=close_window).pack()
#
#     win.mainloop()
#
# def cal_velocity(BarnFilePath):
#     import numpy as np
#     import scipy.io as sio
#
#     # Constants
#     rho = 997
#     fs = 16  # sample rate
#
#     # File retrieving
#     ZeroFolder = "C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/"
#     ZeroFile = 'Mon1527.txt'
#     CalFolder = ZeroFolder
#     CalFile = 'IanYawAndDynCalMk2.mat'
#     Cal = sio.loadmat(CalFolder + CalFile)
#
#     Cal = Cal["Cal"]
#     Dynfit = Cal[0][0][0].flatten()
#     Yawfit = Cal[0][0][1].flatten()
#     LDyn = Cal[0][0][2].flatten()
#     LYaw = Cal[0][0][3].flatten()
#     LDyn_0 = Cal[0][0][4].flatten()
#
#     # Evaluating yawcal for a polynomial Cal.Yawfit and dyncal
#     yawcal = np.zeros((91, 2))
#     yawcal[:, 0] = np.linspace(-45, 45, 91)
#     yawcal[:, 1] = np.polyval(Yawfit, yawcal[:, 0])
#     dyncal = np.polyval(Dynfit, yawcal[:, 0])
#     dyncal = dyncal * LDyn_0
#
#     # Importing Zeroes
#     zeros = {}
#     zeros['pr_raw'] = np.loadtxt(ZeroFolder + ZeroFile, delimiter=',')
#     zeros['pr_mean'] = np.mean(zeros['pr_raw'][1300:1708, :], axis=0)
#
#     # Loading actual Barnacle data
#     prb = {}
#     for i, file_path in enumerate(file_paths):
#         file_name = file_path.split("/")[-1]
#         prb[file_name] = {'raw': {}}
#         prb[file_name]['raw'] = np.loadtxt(file_path, delimiter=',')
#         prb[file_name]['raw'] -= zeros['pr_mean']
#         # Data analysis
#         prb[file_name]['denom'] = np.mean(prb[file_name]['raw'][:, :4], axis=1)
#         prb[file_name]['Lyaw'] = (prb[file_name]['raw'][:, 1] - prb[file_name]['raw'][:, 3]) / prb[file_name]['denom']
#         prb[file_name]['Lpitch'] = (prb[file_name]['raw'][:, 0] - prb[file_name]['raw'][:, 2]) / prb[file_name]['denom']
#
#         from scipy import interpolate
#
#         ayaw_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
#         apitch_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
#         prb[file_name]['ayaw'] = ayaw_interp(prb[file_name]['Lyaw'])
#         prb[file_name]['apitch'] = apitch_interp(prb[file_name]['Lpitch'])
#         prb[file_name]['pitchbigger'] = np.abs(prb[file_name]['apitch']) > np.abs(prb[file_name]['ayaw'])
#         prb[file_name]['amax'] = prb[file_name]['pitchbigger'] * prb[file_name]['apitch'] + (1 - prb[file_name]['pitchbigger']) * prb[file_name]['ayaw']
#         ldyn_interp = interpolate.interp1d(yawcal[:, 0], dyncal, kind='linear', fill_value='extrapolate')
#         prb[file_name]['ldyn'] = ldyn_interp(prb[file_name]['amax'])
#
#         # Splitting into velocities
#         prb[file_name]['U1'] = np.sqrt(2 * -prb[file_name]['ldyn'] * np.mean(prb[file_name]['raw'][:, :4], axis=1) / rho)
#         prb[file_name]['U1'][np.imag(prb[file_name]['U1']) > 0] = 0
#         prb[file_name]['Ux'] = prb[file_name]['U1'] * np.cos(np.deg2rad(prb[file_name]['apitch'])) * np.cos(np.deg2rad(prb[file_name]['ayaw']))
#         prb[file_name]['Uy'] = prb[file_name]['U1'] * np.cos(np.deg2rad(prb[file_name]['apitch'])) * np.sin(np.deg2rad(prb[file_name]['ayaw']))
#         prb[file_name]['Uz'] = prb[file_name]['U1'] * np.sin(np.deg2rad(prb[file_name]['apitch']))
#         prb[file_name]['t'] = np.linspace(0, prb[file_name]['raw'].shape[0] / fs, prb[file_name]['raw'].shape[0]);
#
#     return prb
#
# file_chooser()
#
# # print(file_names)
# # print(file_paths)
#
# prb = cal_velocity(file_paths)


app = Dash(__name__)

app.layout = html.Div([

    html.H1("BARNACLE SENSOR ANALYSIS", style={'text-align': 'center'}),

    dcc.Dropdown( id = "Vect",
                        options = ['Ux', 'Uy', 'Uz'],
                        multi=False,
                        value ='Ux',
                        style={'width': "40%"}
                        ),
    html.Div(id='output_container', children=[]),
    html.Br(),


    dcc.Graph(id='Velocity_Graph', figure={}),

    dcc.RangeSlider(
            id='time-range',
            min=round(prb['t'].min()),
            max=round(prb['t'].max()),
            value=[prb['t'].min(), prb['t'].max()],
            tooltip={"placement": "bottom", "always_visible": True},
            ),
])





# marks = {int(timestamp): {'label': date.strftime('%Y-%m-%d')}
                   # for timestamp, date in,
                        #df.set_index('date').resample('M').mean().reset_index().to_dict('split')['data']},
@app.callback(
    [Output(component_id = 'output_container', component_property = 'children')],
    [Output(component_id = 'Velocity_Graph', component_property = 'figure')],
    [Input(component_id = 'Vect', component_property = 'value')],
    [Input(component_id = 'time-range',component_property = 'value')],
    )

def update_graph(user_input, time_input):

    def plot_graph(V,time_input):
        import numpy as np
        mask = prb['t'] < time_input[0]
        prb['t1'] = np.delete(prb['t'], np.where(mask))
        V1 = np.delete(V, np.where(mask))
        mask = prb['t1'] > time_input[1]
        prb['t12'] = np.delete(prb['t1'], np.where(mask))
        V2 = np.delete(V1, np.where(mask))
        fig = px.line(x=prb['t12'], y=V2)
        fig.update_layout(
            title="Plot Title",
            xaxis_title="Time (s) ",
            yaxis_title="Velocity (m/s)")
        return fig

    container = "The data shown is: {}".format(user_input)

    if user_input == 'Ux':
        V = prb['Ux']
        return container, plot_graph(V,time_input)

    elif user_input == 'Uy':
        V = prb['Uy']
        return container, plot_graph(V, time_input)

    elif user_input == 'Uz':
        V = prb['Uz']
        return container, plot_graph(V, time_input)

    return container, fig  # returned objects are assigned to the component property of the Output

# Run app
if __name__== '__main__':
    app.run_server(debug=True)