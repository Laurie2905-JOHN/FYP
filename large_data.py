
import os
import numpy as np

# Set your custom file path here
file_path = "C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/"

# Read the input file and create a NumPy array
with open(os.path.join(file_path, "Example 1.txt"), "r") as file:
    data_array = np.loadtxt(file, delimiter=',')

# Set your sample rate (fs) and the time threshold here
fs = 16  # Example sample rate

# Calculate the Time array
Time = np.linspace(0, len(data_array) / fs, len(data_array))

# Calculate the total time duration of the data
total_time = Time[-1]

# Set the desired time duration
desired_time = 10

# Example desired time duration in seconds

# Calculate how many times the data should be repeated to fill the desired time
repeat_count = int(np.ceil(desired_time / total_time))

# Repeat the data to fill the desired time
extended_data = np.tile(data_array, (repeat_count, 1))

# Save the extended data as a new text file
with open(os.path.join(file_path, "output.txt"), "w") as file:
    np.savetxt(file, extended_data, delimiter=',', fmt='%s')

    dcc.Upload(
        id='Cal_Upload_files',
        children=html.Div([
            html.A('Select Calibration File')
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
        multiple=False
    ),

    dcc.Upload(
        id='Zero_Upload_files',
        children=html.Div([
            html.A('Select Zero File')
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
        multiple=False
    )
], width = 3),


@app.callback(
    Output(component_id = "tab-content", component_property = "children"),
    [Input(component_id ="tabs", component_property = "active_tab")]
)

def render_tab_content(active_tab):
    print(active_tab)

    if active_tab is not None:
        if active_tab == "Barn_up":
            return dbc.Row([

                dbc.Row([

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

            elif active_tab == "Zero_down":
            return dbc.Row([



        return "No tab selected"