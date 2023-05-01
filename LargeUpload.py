# This script creates a Dash application that allows users to upload large files.
# The file upload component has a unique ID and is configured to accept .txt and .csv
# files with a maximum size of 1800 MB. The uploaded files are stored in a specified
# root folder. The callback function is called upon completion of the file upload and
# prints the file path of the uploaded file.

import uuid
import dash_uploader as du
import dash
from dash import html
from dash.dependencies import Output

# Create a Dash application
app = dash.Dash(__name__)

# Set the root folder for file uploads
UPLOAD_FOLDER_ROOT = r'C:\Users\lauri\Desktop'

# Configure the uploader component with the application and root folder
du.configure_upload(
    app,
    str(UPLOAD_FOLDER_ROOT),
    use_upload_id=True,
)

# Define a function to create an upload component with a unique ID
def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=30000,  # 1800 Mb
        filetypes=['txt', 'csv'],
        upload_id=uuid.uuid1(),  # Unique session id
    )

# Define a function to create the layout of the Dash application
def get_app_layout():
    return html.Div(
        [
            html.H1('Large Upload Example'),
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

# Define a callback function to handle file uploads
@du.callback(
    output=Output("callback-output", "children"),
    id="dash-uploader",
)
def callback_on_completion(status):
    # Get the file path of the uploaded file
    BarnFilePath = [str(x) for x in status.uploaded_files]
    print(BarnFilePath)
    return 'done'

# Set the layout of the application using the get_app_layout function
app.layout = get_app_layout

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)

