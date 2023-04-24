from pathlib import Path
import uuid
import dash_bootstrap_components as dbc
import dash_uploader as du
import dash
from dash import html, dash_table
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

UPLOAD_FOLDER_ROOT = Path("./tmp") / "uploads"
du.configure_upload(
    app,
    str(UPLOAD_FOLDER_ROOT),
    use_upload_id=True,
)

def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=30000,  # 1800 Mb
        filetypes=['txt', 'csv'],
        upload_id=uuid.uuid1(),  # Unique session id
    )


def get_app_layout():

    return html.Div(
        [  dbc.Col(

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
            html.H1('Demo'),
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


@du.callback(
    output=Output("callback-output", "children"),
    id="dash-uploader",
)
def callback_on_completion(status):
    return html.Ul([html.Li(str(x)) for x in status.uploaded_files])

# get_app_layout is a function
# This way we can use unique session id's as upload_id's
app.layout = get_app_layout






if __name__ == '__main__':
    app.run_server(debug=True)

