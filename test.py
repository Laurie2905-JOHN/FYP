import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
])

def parse_contents(contents, filename, date):
    try:
        if isinstance(contents, tuple):
            content_string = contents[0]
        else:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return decoded


@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(decoded_data, list_of_names, list_of_dates):
    if decoded_data is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(decoded_data, list_of_names, list_of_dates)]
        print(decoded_data)
        return children

if __name__ == '__main__':
    app.run_server(debug=True)





