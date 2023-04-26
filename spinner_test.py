import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import asyncio

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Button(
        html.Div([
            html.Div(dbc.Spinner(id="spinner", size="sm"), id="spinner-wrapper", style={"display": "none"}),
            html.Span(id="button-text", children="Analyse Selected Files"),
        ]),
        id='newfile',
        outline=True,
        color="primary",
        className="me-1",
    ),
    dcc.Store(id='store', storage_type='memory'),  # Add a dcc.Store component
])

@app.callback(
    Output('store', 'data'),
    Input('newfile', 'n_clicks'),
    State('store', 'data')
)
async def update_store(n_clicks, data):
    if n_clicks is None:
        return 0
    await asyncio.sleep(5)  # Use asyncio.sleep to simulate a lengthy process without blocking the main thread
    return n_clicks

@app.callback(
    [Output("spinner-wrapper", "style"), Output("button-text", "children")],
    [Input("store", "data")],
)
def toggle_spinner(store_data):
    if store_data % 2 == 1:
        return {"display": "inline-block"}, "Calculating"  # Show the spinner by setting 'display' to 'inline-block'
    else:
        return {"display": "none"}, "Analyse Selected Files"  # Hide the spinner by setting 'display' to 'none'

if __name__ == "__main__":
    app.run_server(debug=True)
