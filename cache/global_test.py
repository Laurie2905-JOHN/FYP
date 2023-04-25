import dash
import dash_core_components as dcc
import dash_html_components as html
import time
import plotly.express as px

from dash.dependencies import Output, Input
from flask_caching.backends import FileSystemCache
from dash_extensions.callback import CallbackCache, Trigger

# Create app.
app = dash.Dash(prevent_initial_callbacks=True)
app.layout = html.Div([
    html.Button("Query data", id="btn"), dcc.Dropdown(id="dd"), dcc.Graph(id="graph"),
    dcc.Loading(dcc.Store(id="store"), fullscreen=True, type="dot")
])
# Create (server side) cache. Works with any flask caching backend.
cc = CallbackCache(cache=FileSystemCache(cache_dir="cache"))


@cc.cached_callback(Output("store", "data"), [Trigger("btn", "n_clicks")])  # Trigger is like Input, but excluded from args
def query_data():
    time.sleep(1)  # sleep to emulate a database call / a long calculation
    return px.data.gapminder()


@cc.callback(Output("dd", "options"), [Input("store", "data")])
def update_dd(df):
    return [{"label": column, "value": column} for column in df["year"]]


@cc.callback(Output("graph", "figure"), [Input("store", "data"), Input("dd", "value")])
def update_graph(df, value):
    df = df.query("year == {}".format(value))
    return px.sunburst(df, path=['continent', 'country'], values='pop', color='lifeExp', hover_data=['iso_alpha'])


# This call registers the callbacks on the application.
cc.register(app)

if __name__ == '__main__':
    app.run_server()