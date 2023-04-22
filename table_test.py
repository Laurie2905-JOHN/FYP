import random
import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

table_header = [
    html.Thead(html.Tr([html.Th("ID"), html.Th("Value 1"), html.Th("Value 2")]))
]

table_body = html.Tbody([])

table = dbc.Table(
    table_header + [table_body],
    bordered=True,
    striped=True,
    hover=True,
    responsive=True,
    id="data_table",
)

app.layout = dbc.Container(
    [
        html.H1("Bootstrap Table Update Example"),
        table,
        dbc.Button("Update Table", id="update_button", color="primary", className="mt-3"),
    ],
    className="mt-5",
)

@app.callback(
    Output("data_table", "children"),
    Input("update_button", "n_clicks"),
    State("data_table", "children"),
)
def update_table(n_clicks, current_table_data):
    if n_clicks is None:
        return current_table_data

    new_data = [
        html.Tr(
            [
                html.Td(i + 1),
                html.Td(random.randint(1, 100)),
                html.Td(random.random()),
            ]
        )
        for i in range(5)
    ]

    return table_header + [html.Tbody(new_data)]

if __name__ == "__main__":
    app.run_server(debug=True)