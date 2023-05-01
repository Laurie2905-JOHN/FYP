# Importing App
from dash_app import app
# Import libs
from dash import dash, dcc, dash_table, html
import dash_bootstrap_components as dbc

if __name__ == '__main__':
    app.run_server(debug=True)