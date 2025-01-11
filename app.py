from shiny import App, ui
from App.single_module.single_module import single_module_server
from App.double_module.double_module import double_module_server
from App.all_module.all_module import all_module_server
from App.about_module.about_module import about_module_server
from App.ui_components import app_ui
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
here = Path(__file__).parent


def server(input, output, session):
    # Call server functions for each module
    single_module_server(input, output, session)
    double_module_server(input, output, session)
    all_module_server(input, output, session)
    about_module_server(input, output, session)


www_dir = Path(__file__).parent / "App/www"
app = App(app_ui, server, static_assets=www_dir)
