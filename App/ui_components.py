from shiny import ui
from App.single_module.single_module import single_module_ui
from App.double_module.double_module import double_module_ui
from App.all_module.all_module import all_module_ui
from App.about_module.about_module import about_module_ui

page_dependencies = ui.head_content(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
    ui.tags.link(rel="icon", type="image/png", href="logomini.png")
)

page_layout = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel("Single", single_module_ui),
    ui.nav_panel("Double", double_module_ui),
    ui.nav_panel("All", all_module_ui),
    ui.nav_panel("About", about_module_ui),
    title="Global Times: Articles Analysis",
    footer=ui.tags.div(
        ui.tags.div("Łukasz Grabarski & Marta Szuwarska", class_="footer")
    )
)

app_ui = ui.page_fluid(
    page_dependencies,
    page_layout,
    title="Global Times: Articles Analysis",  # PRESS ARTICLES EXPLORATION
)
