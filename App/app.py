from pathlib import Path
import seaborn as sns
from shiny import App, render, ui, reactive

from htmltools import tags, Tag

single_module = ui.tags.div(
    ui.tags.div(
        ui.tags.h3(ui.output_text("uploaded_text_header")),
        ui.tags.div(ui.output_text_verbatim("uploaded_text_content"), class_="text-content"),
        ui.input_action_button("view_full_text", "View more", class_="btn btn-primary view-button"),
        class_="main-left-container"
    ),
    ui.tags.div(
        ui.input_file("file_upload", "Upload a text file"),
        class_="main-right-container"
    ),
    class_="main-container",
)

page_dependencies = ui.tags.head(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css")
)

page_layout = ui.page_navbar(
    ui.nav_spacer(),  # Push the navbar items to the right
    ui.nav_panel("SINGLE", single_module),
    ui.nav_panel("DOUBLE", "This is the second 'page'."),
    ui.nav_panel("ALL", "This is the third 'page'."),
    title="PRESS ARTICLES EXPLORATION",
    footer=ui.tags.div(
        ui.tags.div("Łukasz Grabarski & Marta Szuwarska", class_="footer")
    )
)

app_ui = ui.page_fluid(
    page_dependencies,
    page_layout,
    title="PRESS ARTICLES EXPLORATION",
)


def server(input, output, session):
    view_full_text = reactive.Value(False)
    @reactive.Effect
    @reactive.event(input.view_full_text)
    def toggle_view_full_text():
        # Toggle the full text view state
        view_full_text.set(not view_full_text.get())
        # Update the button label
        new_label = "View less" if view_full_text.get() else "View more"
        session.send_input_message("view_full_text", {"label": new_label})

    @output
    @render.text
    def uploaded_text_header():
        file_info = input.file_upload()
        if file_info is None or len(file_info) == 0:
            return "No file uploaded"
        with open(file_info[0]["datapath"], "r") as file:
            lines = file.readlines()
            if len(lines) < 7:
                return "Invalid file format"
            published_time = lines[0].split(": ", 1)[1].strip()
            title = lines[1].split(": ", 1)[1].strip()
            categories = lines[2].split(";")
            category1 = categories[0].split(": ", 1)[1].strip()
            category2 = categories[1].split(": ", 1)[1].strip()
        return f"{title} -- {published_time} -- {category1} / {category2}"

    @output
    @render.text
    def uploaded_text_content():
        file_info = input.file_upload()
        if file_info is None or len(file_info) == 0:
            return "No file uploaded"
        with open(file_info[0]["datapath"], "r") as file:
            lines = file.readlines()
            if len(lines) < 8:
                return "Invalid file format"

            if view_full_text.get():
                content = "\n".join(line.strip() for line in lines[7:])
            else:
                first_five_sentences = lines[7:12]
                content = "\n".join(line.strip() for line in first_five_sentences)

            return content


www_dir = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=www_dir)
