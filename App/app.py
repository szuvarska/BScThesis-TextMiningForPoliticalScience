from pathlib import Path
import seaborn as sns
from shiny import App, render, ui, reactive
from htmltools import tags, Tag
import numpy as np
import matplotlib.pyplot as plt

here = Path(__file__).parent


def handle_file_upload(file_input):
    file_info = file_input()
    if file_info is None or len(file_info) == 0:
        return None, None
    with open(file_info[0]["datapath"], "r") as file:
        lines = file.readlines()
        if len(lines) < 7:
            return "Invalid file format", None
        published_time = lines[0].split(": ", 1)[1].strip()
        title = lines[1].split(": ", 1)[1].strip()
        categories = lines[2].split(";")
        category1 = categories[0].split(": ", 1)[1].strip()
        category2 = categories[1].split(": ", 1)[1].strip()
    return f"{title} -- {published_time} -- {category1} / {category2}", lines


def render_article_content(lines, view_full_text):
    if len(lines) < 8:
        return "Invalid file format"
    if view_full_text.get():
        content = "\n".join(line.strip() for line in lines[7:])
    else:
        first_five_sentences = lines[7:12]
        content = "\n".join(line.strip() for line in first_five_sentences)
    return content


def generate_histogram():
    data = np.random.randn(1000)  # Random data for the histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(data, color="blue", kde=True)  # Use blue color palette
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Example Histogram')
    return plt


single_module = ui.tags.div(
    ui.tags.div(
        ui.tags.div(
            ui.tags.h3(ui.output_text("uploaded_text_header")),
            ui.tags.div(ui.output_text_verbatim("uploaded_text_content"), class_="text-content"),
            ui.input_action_button("view_full_text", "View more", class_="btn btn-primary view-button"),
            class_="article-container single-article-container"
        ),
        ui.output_ui("single_mode_plots"),
        class_="main-left-container single-module"
    ),
    ui.tags.div(
        ui.input_file("file_upload", "UPLOAD ARTICLE"),
        class_="main-right-container",
        id="main-right-container"
    ),
    class_="main-container",
)

double_module = ui.tags.div(
    ui.tags.div(
        ui.tags.div(
            ui.tags.h3(ui.output_text("uploaded_text_header_1")),
            ui.tags.div(ui.output_text_verbatim("uploaded_text_content_1"), class_="text-content"),
            ui.input_action_button("view_full_text_1", "View more", class_="btn btn-primary view-button"),
            class_="article-container"
        ),
        ui.tags.div(
            ui.tags.h3(ui.output_text("uploaded_text_header_2")),
            ui.tags.div(ui.output_text_verbatim("uploaded_text_content_2"), class_="text-content"),
            ui.input_action_button("view_full_text_2", "View more", class_="btn btn-primary view-button"),
            class_="article-container"
        ),
        ui.output_ui("double_mode_plots"),
        class_="main-left-container"
    ),
    ui.tags.div(
        ui.input_file("file_upload_1", "Upload article"),
        ui.input_file("file_upload_2", "Upload the second article"),
        class_="main-right-container"
    ),
    class_="main-container"
)

page_dependencies = ui.tags.head(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css")
)

page_layout = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel("SINGLE", single_module),
    ui.nav_panel("DOUBLE", double_module),
    ui.nav_panel("ALL", "This is the third 'page'."),
    title="PRESS ARTICLES EXPLORATION",
    footer=ui.tags.div(
        ui.tags.div("Łukasz Grabarski & Marta Szuwarska", class_="footer")
    ),
)

app_ui = ui.page_fluid(
    page_dependencies,
    page_layout,
    title="PRESS ARTICLES EXPLORATION",
)


def server(input, output, session):
    view_full_text = reactive.Value(False)
    layout_state = reactive.Value(True)
    view_full_text_1 = reactive.Value(False)
    view_full_text_2 = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.view_full_text)
    def toggle_view_full_text():
        view_full_text.set(not view_full_text.get())
        new_label = "View less" if view_full_text.get() else "View more"
        session.send_input_message("view_full_text", {"label": new_label})

    @reactive.Effect
    @reactive.event(input.view_full_text_1)
    def toggle_view_full_text_1():
        view_full_text_1.set(not view_full_text_1.get())
        new_label = "View less" if view_full_text_1.get() else "View more"
        session.send_input_message("view_full_text_1", {"label": new_label})

    @reactive.Effect
    @reactive.event(input.view_full_text_2)
    def toggle_view_full_text_2():
        view_full_text_2.set(not view_full_text_2.get())
        new_label = "View less" if view_full_text_2.get() else "View more"
        session.send_input_message("view_full_text_2", {"label": new_label})

    @output
    @render.text
    def uploaded_text_header():
        header, _ = handle_file_upload(input.file_upload)
        if header:
            return header
        return "No file uploaded"

    @output
    @render.text
    def uploaded_text_content():
        _, lines = handle_file_upload(input.file_upload)
        if lines:
            return render_article_content(lines, view_full_text)
        return "No file uploaded"

    @output
    @render.text
    def uploaded_text_header_1():
        header, _ = handle_file_upload(input.file_upload_1)
        if header:
            return header
        return "No file uploaded"

    @output
    @render.text
    def uploaded_text_content_1():
        _, lines = handle_file_upload(input.file_upload_1)
        return render_article_content(lines, view_full_text_1) if lines else "No file uploaded"

    @output
    @render.text
    def uploaded_text_header_2():
        header, _ = handle_file_upload(input.file_upload_2)
        if header:
            return header
        return "No file uploaded"

    @output
    @render.text
    def uploaded_text_content_2():
        _, lines = handle_file_upload(input.file_upload_2)
        return render_article_content(lines, view_full_text_2) if lines else "No file uploaded"

    @output
    @render.ui
    def single_mode_plots():
        return ui.div(
            ui.img(src=f"plot1.png", class_="plot-image"),
            ui.img(src=f"plot2.png", class_="plot-image"),
            class_="plots-container"
        )

    @output
    @render.ui
    def double_mode_plots():
        return ui.div(
            ui.img(src=f"plot1.png", class_="plot-image"),
            ui.img(src=f"plot2.png", class_="plot-image"),
            class_="plots-container"
        )

    @reactive.Effect
    def update_layout():
        file_info = input.file_upload()
        if file_info and len(file_info) > 0:
            layout_state.set(False)
        else:
            layout_state.set(True)

    @output
    @render.ui
    def right_container():
        if layout_state.get():
            return ui.div(
                ui.input_file("file_upload", "UPLOAD ARTICLE"),
                class_="main-right-container"
            )


www_dir = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=www_dir)
