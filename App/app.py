from pathlib import Path
import seaborn as sns
from shiny import App, render, ui, reactive
from htmltools import tags, Tag
import numpy as np
import matplotlib.pyplot as plt

here = Path(__file__).parent


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
            class_="article-container"
        ),
        ui.output_ui("left_plots"),  # Include the plots here
        class_="main-left-container"
    ),
    ui.tags.div(
        ui.input_file("file_upload", "UPLOAD ARTICLE"),
        class_="main-right-container",
        id="main-right-container"
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

    @output
    @render.ui
    def left_plots():
        # Generate the plots and convert to static images
        plot1 = generate_histogram()
        plot2 = generate_histogram()

        # Save the plots as images
        plot1_path = "www/plot1.png"
        plot2_path = "www/plot2.png"
        plot1.savefig(plot1_path)
        plot2.savefig(plot2_path)

        # Return the div containers for the plots
        return ui.div(
            ui.img(src=f"plot1.png", class_="plot-image"),
            ui.img(src=f"plot2.png", class_="plot-image"),
            class_="plots-container"
        )

    # Watch for file upload and update visibility of right container
    @reactive.Effect
    def update_layout():
        file_info = input.file_upload()
        if file_info and len(file_info) > 0:  # Check if file is uploaded
            layout_state.set(False)  # Hide the right container
        else:
            layout_state.set(True)  # Show the right container

    # Right container UI (conditionally rendered)
    @output
    @render.ui
    def right_container():
        if layout_state.get():  # If the right container should be visible
            return ui.div(
                ui.input_file("file_upload", "UPLOAD ARTICLE"),
                class_="main-right-container"
            )
        else:
            return ui.div(style="display:none;", id="main_right_container")  # Hide the right container

    # Right tab UI (conditionally rendered)
    @output
    @render.ui
    def right_tab():
        if not layout_state.get():  # If the right container is hidden
            return ui.div(
                "Show Right Container",
                class_="right-tab show",
                onclick="document.getElementById('main_right_container').style.display = 'block'; document.querySelector('.right-tab').style.display = 'none';"
            )
        else:
            return ui.div(style="display:none;", class_="right-tab")  # Hide the tab if the container is visible


www_dir = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=www_dir)
