from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
from App.single_module.single_analysis import analyse_single_article, entity_types_plot_single, \
    most_common_entities_plot_single, sentiment_dist_plot_single, most_common_words_plot_single
import pandas as pd
from App.utils import analyze_file_reactive, render_uploaded_text_content, generate_header, list_files_in_folder
from colors import my_orange
from pathlib import Path

here = Path(__file__).parent.parent.parent

single_module_ui = ui.tags.div(
    ui.tags.div(
        ui.tags.div(
            ui.tags.h3(ui.output_text("uploaded_text_header")),
            ui.output_ui("uploaded_text_content"),
            ui.output_ui("show_view_more_button"),
            class_="article-container single-article-container"
        ),
        ui.output_ui("single_mode_plots"),
        class_="main-left-container single-module"
    ),
    ui.output_ui("right_container_single"),
    ui.busy_indicators.options(spinner_type="dots2", spinner_color=my_orange, spinner_size="50px", fade_opacity=1),
    class_="main-container",
)


def single_module_server(input, output, session):
    view_full_text = reactive.Value(False)
    right_container_visible_single = reactive.Value(True)
    article_analysis = reactive.Value(None)
    entity_sentiments = reactive.Value(pd.DataFrame())
    sentiment_sentences = reactive.Value(pd.DataFrame())
    article_header = reactive.Value("No file uploaded")
    selected_file_value = reactive.Value("None")

    @reactive.Effect
    @reactive.event(input.view_full_text)
    def toggle_view_full_text():
        view_full_text.set(not view_full_text.get())
        new_label = "View less" if view_full_text.get() else "View more"
        session.send_input_message("view_full_text", {"label": new_label})

    @output
    @render.text
    def uploaded_text_header():
        return generate_header(input.file_upload, input.file_select, article_analysis, article_header)

    @output
    @render.ui
    def uploaded_text_content():
        return render_uploaded_text_content(
            article_analysis=article_analysis,
            view_full_text=view_full_text,
            file_upload=input.file_upload,
            file_select=input.file_select
        )

    @output
    @render.ui
    def single_mode_plots():
        if input.file_upload() or input.file_select() != "None":
            return ui.div(
                output_widget("entity_types_single_plot"),
                output_widget("most_common_entities_single_plot"),
                output_widget("sentiment_dist_single_plot"),
                ui.output_plot("most_common_words_single_plot"),
                class_="plots-container"
            )
        return ui.div()

    @output
    @render_widget
    def entity_types_single_plot():
        if isinstance(entity_sentiments.get(), pd.DataFrame):
            return entity_types_plot_single(entity_sentiments.get())

    @output
    @render_widget
    def most_common_entities_single_plot():
        if isinstance(entity_sentiments.get(), pd.DataFrame):
            return most_common_entities_plot_single(entity_sentiments.get())

    @output
    @render_widget
    def sentiment_dist_single_plot():
        if isinstance(sentiment_sentences.get(), pd.DataFrame) and isinstance(entity_sentiments.get(), pd.DataFrame):
            return sentiment_dist_plot_single(entity_sentiments.get(), sentiment_sentences.get())

    @output
    @render.plot
    def most_common_words_single_plot():
        if isinstance(sentiment_sentences.get(), pd.DataFrame):
            return most_common_words_plot_single(sentiment_sentences.get(), article="")

    @reactive.Effect
    @reactive.event(input.hide_container_button_single)
    def toggle_container_visibility_single():
        right_container_visible_single.set(not right_container_visible_single.get())

    @reactive.Effect
    @reactive.event(input.file_upload, input.file_select)
    def auto_hide_container_single():
        if input.file_upload() or (input.file_select() != "None"):
            right_container_visible_single.set(False)

    @output
    @render.ui
    def right_container_single():
        folder_path = here / "BRAT_Data"
        file_choices = list_files_in_folder(folder_path)
        display_choices = ["None"] + [display for _, display in file_choices]
        if right_container_visible_single.get():
            return ui.div(
                ui.input_select("file_select", "Select article", choices=display_choices,
                                selected=selected_file_value.get()),
                ui.input_file("file_upload", "UPLOAD ARTICLE"),
                ui.input_action_button("hide_container_button_single", "Hide Menu", class_="btn btn-secondary"),
                class_="main-right-container",
                id="main-right-container-single"
            )
        else:
            return ui.div(
                ui.input_action_button("hide_container_button_single", "Show Menu", class_="show-container-tab"),
                class_="main-right-container hidden",
                id="main-right-container-single"
            )

    @output
    @render.ui
    def show_view_more_button():
        if article_analysis.get():
            return ui.input_action_button("view_full_text", "View more", class_="btn btn-secondary view-button")
        return ui.div()

    @reactive.Effect
    @reactive.event(input.file_upload, input.file_select)
    async def analyze_single_file():
        await analyze_file_reactive(
            file_input=input.file_upload,
            file_select=input.file_select,
            article_analysis=article_analysis,
            entity_sentiments=entity_sentiments,
            sentiment_sentences=sentiment_sentences
        )
