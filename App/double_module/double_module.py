from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
import pandas as pd
from App.utils import analyze_file_reactive, render_uploaded_text_content, generate_header, list_files_in_folder
from App.double_module.double_analysis import entity_types_plot_double, most_common_entities_plot_double, \
    sentiment_dist_plot_double
from App.single_module.single_analysis import most_common_words_plot_single
from colors import my_orange
from pathlib import Path

here = Path(__file__).parent.parent.parent

double_module_ui = ui.tags.div(
    ui.tags.div(
        ui.tags.div(
            ui.tags.h3(ui.output_text("uploaded_text_header_1")),
            ui.output_ui("uploaded_text_content_1"),
            ui.output_ui("show_view_more_button_1"),
            class_="article-container"
        ),
        ui.tags.div(
            ui.tags.h3(ui.output_text("uploaded_text_header_2")),
            ui.output_ui("uploaded_text_content_2"),
            ui.output_ui("show_view_more_button_2"),
            class_="article-container"
        ),
        ui.output_ui("double_mode_plots"),
        class_="main-left-container"
    ),
    ui.output_ui("right_container_double"),
    ui.busy_indicators.options(spinner_type="dots2", spinner_color=my_orange, spinner_size="50px", fade_opacity=1),
    class_="main-container"
)


def double_module_server(input, output, session):
    view_full_text_1 = reactive.Value(False)
    view_full_text_2 = reactive.Value(False)
    right_container_visible_double = reactive.Value(True)
    article_analysis_1 = reactive.Value(None)
    article_analysis_2 = reactive.Value(None)
    entity_sentiments_1 = reactive.Value(pd.DataFrame())
    entity_sentiments_2 = reactive.Value(pd.DataFrame())
    sentiment_sentences_1 = reactive.Value(pd.DataFrame())
    sentiment_sentences_2 = reactive.Value(pd.DataFrame())
    article_header_1 = reactive.Value("No file uploaded")
    article_header_2 = reactive.Value("No file uploaded")
    selected_file_value_1 = reactive.Value("None")
    selected_file_value_2 = reactive.Value("None")

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
    def uploaded_text_header_1():
        return generate_header(input.file_upload_1, input.file_select_1, article_analysis_1, article_header_1)

    @output
    @render.text
    def uploaded_text_header_2():
        return generate_header(input.file_upload_2, input.file_select_2, article_analysis_2, article_header_2)

    @output
    @render.ui
    def uploaded_text_content_1():
        return render_uploaded_text_content(
            article_analysis=article_analysis_1,
            view_full_text=view_full_text_1,
            file_upload=input.file_upload_1,
            file_select=input.file_select_1
        )

    @output
    @render.ui
    def uploaded_text_content_2():
        return render_uploaded_text_content(
            article_analysis=article_analysis_2,
            view_full_text=view_full_text_2,
            file_upload=input.file_upload_2,
            file_select=input.file_select_2,
        )

    @output
    @render.ui
    def show_view_more_button_1():
        if article_analysis_1.get():
            return ui.input_action_button("view_full_text_1", "View more", class_="btn btn-secondary view-button")
        return ui.div()

    @output
    @render.ui
    def show_view_more_button_2():
        if article_analysis_2.get():
            return ui.input_action_button("view_full_text_2", "View more", class_="btn btn-secondary view-button")
        return ui.div()

    @output
    @render.ui
    def double_mode_plots():
        if ((input.file_upload_1() or input.file_select_1() != "None") and (
                input.file_upload_2() or input.file_select_2() != "None")) or (
                article_analysis_1.get() is not None and article_analysis_2.get() is not None
        ):
            return ui.div(
                output_widget("entity_types_double_plot"),
                output_widget("most_common_entities_double_plot"),
                output_widget("sentiment_dist_sentence_double_plot"),
                output_widget("sentiment_dist_entity_double_plot"),
                ui.output_plot("most_common_words_double_plot_1"),
                ui.output_plot("most_common_words_double_plot_2"),
                class_="plots-container"
            )
        return ui.div()

    @output
    @render_widget
    def entity_types_double_plot():
        if isinstance(entity_sentiments_1.get(), pd.DataFrame) and isinstance(entity_sentiments_2.get(), pd.DataFrame):
            return entity_types_plot_double(entity_sentiments_1.get(), entity_sentiments_2.get())

    @output
    @render_widget
    def most_common_entities_double_plot():
        if isinstance(entity_sentiments_1.get(), pd.DataFrame) and isinstance(entity_sentiments_2.get(), pd.DataFrame):
            return most_common_entities_plot_double(entity_sentiments_1.get(), entity_sentiments_2.get())

    @output
    @render_widget
    def sentiment_dist_sentence_double_plot():
        if isinstance(sentiment_sentences_1.get(), pd.DataFrame) and isinstance(sentiment_sentences_2.get(),
                                                                                pd.DataFrame):
            return sentiment_dist_plot_double(sentiment_sentences_1.get(), sentiment_sentences_2.get(),
                                              base="Sentences")

    @output
    @render_widget
    def sentiment_dist_entity_double_plot():
        if isinstance(entity_sentiments_1.get(), pd.DataFrame) and isinstance(entity_sentiments_2.get(), pd.DataFrame):
            return sentiment_dist_plot_double(entity_sentiments_1.get(), entity_sentiments_2.get(), base="Entities")

    @output
    @render.plot
    def most_common_words_double_plot_1():
        if isinstance(sentiment_sentences_1.get(), pd.DataFrame):
            return most_common_words_plot_single(sentiment_sentences_1.get(), article="Article 1")

    @output
    @render.plot
    def most_common_words_double_plot_2():
        if isinstance(sentiment_sentences_2.get(), pd.DataFrame):
            return most_common_words_plot_single(sentiment_sentences_2.get(), article="Article 2")

    @reactive.Effect
    @reactive.event(input.hide_container_button_double)
    def toggle_container_visibility_double():
        right_container_visible_double.set(not right_container_visible_double.get())

    @reactive.Effect
    @reactive.event(input.file_upload_1, input.file_upload_2, input.file_select_1, input.file_select_2)
    def auto_hide_container_double():
        if (input.file_upload_1() or (input.file_select_1() != "None")) and (
                input.file_upload_2() or (input.file_select_2() != "None")):
            right_container_visible_double.set(False)

    @output
    @render.ui
    def right_container_double():
        folder_path = here / "BRAT_Data"
        file_choices = list_files_in_folder(folder_path)
        display_choices = ["None"] + [display for _, display in file_choices]
        if right_container_visible_double.get():
            return ui.div(
                ui.input_select("file_select_1", "Select the first article", choices=display_choices,
                                selected=selected_file_value_1.get()),
                ui.input_file("file_upload_1", "Upload the first article"),
                ui.input_select("file_select_2", "Select the second article", choices=display_choices,
                                selected=selected_file_value_2.get()),
                ui.input_file("file_upload_2", "Upload the second article"),
                ui.input_action_button("hide_container_button_double", "Hide Menu", class_="btn btn-secondary"),
                class_="main-right-container",
                id="main-right-container-double"
            )
        else:
            return ui.div(
                ui.input_action_button("hide_container_button_double", "Show Menu", class_="show-container-tab"),
                class_="main-right-container hidden",
                id="main-right-container-double"
            )

    @reactive.Effect
    @reactive.event(input.file_upload_1, input.file_select_1)
    async def analyze_first_file():
        await analyze_file_reactive(
            file_input=input.file_upload_1,
            file_select=input.file_select_1,
            article_analysis=article_analysis_1,
            entity_sentiments=entity_sentiments_1,
            sentiment_sentences=sentiment_sentences_1
        )

    @reactive.Effect
    @reactive.event(input.file_upload_2, input.file_select_2)
    async def analyze_second_file():
        await analyze_file_reactive(
            file_input=input.file_upload_2,
            file_select=input.file_select_2,
            article_analysis=article_analysis_2,
            entity_sentiments=entity_sentiments_2,
            sentiment_sentences=sentiment_sentences_2
        )

    @reactive.Effect
    @reactive.event(input.file_select_1)
    def set_selected_file_value_1():
        selected_file_value_1.set(input.file_select_1())

    @reactive.Effect
    @reactive.event(input.file_select_2)
    def set_selected_file_value_2():
        selected_file_value_2.set(input.file_select_2())
