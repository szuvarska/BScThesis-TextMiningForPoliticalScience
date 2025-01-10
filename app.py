import os
import asyncio
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging messages
from pathlib import Path
import pandas as pd
import seaborn as sns
from shiny import App, render, ui, reactive
import numpy as np
import matplotlib.pyplot as plt
from App.plots import (generate_entity_types_plot, generate_most_common_entities_plot, generate_sentiment_dist_plot,
                       generate_sentiment_over_time_plot, generate_sentiment_word_cloud_plot,
                       generate_sentiment_dist_per_target_plot,
                       generate_sentiment_over_time_per_target_plot, generate_sentiment_dist_over_time_by_target_plot,
                       generate_word_count_distribution_plot, generate_sentence_count_distribution_plot,
                       generate_top_N_common_words_plot, generate_top_N_common_pos_plot, generate_pos_wordclouds_plot,
                       generate_community_graph, generate_pos_choices, generate_bigrams_plot, generate_concordance,
                       generate_keywords_over_time_plot, generate_stacked_keywords_over_time_plot, generate_keywords)
from shinywidgets import output_widget, render_widget
from App.single_analysis import analyse_single_article, entity_types_plot_single, most_common_entities_plot_single, \
    sentiment_dist_plot_single, most_common_words_plot_single
from App.double_analysis import entity_types_plot_double, most_common_entities_plot_double, sentiment_dist_plot_double
from colors import main_color, my_red, my_blue, my_gray, my_green, my_yellow, my_orange

here = Path(__file__).parent


def handle_file_upload(file_input):
    file_info = file_input()
    if file_info is None or len(file_info) == 0:
        return None, None
    try:
        with open(file_info[0]["datapath"], "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
            if len(lines) < 7:
                return "Invalid file format", None
            published_time = lines[0].split(": ", 1)[1].strip()
            title = lines[1].split(": ", 1)[1].strip()
            categories = lines[2].split(";")
            category1 = categories[0].split(": ", 1)[1].strip()
            category2 = categories[1].split(": ", 1)[1].strip()
        return f"{title} -- {published_time} -- {category1} / {category2}", lines
    except UnicodeDecodeError:
        return "File encoding error", None


def generate_histogram():
    data = np.random.randn(1000)
    plt.figure(figsize=(6, 4))
    sns.histplot(data, color="blue", kde=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Example Histogram')
    return plt


def collapsible_section(header, button_id, plot_id):
    return ui.div(
        ui.div(
            ui.tags.h3(header, style="display: inline;"),
            ui.input_action_button(button_id, "⯆", class_="toggle-button",
                                   style="font-size: 20px; display: inline; margin-left: 10px;"),
            style="display: flex; align-items: center; margin-bottom: 10px;",
            class_="collapsible-section-header"
        ),
        ui.output_ui(plot_id),
        class_="collapsible-section"
    )


def list_files_in_folder(folder_path):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                display_path = relative_path.replace('Articles_for_Agnieszka/', '').replace('Articles_for_Ania/', '')
                file_list.append((relative_path, display_path))
    return file_list


def analyze_file(file_path_or_lines, is_uploaded, progress_callback=None):
    """
    Blocking file analysis function that updates progress.
    Args:
        file_path_or_lines: Lines of file (uploaded) or file path (selected).
        is_uploaded: Whether the file is uploaded or selected.
        progress_callback: Callable for updating progress.
    Returns:
        Analysis results, entity sentiments, and sentiment sentences.
    """
    if is_uploaded:
        lines = file_path_or_lines
    else:
        with open(file_path_or_lines, "r") as file:
            lines = file.readlines()

    if not lines or len(lines) < 7:
        raise ValueError("Invalid file format or insufficient lines in the file.")

    # Start analysis
    if progress_callback:
        progress_callback(1, message="Reading and preparing file...")

    article_text = "\n".join(line.strip() for line in lines[7:])
    if progress_callback:
        progress_callback(3, message="Analyzing text...")

    # Call `analyse_single_article` directly (blocking)
    return analyse_single_article(article_text, progress_callback)


async def analyze_file_reactive(file_input, file_select, article_analysis, entity_sentiments, sentiment_sentences,
                                progress_id):
    """
    Handles file analysis for uploaded or selected files reactively.

    Args:
        file_input: Reactive input for file upload.
        file_select: Reactive input for file selection.
        article_analysis: Reactive value to store article analysis results.
        entity_sentiments: Reactive value to store entity sentiments.
        sentiment_sentences: Reactive value to store sentiment sentences.
        progress_id: ID for the progress bar UI.
    """
    if file_input() is not None:
        # Handle uploaded file
        _, lines = handle_file_upload(file_input)
        if not lines:
            ui.notification_show("Invalid uploaded file.", type="error")
            return
        file_path_or_lines = lines
        is_uploaded = True
    elif file_select() and file_select() != "None":
        # Handle selected file
        folder_path = here / "BRAT_Data"
        file_choices = list_files_in_folder(folder_path)
        selected_file = next(full for full, display in file_choices if display == file_select())
        file_path_or_lines = folder_path / selected_file
        is_uploaded = False
    else:
        # No file provided
        return

    with ui.Progress(min=1, max=100) as p:
        try:
            # Set progress at the beginning
            p.set(message="Starting analysis...")

            # Blocking function (ensure progress updates are in the main thread)
            analysis, entities, sentences = analyze_file(
                file_path_or_lines=file_path_or_lines,
                is_uploaded=is_uploaded,
                progress_callback=p.set  # Directly pass progress updates
            )
            # Store results
            article_analysis.set(analysis)
            entity_sentiments.set(entities)
            sentiment_sentences.set(sentences)
        except ValueError as e:
            ui.notification_show(str(e), type="error")


def render_uploaded_text_content(article_analysis, view_full_text, file_upload, file_select):
    if article_analysis.get():
        # If analysis is complete, display the results
        lines = article_analysis.get().split("<br>")
        if view_full_text.get():
            content_lines = lines
        else:
            content_lines = lines[:50]  # Show only the first 50 lines by default
        return ui.HTML("<br>".join(content_lines))
    elif file_upload() or file_select() != "None":
        # Show the progress bar while analysis is ongoing
        return ui.div(
            ui.div("Analysis in progress...", class_="progress-message"),
            class_="progress-container"
        )
    return "No file uploaded"


def generate_header(file_input, file_select, article_analysis, current_header):
    header, _ = handle_file_upload(file_input)
    if header:
        current_header.set(header)
        return header
    selected_file = file_select()
    if selected_file and selected_file != "None":
        folder_path = here / "BRAT_Data"
        file_choices = list_files_in_folder(folder_path)
        selected_file_full_path = next(full for full, display in file_choices if display == selected_file)
        file_path = folder_path / selected_file_full_path
        with open(file_path, "r") as file:
            lines = file.readlines()
            if len(lines) >= 7:
                display_header = (f"{lines[1].split(': ', 1)[1].strip()} -- "
                                  f"{lines[0].split(': ', 1)[1].strip()} -- "
                                  f"{lines[2].split(';')[0].split(': ', 1)[1].strip()} / "
                                  f"{lines[2].split(';')[1].split(': ', 1)[1].strip()}")
                current_header.set(display_header)
                return display_header
    if not article_analysis():
        return "No file uploaded"
    return current_header()


def remove_png_files():
    png_files = glob.glob('App/www/*.png')
    for file in png_files:
        if 'logomini.png' not in file:
            os.remove(file)


single_module = ui.tags.div(
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

double_module = ui.tags.div(
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

all_module = ui.tags.div(
    ui.tags.div(
        ui.output_ui("all_mode_plots"),
        class_="main-left-container"
    ),
    ui.output_ui("right_container_all"),
    ui.busy_indicators.options(spinner_type="dots2", spinner_color=my_orange, spinner_size="50px"),
    class_="main-container"
)

about_module = ui.tags.div(
    ui.tags.div(
        collapsible_section(
            "Introduction",
            "toggle_introduction_button",
            "introduction_content"
        ),
        collapsible_section(
            "Guide",
            "toggle_guide_button",
            "guide_content"
        ),
        collapsible_section(
            "Authors",
            "toggle_authors_button",
            "authors_content"
        ),
        class_="main-left-container"
    ),
    class_="main-container about-module"
)

page_dependencies = ui.head_content(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
    ui.tags.link(rel="icon", type="image/png", href="logomini.png")
)

page_layout = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel("Single", single_module),
    ui.nav_panel("Double", double_module),
    ui.nav_panel("All", all_module),
    ui.nav_panel("About", about_module),
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


def server(input, output, session):
    view_full_text = reactive.Value(False)
    view_full_text_1 = reactive.Value(False)
    view_full_text_2 = reactive.Value(False)
    right_container_visible_single = reactive.Value(True)
    right_container_visible_double = reactive.Value(True)
    right_container_visible_all = reactive.Value(True)
    eda_visible = reactive.Value(True)
    ner_visible = reactive.Value(True)
    sentiment_visible = reactive.Value(True)
    communities_visible = reactive.Value(True)
    keywords_trends_visible = reactive.Value(True)
    ngrams_visible = reactive.Value(True)
    introduction_visible = reactive.Value(True)
    guide_visible = reactive.Value(True)
    authors_visible = reactive.Value(True)
    article_analysis = reactive.Value(None)
    entity_sentiments = reactive.Value(None)
    sentiment_sentences = reactive.Value(None)
    article_analysis_1 = reactive.Value(None)
    article_analysis_2 = reactive.Value(None)
    entity_sentiments_1 = reactive.Value(None)
    sentiment_sentences_1 = reactive.Value(None)
    entity_sentiments_2 = reactive.Value(None)
    sentiment_sentences_2 = reactive.Value(None)
    selected_file_value = reactive.Value("None")
    selected_file_value_1 = reactive.Value("None")
    selected_file_value_2 = reactive.Value("None")
    article_header = reactive.Value("No file uploaded")
    article_header_1 = reactive.Value("No file uploaded")
    article_header_2 = reactive.Value("No file uploaded")
    keywords_choices = reactive.Value(generate_keywords("Gaza before conflict"))

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

    # @reactive.Effect
    # @reactive.event(input.file_upload)
    # async def analyze_uploaded_file():
    #     _, lines = handle_file_upload(input.file_upload)
    #     if lines:
    #         with ui.Progress(min=1, max=100, id="analysis_progress") as p:
    #             try:
    #                 analysis, entities, sentences = await analyze_file(
    #                     file_path_or_lines=lines,
    #                     is_uploaded=True,
    #                     progress_callback=p.set
    #                 )
    #                 article_analysis.set(analysis)
    #                 entity_sentiments.set(entities)
    #                 sentiment_sentences.set(sentences)
    #             except ValueError as e:
    #                 ui.notification_show(str(e), type="error")

    # @reactive.Effect
    # @reactive.event(input.file_upload_1)
    # def analyze_article_1():
    #     _, lines = handle_file_upload(input.file_upload_1)
    #     if lines:
    #         article_text = "\n".join(line.strip() for line in lines[7:])
    #         analysis, entities, sentences = analyse_single_article(article_text)
    #         article_analysis_1.set(analysis)
    #         entity_sentiments_1.set(entities)
    #         sentiment_sentences_1.set(sentences)
    #
    # @reactive.Effect
    # @reactive.event(input.file_upload_2)
    # def analyze_article_2():
    #     _, lines = handle_file_upload(input.file_upload_2)
    #     if lines:
    #         article_text = "\n".join(line.strip() for line in lines[7:])
    #         analysis, entities, sentences = analyse_single_article(article_text)
    #         article_analysis_2.set(analysis)
    #         entity_sentiments_2.set(entities)
    #         sentiment_sentences_2.set(sentences)

    @output
    @render.text
    def uploaded_text_header():
        return generate_header(input.file_upload, input.file_select, article_analysis, article_header)

    # @output
    # @render.ui
    # def uploaded_text_content():
    #     if article_analysis.get():
    #         lines = article_analysis.get().split("<br>")
    #         if view_full_text.get():
    #             content_lines = lines
    #         else:
    #             content_lines = lines[:50]
    #         return ui.HTML("<br>".join(content_lines))
    #     return "No file uploaded"

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
    @render.text
    def uploaded_text_header_1():
        return generate_header(input.file_upload_1, input.file_select_1, article_analysis_1, article_header_1)

    # @output
    # @render.ui
    # def uploaded_text_content_1():
    #     if article_analysis_1.get():
    #         lines = article_analysis_1.get().split("<br>")
    #         if view_full_text_1.get():
    #             content_lines = lines
    #         else:
    #             content_lines = lines[:50]
    #         return ui.HTML("<br>".join(content_lines))
    #     return "No file uploaded"

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
    @render.text
    def uploaded_text_header_2():
        return generate_header(input.file_upload_2, input.file_select_2, article_analysis_2, article_header_2)

    # @output
    # @render.ui
    # def uploaded_text_content_2():
    #     if article_analysis_2.get():
    #         lines = article_analysis_2.get().split("<br>")
    #         if view_full_text_2.get():
    #             content_lines = lines
    #         else:
    #             content_lines = lines[:50]
    #         return ui.HTML("<br>".join(content_lines))
    #     return "No file uploaded"

    @output
    @render.ui
    def uploaded_text_content_2():
        return render_uploaded_text_content(
            article_analysis=article_analysis_2,
            view_full_text=view_full_text_2,
            file_upload=input.file_upload_2,
            file_select=input.file_select_2
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
    @render.ui
    def double_mode_plots():
        if (input.file_upload_1() or input.file_select_1() != "None") and (
                input.file_upload_2() or input.file_select_2() != "None"):
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
    def entity_types_plot():
        dataset_name = input.dataset_filter()
        plot = generate_entity_types_plot(dataset_name)
        return plot

    @output
    @render_widget
    def most_common_entities_plot():
        dataset_name = input.dataset_filter()
        entity_type = input.entity_type_filter()
        plot = generate_most_common_entities_plot(dataset_name, entity_type)
        return plot

    @output
    @render_widget
    def sentiment_dist_plot():
        dataset_name = input.dataset_filter()
        plot = generate_sentiment_dist_plot(dataset_name)
        return plot

    @output
    @render_widget
    def sentiment_over_time_plot():
        dataset_name = input.dataset_filter()
        model_name = input.sentiment_model_filter().lower()
        plot = generate_sentiment_over_time_plot(dataset_name, model_name)
        return plot

    @output
    @render.plot
    def sentiment_word_cloud_plot():
        dataset_name = input.dataset_filter()
        model_name = input.sentiment_model_filter().lower()
        sentiment = input.sentiment_filter()
        plot = generate_sentiment_word_cloud_plot(dataset_name, model_name, sentiment)
        return plot

    @output
    @render_widget
    def sentiment_dist_per_target_plot():
        dataset_name = input.dataset_filter()
        plot = generate_sentiment_dist_per_target_plot(dataset_name)
        return plot

    @output
    @render_widget
    def sentiment_over_time_per_target_plot():
        dataset_name = input.dataset_filter()
        plot = generate_sentiment_over_time_per_target_plot(dataset_name)
        return plot

    @output
    @render_widget
    def sentiment_dist_over_time_by_target_plot():
        dataset_name = input.dataset_filter()
        sentiment = input.sentiment_filter()
        plot = generate_sentiment_dist_over_time_by_target_plot(dataset_name, sentiment)
        return plot

    @output
    @render_widget
    def word_count_distribution_plot():
        dataset_name = input.dataset_filter()
        plot = generate_word_count_distribution_plot(dataset_name)
        return plot

    @output
    @render_widget
    def sentence_count_distribution_plot():
        dataset_name = input.dataset_filter()
        plot = generate_sentence_count_distribution_plot(dataset_name)
        return plot

    @output
    @render.plot
    def top_N_common_words_plot():
        dataset_name = input.dataset_filter()
        N = input.word_cloud_n()
        plot = generate_top_N_common_words_plot(dataset_name, N)
        return plot

    @output
    @render_widget
    def top_N_common_pos_plot():
        dataset_name = input.dataset_filter()
        N = input.word_cloud_n()
        plot = generate_top_N_common_pos_plot(dataset_name, N)
        return plot

    @output
    @render.plot
    def pos_wordclouds_plot():
        dataset_name = input.dataset_filter()
        N = input.word_cloud_n()
        pos = input.pos_filter()
        plot = generate_pos_wordclouds_plot(dataset_name, N, pos)
        return plot

    @output
    @render.image
    def community_graph():
        dataset_name = input.dataset_filter()
        image_path = generate_community_graph(dataset_name)
        return {"src": image_path, "alt": "Community Graph", "width": "100%"}

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
    @render_widget
    def keywords_over_time_plot():
        dataset_name = input.dataset_filter()
        return generate_keywords_over_time_plot(dataset_name)

    @output
    @render_widget
    def stacked_keywords_over_time_plot():
        dataset_name = input.dataset_filter()
        selected_keywords_widget = input.selected_keywords()
        selected_date_agg = input.date_range()
        return generate_stacked_keywords_over_time_plot(dataset_name, selected_keywords_widget, selected_date_agg)

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

    @output
    @render.image
    def bigrams_plot():
        dataset_name = input.dataset_filter()
        image_path = generate_bigrams_plot(dataset_name)
        return {"src": image_path, "alt": "Bigrams Plot", "width": "100%"}

    @output
    @render.ui
    def all_mode_plots():
        return ui.div(
            collapsible_section(
                "Exploratory Data Analysis",
                "toggle_eda_button",
                "eda_plots"
            ),
            collapsible_section(
                "Named Entity Recognition",
                "toggle_ner_button",
                "ner_plots"
            ),
            collapsible_section(
                "Sentiment",
                "toggle_sentiment_button",
                "sentiment_plots"
            ),
            collapsible_section(
                "Community Graphs",
                "toggle_community_button",
                "community_plots"
            ),
            collapsible_section(
                "N-grams",
                "toggle_ngrams_button",
                "ngrams_plots"
            ),
            collapsible_section(
                "Keywords trends",
                "toggle_keywords_trends_button",
                "keywords_trends_plots"
            ),
            class_="plots-container"
        )

    @reactive.Effect
    @reactive.event(input.hide_container_button_single)
    def toggle_container_visibility_single():
        right_container_visible_single.set(not right_container_visible_single.get())

    @reactive.Effect
    @reactive.event(input.hide_container_button_double)
    def toggle_container_visibility_double():
        right_container_visible_double.set(not right_container_visible_double.get())

    @reactive.Effect
    @reactive.event(input.file_upload, input.file_select)
    def auto_hide_container_single():
        if input.file_upload() or (input.file_select() != "None"):
            right_container_visible_single.set(False)

    @reactive.Effect
    @reactive.event(input.file_upload_1, input.file_upload_2)
    def auto_hide_container_double():
        if input.file_upload_1() and input.file_upload_2():
            right_container_visible_double.set(False)

    @output
    @render.ui
    def right_container_single():
        folder_path = here / "BRAT_Data"  # Adjust the folder path as needed
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
    def right_container_double():
        folder_path = here / "BRAT_Data"  # Adjust the folder path as needed
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

    @output
    @render.ui
    def right_container_all():
        if right_container_visible_all.get():
            return ui.div(
                ui.input_select("dataset_filter", "Select Dataset", choices=[
                    "Gaza before conflict", "Gaza during conflict",
                    "Ukraine before conflict", "Ukraine during conflict"
                ]),
                ui.input_select("sentiment_filter", "Select Sentiment", choices=["Positive", "Neutral", "Negative"]),
                ui.input_select("entity_type_filter", "Select Entity Type",
                                choices=["Person", "Organisation", "Location", "Miscellaneous"]),
                ui.input_select("sentiment_model_filter", "Select Sentiment Model", choices=["TSC", "VADER"]),
                ui.input_numeric("word_cloud_n", "Number of Words in Word Cloud", value=100, min=1),
                ui.input_selectize("pos_filter", "Select Part of Speech", choices=generate_pos_choices(),
                                   multiple=False, selected="Common Singular Nouns",
                                   options={"create": False, "searchField": ["label"]}),
                ui.input_text("filter_words", "Filter Words (comma-separated)", value="US, China"),
                ui.input_numeric("ngram_number", "N-gram Number", value=2, min=2),
                ui.input_selectize(
                    "selected_keywords",
                    "Select keywords to analyze",
                    choices=keywords_choices.get(),
                    selected=keywords_choices.get()[0],
                    multiple=True,
                    options={"create": False, "placeholder": "Select or type keywords"}
                ),

                ui.input_select(
                    "date_range",
                    "Select date aggregation",
                    choices=['daily', 'weekly', 'monthly'],
                    selected='monthly'
                ),
                ui.input_action_button("hide_container_button_all", "Hide Menu", class_="btn btn-secondary"),
                class_="main-right-container",
                id="main-right-container-all"
            )
        else:
            return ui.div(
                ui.input_action_button("hide_container_button_all", "Show Menu", class_="show-container-tab"),
                class_="main-right-container hidden",
                id="main-right-container-all"
            )

    @reactive.Effect
    @reactive.event(input.hide_container_button_all)
    def toggle_container_visibility_all():
        right_container_visible_all.set(not right_container_visible_all.get())

    @reactive.Effect
    @reactive.event(input.toggle_ner_button)
    def toggle_ner_visibility():
        ner_visible.set(not ner_visible.get())
        session.send_input_message("toggle_ner_button", {"label": "⯆" if ner_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_sentiment_button)
    def toggle_sentiment_visibility():
        sentiment_visible.set(not sentiment_visible.get())
        session.send_input_message("toggle_sentiment_button", {"label": "⯆" if sentiment_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_eda_button)
    def toggle_eda_visibility():
        eda_visible.set(not eda_visible.get())
        session.send_input_message("toggle_eda_button", {"label": "⯆" if eda_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_community_button)
    def toggle_community_visibility():
        communities_visible.set(not communities_visible.get())
        session.send_input_message("toggle_community_button", {"label": "⯆" if communities_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_ngrams_button)
    def toggle_ngrams_visibility():
        ngrams_visible.set(not ngrams_visible.get())
        session.send_input_message("toggle_ngrams_button", {"label": "⯆" if ngrams_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_keywords_trends_button)
    def toggle_keywords_trends_visibility():
        keywords_trends_visible.set(not keywords_trends_visible.get())
        session.send_input_message("toggle_keywords_trends_button",
                                   {"label": "⯆" if keywords_trends_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_introduction_button)
    def toggle_introduction_visibility():
        introduction_visible.set(not introduction_visible.get())
        session.send_input_message("toggle_introduction_button", {"label": "⯆" if introduction_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_guide_button)
    def toggle_guide_visibility():
        guide_visible.set(not guide_visible.get())
        session.send_input_message("toggle_guide_button", {"label": "⯆" if guide_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_authors_button)
    def toggle_authors_visibility():
        authors_visible.set(not authors_visible.get())
        session.send_input_message("toggle_authors_button", {"label": "⯆" if authors_visible.get() else "⯈"})

    @output
    @render.ui
    def eda_plots():
        if eda_visible.get():
            return ui.div(
                # ui.img(src='plot1.png', class_="plot-image eda-plot"),
                output_widget("word_count_distribution_plot"),
                output_widget("sentence_count_distribution_plot"),
                ui.output_plot("top_N_common_words_plot"),
                output_widget("top_N_common_pos_plot"),
                ui.output_plot("pos_wordclouds_plot"),
                class_="plots-row"
            )
        return ui.div()

    @output
    @render.ui
    def ner_plots():
        if ner_visible.get():
            return ui.div(
                output_widget("entity_types_plot"),
                output_widget("most_common_entities_plot"),
                class_="plots-row"
            )
        return ui.div()  # Return empty div if hidden

    @output
    @render.ui
    def sentiment_plots():
        if sentiment_visible.get():
            # dataset_name = input.dataset_filter()
            # sentiment = input.sentiment_filter().lower()
            # sentiment_over_time_by_target = f'Sentiment/{sentiment}_sentiment_over_time_by_target_{dataset_name}.png'
            # return ui.div(
            #     ui.img(src=sentiment_over_time_by_target, class_="plot-image sentiment-plot"),
            #     class_="plots-row"
            # )
            return ui.div(
                output_widget("sentiment_dist_plot"),
                output_widget("sentiment_over_time_plot"),
                ui.output_plot("sentiment_word_cloud_plot"),
                output_widget("sentiment_dist_per_target_plot"),
                output_widget("sentiment_over_time_per_target_plot"),
                output_widget("sentiment_dist_over_time_by_target_plot"),
                class_="plots-row"
            )
        return ui.div()

    @output
    @render.ui
    def community_plots():
        if communities_visible.get():
            return ui.div(
                ui.output_image("community_graph"),
                # output_widget("community_graph"),
                # ui.img(src='plot1.png', class_="plot-image eda-plot"),
                class_="plots-row"
            )
        return ui.div()

    @output
    @render.ui
    def show_view_more_button():
        if article_analysis.get():
            return ui.input_action_button("view_full_text", "View more", class_="btn btn-secondary view-button")
        return ui.div()

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
    def ngrams_plots():
        if ngrams_visible.get():
            return ui.div(
                ui.output_image("bigrams_plot"),
                ui.output_ui("concordance_table", class_="concordance-table"),
                class_="plots-row"
            )
        return ui.div()

    @output
    @render.ui
    def keywords_trends_plots():
        if keywords_trends_visible.get():
            return ui.div(
                output_widget("keywords_over_time_plot"),
                output_widget("stacked_keywords_over_time_plot"),
                class_="plots-row"
            )
        return ui.div()

    @output
    @render.ui
    def concordance_table():
        dataset_name = input.dataset_filter()
        filter_words = [word.strip() for word in input.filter_words().split(",") if word.strip()]
        ngram_number = input.ngram_number()

        # Generate concordance DataFrame
        concordance_df = generate_concordance(dataset_name, filter_words, ngram_number)

        # Ensure only the required columns are used
        concordance_df = concordance_df[["lefts", "center", "rights", "count"]]

        if concordance_df.empty:
            return ui.div("No concordance results found.")

        table_html = (
            f'<h3 style="text-align: center;">Concordance Results with {ngram_number}-Grams for {input.filter_words()} - {dataset_name}</h3>'
            '<table id="concordance-table" class="concordance-table">'
            '<tbody>'
            '<tr>'
            '<th>Index</th>'
            '<th>Lefts</th>'
            '<th>Center</th>'
            '<th>Rights</th>'
            '<th>Count</th>'
            '</tr>'
        )

        for idx, row in concordance_df.iterrows():
            table_html += (
                f'<tr>'
                f'<td>{idx + 1}</td>'
                f'<td class="column-lefts">{row["lefts"]}</td>'
                f'<td class="column-center">{row["center"]}</td>'
                f'<td class="column-rights">{row["rights"]}</td>'
                f'<td class="column-count">{row["count"]}</td>'
                f'</tr>'
            )

        table_html += '</tbody></table>'

        # Return the styled HTML table
        return ui.HTML(table_html)

    # @reactive.Effect
    # @reactive.event(input.file_select)
    # def analyze_selected_file():
    #     selected_display_file = input.file_select()
    #     if selected_display_file and selected_display_file != "None":
    #         selected_file_value.set(selected_display_file)
    #         folder_path = here / "BRAT_Data"
    #         file_choices = list_files_in_folder(folder_path)
    #         selected_file = next(full for full, display in file_choices if display == selected_display_file)
    #         file_path = folder_path / selected_file
    #         analyze_file(file_path, article_analysis, entity_sentiments, sentiment_sentences)

    # @reactive.Effect
    # @reactive.event(input.file_select_1)
    # def analyze_selected_file_1():
    #     selected_display_file_1 = input.file_select_1()
    #     if selected_display_file_1 and selected_display_file_1 != "None":
    #         selected_file_value_1.set(selected_display_file_1)
    #         folder_path = here / "BRAT_Data"
    #         file_choices = list_files_in_folder(folder_path)
    #         selected_file_1 = next(full for full, display in file_choices if display == selected_display_file_1)
    #         file_path_1 = folder_path / selected_file_1
    #         analyze_file(file_path_1, article_analysis_1, entity_sentiments_1, sentiment_sentences_1)
    #
    # @reactive.Effect
    # @reactive.event(input.file_select_2)
    # def analyze_selected_file_2():
    #     selected_display_file_2 = input.file_select_2()
    #     if selected_display_file_2 and selected_display_file_2 != "None":
    #         selected_file_value_2.set(selected_display_file_2)
    #         folder_path = here / "BRAT_Data"
    #         file_choices = list_files_in_folder(folder_path)
    #         selected_file_2 = next(full for full, display in file_choices if display == selected_display_file_2)
    #         file_path_2 = folder_path / selected_file_2
    #         analyze_file(file_path_2, article_analysis_2, entity_sentiments_2, sentiment_sentences_2)

    @output
    @render.ui
    def introduction_content():
        if introduction_visible.get():
            return ui.HTML("""
            <div class="collapsible-section-content">
                <h3>This application provides an analysis of articles from the Global Times. It includes various modules:</h3>
                <br>
                <ul>
                    <li><strong>Single:</strong> Analyze a single article.</li>
                    <li><strong>Double:</strong> Compare two articles.</li>
                    <li><strong>All:</strong> Perform exploratory data analysis on a dataset of articles.</li>
                </ul>
            </div>
        """)
        return ui.div()

    @output
    @render.ui
    def guide_content():
        if guide_visible.get():
            return ui.HTML("""
                <div class="collapsible-section-content">
                    <h3>Instructions on how to use the application:</h3><br>
                    <ul>
                        <li><strong style="color: #333;">Select a file:</strong> Use the dropdown to select a file from the list.</li>
                        <li><strong style="color: #333;">Upload a file:</strong> Click the "UPLOAD ARTICLE" button to upload a new file.</li>
                        <li><strong style="color: #333;">View more/less:</strong> Click the "View more" button to expand the text, and "View less" to collapse it.</li>
                        <li><strong style="color: #333;">Hide/Show Menu:</strong> Click the "Hide Menu" button to hide the right container, and "Show Menu" to display it again.</li>
                        <li><strong style="color: #333;">Select Dataset:</strong> Use the dropdown to select a dataset for analysis.</li>
                        <li><strong style="color: #333;">Select Sentiment:</strong> Use the dropdown to filter by sentiment (Positive, Neutral, Negative).</li>
                        <li><strong style="color: #333;">Select Entity Type:</strong> Use the dropdown to filter by entity type (Person, Organisation, Location, Miscellaneous).</li>
                        <li><strong style="color: #333;">Select Sentiment Model:</strong> Use the dropdown to select a sentiment model (TSC, VADER).</li>
                        <li><strong style="color: #333;">Number of Words in Word Cloud:</strong> Use the numeric input to specify the number of words in the word cloud.</li>
                        <li><strong style="color: #333;">Select Part of Speech:</strong> Use the dropdown to filter by part of speech.</li>
                        <li><strong style="color: #333;">Filter Words:</strong> Use the text input to filter words (comma-separated).</li>
                        <li><strong style="color: #333;">N-gram Number:</strong> Use the numeric input to specify the n-gram number.</li>
                    </ul>
                </div>
            """)
        return ui.div()

    @output
    @render.ui
    def authors_content():
        if authors_visible.get():
            return ui.HTML("""
                <div class="collapsible-section-content">
                    <p><strong style="color: #333;">Authors:</strong> Łukasz Grabarski & Marta Szuwarska (Warsaw University of Technology)</p>
                    <p><strong style="color: #333;">Supervisor:</strong> Dr. Anna Wróblewska (Warsaw University of Technology)</p>
                    <p><strong style="color: #333;">Co-supervisor:</strong> Prof. Agnieszka Kaliska (Adam Mickiewicz University in Poznań)</p>
                    <p><strong style="color: #333;">Co-operations:</strong> Prof. Anna Rudakowska (Tamkang University in Taiwan), Dr. Daniel Dan (Modul University in Vienna)</p>
                </div>
            """)
        return ui.div()

    @reactive.Effect
    @reactive.event(input.file_upload, input.file_select)
    async def analyze_single_file():
        await analyze_file_reactive(
            file_input=input.file_upload,
            file_select=input.file_select,
            article_analysis=article_analysis,
            entity_sentiments=entity_sentiments,
            sentiment_sentences=sentiment_sentences,
            progress_id="analysis_progress_single"
        )

    @reactive.Effect
    @reactive.event(input.file_upload_1, input.file_select_1)
    async def analyze_first_file():
        await analyze_file_reactive(
            file_input=input.file_upload_1,
            file_select=input.file_select_1,
            article_analysis=article_analysis_1,
            entity_sentiments=entity_sentiments_1,
            sentiment_sentences=sentiment_sentences_1,
            progress_id="analysis_progress_double_1"
        )

    @reactive.Effect
    @reactive.event(input.file_upload_2, input.file_select_2)
    async def analyze_second_file():
        await analyze_file_reactive(
            file_input=input.file_upload_2,
            file_select=input.file_select_2,
            article_analysis=article_analysis_2,
            entity_sentiments=entity_sentiments_2,
            sentiment_sentences=sentiment_sentences_2,
            progress_id="analysis_progress_double_2"
        )

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def on_dataset_change():
        remove_png_files()

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def update_keywords_choices():
        dataset_name = input.dataset_filter()
        keywords_choices.set(generate_keywords(dataset_name))


www_dir = Path(__file__).parent / "App/www"
app = App(app_ui, server, static_assets=www_dir)
