import time
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
from shiny import App, render, ui, reactive
from htmltools import tags, Tag
import numpy as np
import matplotlib.pyplot as plt
from App.plots import generate_entity_types_plot, generate_most_common_entities_plot, generate_sentiment_dist_plot, \
    generate_sentiment_over_time_plot, generate_sentiment_word_cloud_plot, generate_sentiment_dist_per_target_plot, \
    generate_sentiment_over_time_per_target_plot, generate_sentiment_dist_over_time_by_target_plot, \
    generate_word_count_distribution_plot, generate_sentance_count_distribution_plot, generate_top_N_common_words_plot, \
    generate_top_N_common_pos_plot, generate_pos_wordclouds_plot, generate_community_graph
from shinywidgets import output_widget, render_widget

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
            style="display: flex; align-items: center; margin-bottom: 10px;"
        ),
        ui.output_ui(plot_id),
        ui.busy_indicators.options(spinner_type="bars3"),
        class_="collapsible-section"
    )


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
    ui.output_ui("right_container_single"),
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
    ui.output_ui("right_container_double"),
    class_="main-container"
)

all_module = ui.tags.div(
    ui.tags.div(
        ui.output_ui("all_mode_plots"),
        class_="main-left-container"
    ),
    ui.output_ui("right_container_all"),
    class_="main-container"
)

page_dependencies = ui.tags.head(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css")
)

page_layout = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel("SINGLE", single_module),
    ui.nav_panel("DOUBLE", double_module),
    ui.nav_panel("ALL", all_module),
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
    view_full_text_1 = reactive.Value(False)
    view_full_text_2 = reactive.Value(False)
    right_container_visible_single = reactive.Value(True)
    right_container_visible_double = reactive.Value(True)
    right_container_visible_all = reactive.Value(True)
    eda_visible = reactive.Value(True)
    ner_visible = reactive.Value(True)
    sentiment_visible = reactive.Value(True)
    communities_visible = reactive.Value(True)

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
    def sentance_count_distribution_plot():
        dataset_name = input.dataset_filter()
        plot = generate_sentance_count_distribution_plot(dataset_name)
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
    @render.plot
    def community_graph():
        dataset_name = input.dataset_filter()
        plot = generate_community_graph(dataset_name)
        return plot

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
    @reactive.event(input.file_upload)
    def auto_hide_container_single():
        right_container_visible_single.set(False)

    @reactive.Effect
    @reactive.event(input.file_upload_1, input.file_upload_2)
    def auto_hide_container_double():
        if input.file_upload_1() and input.file_upload_2():
            right_container_visible_double.set(False)

    @output
    @render.ui
    def right_container_single():
        if right_container_visible_single.get():
            return ui.div(
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
        if right_container_visible_double.get():
            return ui.div(
                ui.input_file("file_upload_1", "Upload the first article"),
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
                ui.input_select("sentiment_filter", "Select Sentiment", choices=["Positive", "Negative", "Neutral"]),
                ui.input_select("entity_type_filter", "Select Entity Type",
                                choices=["Person", "Organisation", "Location", "Miscellaneous"]),
                ui.input_select("sentiment_model_filter", "Select Sentiment Model", choices=["TSC", "VADER"]),
                ui.input_numeric("word_cloud_n", "Number of Words in Word Cloud", value=100, min=1),
                ui.input_selectize("pos_filter", "Select Part of Speech", choices=[
                    "Common Singular Nouns",
                    "Common Plural Nouns",
                    "Proper Singular Nouns",
                    "Proper Plural Nouns",
                    "Adjectives in Positive Form",
                    "Adjectives in Comparative Form",
                    "Adjectives in Superlative Form",
                    "Verbs in Base Form",
                    "Verbs in Past Tense",
                    "Verbs in Present Participle",
                    "Verbs in Past Participle",
                    "Verbs in Non-3rd Person Singular Present Form",
                    "Verbs in 3rd Person Singular Present Form",
                    "Adverbs in Positive Form",
                    "Adverbs in Comparative Form",
                    "Adverbs in Superlative Form",
                    "Wh-determiners",
                    "Wh-pronouns",
                    "Wh-adverbs",
                    "Prepositions",
                    "Conjunctions",
                    "Determiners",
                    "Existential There",
                    "Foreign Words",
                    "List Item Marker",
                    "Modal",
                    "Cardinal Numbers",
                    "Possessive Ending",
                    "Personal Pronouns",
                    "Possessive Pronouns",
                    "Particles",
                    "To",
                    "Interjection",
                    "Symbol"
                ], multiple=False, options={"create": False, "searchField": ["label"]}),
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

    @output
    @render.ui
    def eda_plots():
        if eda_visible.get():
            return ui.div(
                # ui.img(src='plot1.png', class_="plot-image eda-plot"),
                output_widget("word_count_distribution_plot"),
                output_widget("sentance_count_distribution_plot"),
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
                # ui.output_plot("community_graph"),
                ui.img(src='plot1.png', class_="plot-image eda-plot"),
                class_="plots-row"
            )
        return ui.div()


www_dir = Path(__file__).parent / "App/www"
app = App(app_ui, server, static_assets=www_dir)
