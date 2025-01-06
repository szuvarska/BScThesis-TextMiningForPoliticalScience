import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging messages
from pathlib import Path
import pandas as pd
import seaborn as sns
from shiny import App, render, ui, reactive
import numpy as np
import matplotlib.pyplot as plt
from App.plots import generate_entity_types_plot, generate_most_common_entities_plot, generate_sentiment_dist_plot, \
    generate_sentiment_over_time_plot, generate_sentiment_word_cloud_plot, generate_sentiment_dist_per_target_plot, \
    generate_sentiment_over_time_per_target_plot, generate_sentiment_dist_over_time_by_target_plot, \
    generate_word_count_distribution_plot, generate_sentence_count_distribution_plot, generate_top_N_common_words_plot, \
    generate_top_N_common_pos_plot, generate_pos_wordclouds_plot, generate_community_graph, generate_pos_choices, \
    generate_bigrams_plot, generate_concordance
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
    ui.busy_indicators.options(spinner_type="dots2", spinner_color=my_orange, spinner_size="50px"),
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
    ui.busy_indicators.options(spinner_type="dots2", spinner_color=my_orange, spinner_size="50px"),
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

page_dependencies = ui.head_content(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
    ui.tags.link(rel="icon", type="image/png", href="logomini.png")
)

page_layout = ui.page_navbar(
    ui.nav_spacer(),
    ui.nav_panel("Single", single_module),
    ui.nav_panel("Double", double_module),
    ui.nav_panel("All", all_module),
    title="Global Times: Articles Analysis",  #PRESS ARTICLES EXPLORATION
    footer=ui.tags.div(
        ui.tags.div("Łukasz Grabarski & Marta Szuwarska", class_="footer")
    )
)

app_ui = ui.page_fluid(
    page_dependencies,
    page_layout,
    title="Global Times: Articles Analysis",  #PRESS ARTICLES EXPLORATION
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
    ngrams_visible = reactive.Value(True)
    article_analysis = reactive.Value(None)
    entity_sentiments = reactive.Value(None)
    sentiment_sentences = reactive.Value(None)
    article_analysis_1 = reactive.Value(None)
    article_analysis_2 = reactive.Value(None)
    entity_sentiments_1 = reactive.Value(None)
    sentiment_sentences_1 = reactive.Value(None)
    entity_sentiments_2 = reactive.Value(None)
    sentiment_sentences_2 = reactive.Value(None)

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

    @reactive.Effect
    @reactive.event(input.file_upload)
    def analyze_article():
        _, lines = handle_file_upload(input.file_upload)
        if lines:
            article_text = "\n".join(line.strip() for line in lines[7:])
            analysis, entities, sentences = analyse_single_article(article_text)
            article_analysis.set(analysis)
            entity_sentiments.set(entities)
            sentiment_sentences.set(sentences)

    @reactive.Effect
    @reactive.event(input.file_upload_1)
    def analyze_article_1():
        _, lines = handle_file_upload(input.file_upload_1)
        if lines:
            article_text = "\n".join(line.strip() for line in lines[7:])
            analysis, entities, sentences = analyse_single_article(article_text)
            article_analysis_1.set(analysis)
            entity_sentiments_1.set(entities)
            sentiment_sentences_1.set(sentences)

    @reactive.Effect
    @reactive.event(input.file_upload_2)
    def analyze_article_2():
        _, lines = handle_file_upload(input.file_upload_2)
        if lines:
            article_text = "\n".join(line.strip() for line in lines[7:])
            analysis, entities, sentences = analyse_single_article(article_text)
            article_analysis_2.set(analysis)
            entity_sentiments_2.set(entities)
            sentiment_sentences_2.set(sentences)

    @output
    @render.text
    def uploaded_text_header():
        header, _ = handle_file_upload(input.file_upload)
        if header:
            return header
        return "No file uploaded"

    @output
    @render.ui
    def uploaded_text_content():
        if article_analysis.get():
            lines = article_analysis.get().split("<br>")
            if view_full_text.get():
                content_lines = lines
            else:
                content_lines = lines[:50]
            return ui.HTML("<br>".join(content_lines))
        return "No file uploaded"

    @output
    @render.text
    def uploaded_text_header_1():
        header, _ = handle_file_upload(input.file_upload_1)
        if header:
            return header
        return "No file uploaded"

    @output
    @render.ui
    def uploaded_text_content_1():
        if article_analysis_1.get():
            lines = article_analysis_1.get().split("<br>")
            if view_full_text_1.get():
                content_lines = lines
            else:
                content_lines = lines[:50]
            return ui.HTML("<br>".join(content_lines))
        return "No file uploaded"

    @output
    @render.text
    def uploaded_text_header_2():
        header, _ = handle_file_upload(input.file_upload_2)
        if header:
            return header
        return "No file uploaded"

    @output
    @render.ui
    def uploaded_text_content_2():
        if article_analysis_2.get():
            lines = article_analysis_2.get().split("<br>")
            if view_full_text_2.get():
                content_lines = lines
            else:
                content_lines = lines[:50]
            return ui.HTML("<br>".join(content_lines))
        return "No file uploaded"

    @output
    @render.ui
    def single_mode_plots():
        if input.file_upload():
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
        if input.file_upload_1() and input.file_upload_2():
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

    # @output
    # @render.ui
    # def concordance_table():
    #     dataset_name = input.dataset_filter()
    #     filter_words = [word.strip() for word in input.filter_words().split(",") if word.strip()]
    #     ngram_number = input.ngram_number()
    #     concordance_df = generate_concordance(dataset_name, filter_words, ngram_number)
    #     concordance_df = concordance_df[["formated_ngram", "count"]]
    #
    #     if concordance_df.empty:
    #         return ui.div("No concordance results found.")
    #
    #     formatted_text = "Concordance Results:\n\n"
    #     formatted_text += "\n".join(
    #         [f"{row['formated_ngram']} \t Count: {row['count']}" for _, row in concordance_df.iterrows()])
    #     return ui.HTML(f"<pre>{formatted_text}</pre>")

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


www_dir = Path(__file__).parent / "App/www"
app = App(app_ui, server, static_assets=www_dir)
