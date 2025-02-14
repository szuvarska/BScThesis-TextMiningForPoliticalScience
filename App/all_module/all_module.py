from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
from App.all_module.plots import generate_pos_choices, generate_keywords, generate_concordance, generate_target_choices
from App.all_module.render_plots import setup_plot_outputs
from App.all_module.render_ui import setup_ui_outputs
from App.all_module.dataset_analysis import analyze_dataset_reactive
from App.utils import collapsible_section, remove_png_files
from colors import my_orange
import asyncio

all_module_ui = ui.tags.div(
    ui.tags.div(
        ui.output_ui("all_mode_plots"),
        class_="main-left-container"
    ),
    ui.output_ui("right_container_all"),
    ui.busy_indicators.options(spinner_type="dots2", spinner_color=my_orange, spinner_size="50px"),
    class_="main-container"
)


def all_module_server(input, output, session):
    right_container_visible_all = reactive.Value(True)
    eda_visible = reactive.Value(True)
    ner_visible = reactive.Value(True)
    sentiment_visible = reactive.Value(True)
    communities_visible = reactive.Value(True)
    keywords_trends_visible = reactive.Value(True)
    ngrams_visible = reactive.Value(True)
    keywords = generate_keywords("Gaza before conflict")
    keywords_choices = reactive.Value(keywords)
    dataset_choices = reactive.Value(
        ["Gaza before conflict", "Gaza during conflict", "Ukraine before conflict", "Ukraine during conflict"])
    dataset_filter_value = reactive.Value("Gaza before conflict")
    sentiment_filter_value = reactive.Value("Positive")
    entity_type_filter_value = reactive.Value("Person")
    sentiment_model_filter_value = reactive.Value("TSC")
    sentiment_targets = generate_target_choices("Gaza before conflict")
    sentiment_targets_choices = reactive.Value(sentiment_targets)
    sentiment_target_filter_value = reactive.Value("China")
    word_cloud_n_value = reactive.Value(100)
    pos_filter_value = reactive.Value("Common Singular Nouns")
    filter_words_value = reactive.Value("US, China")
    ngram_number_value = reactive.Value(3)
    selected_keywords_value = reactive.Value(keywords[:2])
    date_range_value = reactive.Value("daily")

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def update_dataset_filter():
        dataset_filter_value.set(input.dataset_filter())

    @reactive.Effect
    @reactive.event(input.sentiment_filter)
    def update_sentiment_filter():
        sentiment_filter_value.set(input.sentiment_filter())

    @reactive.Effect
    @reactive.event(input.entity_type_filter)
    def update_entity_type_filter():
        entity_type_filter_value.set(input.entity_type_filter())

    @reactive.Effect
    @reactive.event(input.sentiment_model_filter)
    def update_sentiment_model_filter():
        sentiment_model_filter_value.set(input.sentiment_model_filter())

    @reactive.Effect
    @reactive.event(input.word_cloud_n)
    def update_word_cloud_n():
        word_cloud_n_value.set(input.word_cloud_n())

    @reactive.Effect
    @reactive.event(input.pos_filter)
    def update_pos_filter():
        pos_filter_value.set(input.pos_filter())

    @reactive.Effect
    @reactive.event(input.filter_words)
    def update_filter_words():
        filter_words_value.set(input.filter_words())

    @reactive.Effect
    @reactive.event(input.ngram_number)
    def update_ngram_number():
        ngram_number_value.set(input.ngram_number())

    @reactive.Effect
    @reactive.event(input.selected_keywords)
    def update_selected_keywords():
        selected_keywords_value.set(list(input.selected_keywords()))

    @reactive.Effect
    @reactive.event(input.date_range)
    def update_date_range():
        date_range_value.set(input.date_range())

    @reactive.Effect
    @reactive.event(input.sentiment_target_filter)
    def update_sentiment_target_filter():
        sentiment_target_filter_value.set(input.sentiment_target_filter())

    @output
    @render.ui
    def right_container_all():
        if right_container_visible_all.get():
            return ui.div(
                ui.input_select(
                    "dataset_filter",
                    "Select dataset",
                    choices=dataset_choices.get(),
                    selected='Gaza before conflict'
                ),
                ui.input_file(
                    "upload_folder",
                    "Upload dataset",
                    multiple=True,
                    accept=[".txt", ".zip"]
                ),
                ui.input_text(
                    "dataset_name",
                    "",
                    placeholder="Enter dataset name",
                ),
                ui.input_task_button("analyze_dataset_button", "Analyse Dataset", label_busy='Analysing...',
                                     class_="btn btn-file"),
                ui.input_numeric(
                    "word_cloud_n",
                    "Enter number of most common words in word clouds",
                    value=100,
                    min=1
                ),
                ui.input_selectize(
                    "pos_filter",
                    "Select part of speech for word cloud",
                    choices=generate_pos_choices(),
                    multiple=False,
                    selected='Common Singular Nouns'
                ),
                ui.input_select(
                    "entity_type_filter",
                    "Select entity type",
                    choices=["Person", "Organisation", "Location", "Miscellaneous"],
                    selected='Person'
                ),
                ui.input_select(
                    "sentiment_model_filter",
                    "Select sentiment model",
                    choices=["TSC", "VADER"],
                    selected='TSC'
                ),
                ui.input_select(
                    "sentiment_filter",
                    "Select sentiment type for word cloud and sentiment by target",
                    choices=["Positive", "Neutral", "Negative"],
                    selected='Positive'
                ),
                ui.input_select(
                    "target_filter",
                    "Select target for sentiment over time",
                    choices=sentiment_targets,
                    selected='China'
                ),
                ui.input_text(
                    "filter_words",
                    "Enter words for concordance (comma-separated)",
                    value="US, China"
                ),
                ui.input_numeric(
                    "ngram_number",
                    "Enter N-gram number",
                    value=3,
                    min=2
                ),
                ui.input_selectize(
                    "selected_keywords",
                    "Select keywords for trends",
                    choices=keywords,
                    selected=keywords[:2],
                    multiple=True
                ),
                ui.input_select(
                    "date_range",
                    "Select date aggregation for keywords trends",
                    choices=['daily', 'weekly', 'monthly'],
                    selected='daily'
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

    setup_plot_outputs(input, output, session)
    setup_ui_outputs(input, output, session)

    @reactive.Effect
    @reactive.event(input.hide_container_button_all)
    def toggle_container_visibility_all():
        right_container_visible_all.set(not right_container_visible_all.get())
        ui.update_select("dataset_filter", selected=dataset_filter_value.get())
        ui.update_text("filter_words", value=filter_words_value.get())
        ui.update_select("pos_filter", selected=pos_filter_value.get())
        ui.update_select("entity_type_filter", selected=entity_type_filter_value.get())
        ui.update_select("sentiment_model_filter", selected=sentiment_model_filter_value.get())
        ui.update_select("sentiment_filter", selected=sentiment_filter_value.get())
        ui.update_select("target_filter", choices=sentiment_targets_choices.get(),
                         selected=sentiment_target_filter_value.get())
        ui.update_numeric("word_cloud_n", value=word_cloud_n_value.get())
        ui.update_numeric("ngram_number", value=ngram_number_value.get())
        ui.update_selectize("selected_keywords", choices=keywords_choices.get(), selected=selected_keywords_value.get())
        ui.update_select("date_range", selected=date_range_value.get())

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
                class_="plots-row"
            )
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

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def on_dataset_change():
        remove_png_files()

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def update_keywords_choices():
        dataset_name = input.dataset_filter()
        keywords_list = generate_keywords(dataset_name)
        keywords_choices.set(keywords_list)
        selected_keywords_value.set(keywords_list[:2])
        ui.update_selectize("selected_keywords", choices=keywords_list, selected=keywords_list[:2])

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def update_sentiment_targets_choices():
        dataset_name = input.dataset_filter()
        sentiment_targets_list = generate_target_choices(dataset_name)
        sentiment_targets_choices.set(sentiment_targets_list)
        ui.update_select("target_filter", choices=sentiment_targets_list)

        if 'China' in sentiment_targets_list:
            sentiment_target_filter_value.set("China")
            ui.update_select("target_filter", selected="China")
        else:
            sentiment_target_filter_value.set(sentiment_targets_list[0])
            ui.update_select("target_filter", selected=sentiment_targets_list[0])

    @reactive.Effect
    @reactive.event(input.analyze_dataset_button)
    async def analyze_dataset():
        await analyze_dataset_reactive(
            files=input.upload_folder(),
            dataset_choices=dataset_choices,
            dataset_filter_value=dataset_filter_value,
            dataset_name=input.dataset_name(),
        )
