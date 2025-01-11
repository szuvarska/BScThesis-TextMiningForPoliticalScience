from shiny import reactive, render, ui
from shinywidgets import output_widget, render_widget
from App.all_module.plots import (generate_entity_types_plot, generate_most_common_entities_plot,
                                  generate_sentiment_dist_plot, generate_sentiment_over_time_plot,
                                  generate_sentiment_word_cloud_plot, generate_sentiment_dist_per_target_plot,
                                  generate_sentiment_over_time_per_target_plot,
                                  generate_sentiment_dist_over_time_by_target_plot,
                                  generate_word_count_distribution_plot, generate_sentence_count_distribution_plot,
                                  generate_top_N_common_words_plot, generate_top_N_common_pos_plot,
                                  generate_pos_wordclouds_plot, generate_community_graph, generate_pos_choices,
                                  generate_bigrams_plot, generate_concordance, generate_keywords_over_time_plot,
                                  generate_stacked_keywords_over_time_plot, generate_keywords)
from App.utils import collapsible_section, remove_png_files
from colors import my_orange

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
    keywords_choices = reactive.Value(generate_keywords("Gaza before conflict"))

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
            f'<h3 style="text-align: center;">Concordance Results with {ngram_number}-Grams for {input.filter_words()} '
            f'- {dataset_name}</h3>'
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

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def on_dataset_change():
        remove_png_files()

    @reactive.Effect
    @reactive.event(input.dataset_filter)
    def update_keywords_choices():
        dataset_name = input.dataset_filter()
        keywords_choices.set(generate_keywords(dataset_name))