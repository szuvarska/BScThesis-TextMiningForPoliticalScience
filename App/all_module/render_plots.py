from shiny import render, ui
from shinywidgets import render_widget
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


def setup_plot_outputs(input, output, session):
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
