from shiny import render, ui
from App.utils import collapsible_section
from App.all_module.plots import generate_concordance


def setup_ui_outputs(input, output, session):
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
