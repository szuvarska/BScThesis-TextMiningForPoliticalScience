import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from App.single_analysis import entity_legend, analyse_single_article, most_common_words_plot_single
import plotly.express as px

color_palette = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33', '#8C33FF'
]
entity_colors = {entity: color_palette[i % len(color_palette)] for i, entity in enumerate(entity_legend.keys())}


def entity_types_plot_double(entity_sentiments_1: pd.DataFrame, entity_sentiments_2: pd.DataFrame):
    entity_types_1 = entity_sentiments_1["Entity Type"].value_counts()
    entity_types_1 = entity_types_1.reindex(entity_legend.keys()).fillna(0)
    entity_types_1 = entity_types_1.reset_index()
    entity_types_1.columns = ["Entity Type", "Frequency"]

    entity_types_2 = entity_sentiments_2["Entity Type"].value_counts()
    entity_types_2 = entity_types_2.reindex(entity_legend.keys()).fillna(0)
    entity_types_2 = entity_types_2.reset_index()
    entity_types_2.columns = ["Entity Type", "Frequency"]

    entity_types_combined = pd.concat(
        [entity_types_1.set_index('Entity Type').T, entity_types_2.set_index('Entity Type').T])
    entity_types_combined.index = ["Article 1", "Article 2"]

    fig = go.Figure()
    for entity_type in entity_legend.keys():
        fig.add_trace(
            go.Bar(
                name=entity_type,
                x=entity_types_combined.index,
                y=entity_types_combined[entity_type],
                marker=dict(color=entity_colors[entity_type]),
            )
        )
    # Update layout
    fig.update_layout(
        barmode='group',  # Group bars side by side
        title=dict(
            text='Most Common Named Entity Types for Both Articles',
            x=0.5  # Center-align the title
        ),
        xaxis_title='',
        yaxis=dict(
            title='Frequency',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        xaxis=dict(tickangle=0),  # Keep x-axis labels horizontal
        legend_title='Entity Type',
        legend=dict(traceorder='normal'),
        height=600,
        width=800
    )

    return fig


def most_common_entities_plot_double(entity_sentiments_1: pd.DataFrame, entity_sentiments_2: pd.DataFrame):
    entity_counts_1 = entity_sentiments_1["Entity"].value_counts().head(5)
    entity_counts_1 = entity_counts_1.reset_index()
    entity_counts_1.columns = ["Entity", "Frequency"]

    entity_counts_2 = entity_sentiments_2["Entity"].value_counts().head(5)
    entity_counts_2 = entity_counts_2.reset_index()
    entity_counts_2.columns = ["Entity", "Frequency"]

    # Merge the two DataFrames on the Entity column
    entity_counts_combined = pd.merge(
        entity_counts_1,
        entity_counts_2,
        on="Entity",
        how="outer",  # Ensures all entities from both DataFrames are included
        suffixes=("_1", "_2")  # Adds suffixes to distinguish the Frequency columns
    )
    entity_counts_combined.fillna(0, inplace=True)

    entity_types = entity_counts_combined["Entity"]
    article_1_values = -entity_counts_combined["Frequency_1"]  # Negative for mirroring
    article_2_values = entity_counts_combined["Frequency_2"]

    fig = go.Figure()

    # Add bar for Article 1
    fig.add_trace(go.Bar(
        y=entity_types,
        x=article_1_values,
        orientation='h',
        name='Article 1',
        marker=dict(color='skyblue')
    ))

    # Add bar for Article 2
    fig.add_trace(go.Bar(
        y=entity_types,
        x=article_2_values,
        orientation='h',
        name='Article 2',
        marker=dict(color='salmon')
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text="Most Common Named Entities for Both Articles",
            x=0.5  # Center-align the title
        ),
        xaxis_title="Frequency",
        xaxis=dict(
            tickvals=[i for i in range(-int(max(article_1_values.abs().max(), article_2_values.max())),
                                       int(max(article_1_values.abs().max(), article_2_values.max())) + 1)],
            ticktext=[abs(i) for i in range(-int(max(article_1_values.abs().max(), article_2_values.max())),
                                            int(max(article_1_values.abs().max(), article_2_values.max())) + 1)],
        ),
        yaxis=dict(
            title='Entity',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        barmode='relative',
        legend=dict(title='Articles'),
        coloraxis_showscale=False,
    )

    return fig


def sentiment_dist_plot_double(sentiment_1: pd.DataFrame, sentiment_2: pd.DataFrame, base: str = "Sentences"):
    sentiment_1 = sentiment_1["Sentiment"].value_counts()
    sentiment_2 = sentiment_2["Sentiment"].value_counts()
    sentiment_comparison = pd.DataFrame({
        'Negative': [sentiment_1.get('negative', 0), sentiment_2.get('negative', 0)],
        'Neutral': [sentiment_1.get('neutral', 0), sentiment_2.get('neutral', 0)],
        'Positive': [sentiment_1.get('positive', 0), sentiment_2.get('positive', 0)]
    }, index=['Article 1', 'Article 2'])

    fig = go.Figure()
    sentiments = ['Positive', 'Neutral', 'Negative']
    colors = ['green', 'gray', 'red']

    for sentiment, color in zip(sentiments, colors):
        fig.add_trace(
            go.Bar(
                name=sentiment,
                x=sentiment_comparison.index,
                y=sentiment_comparison[sentiment],
                marker_color=color
            )
        )
    # Update layout
    fig.update_layout(
        barmode='group',  # Group bars side by side
        title=dict(
            text=f'Sentiment Distribution Based on {base} for Both Articles',
            x=0.5  # Center-align the title
        ),
        xaxis_title='',
        yaxis=dict(
            title='Count',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        xaxis=dict(tickangle=0),  # Keep x-axis labels horizontal
        legend_title='Sentiment',
        legend=dict(traceorder='normal'),
        height=600,
        width=800
    )

    return fig

#
# def most_common_words_plot_single(sentiment_sentences: pd.DataFrame, N=100):
#     stop_words = set(nltk.corpus.stopwords.words("english"))
#     fdist = FreqDist(
#         [word for word in word_tokenize(' '.join(sentiment_sentences["Sentence"])) if
#          word.lower() not in stop_words and word.isalpha()])
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
#         dict(fdist.most_common(N)))
#     plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')
#     plt.axis('off')
#     plt.title(f'Top {N} Most Common Words')
#     plt.tight_layout()
#     return plt.gcf()

if __name__ == "__main__":
    file1 = "../BRAT_Data/Gaza_after/Articles_for_Agnieszka/2023-10-22_DIPLOMACY_Israel_s_deepening_attacks_in_Gaza_likely_to_embro.txt"
    # file2 = "../BRAT_Data/Gaza_after/Articles_for_Agnieszka/2023-10-22_VIEWPOINT_US_biased_attitude_in_Israel-Palestine_conflict_ma.txt"
    article_text_1 = ''.join(open(file1, "r").readlines()[:20])
    # article_text_2 = ''.join(open(file2, "r").readlines()[:20])
    _, entity_sentiments_1, sentiment_sentences_1 = analyse_single_article(article_text_1)
    # _, entity_sentiments_2, sentiment_sentences_2 = analyse_single_article(article_text_2)
    fig = most_common_words_plot_single(sentiment_sentences_1, article=" from Article 1")
    fig.show()
