import pandas as pd
import ast
import plotly.express as px
from colors import my_orange, my_red, my_green, my_blue, my_yellow, my_gray, my_purple, my_lightblue

def calculate_mean_sentiment(df: pd.DataFrame):
    df_aggregated = df
    df_aggregated['sentiment_number'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else -1 if x == 'negative' else 0)
    return df_aggregated.groupby('article_id')['sentiment_number'].mean()


def aggregate_topics(df:pd.DataFrame):
    all_topics = [[] for _ in range(max(df['article_id'] + 1))]
    for i in range(len(df)):
        if df['topic_id'][i] >= 0:
            topic_representation_list = df['topic_representation'][i].replace('[', '').replace(']', '').replace(',', '').replace("'", '').split()
            all_topics[df['article_id'][i]].extend(topic_representation_list)

    aggregated_topics = []
    for article_id, topics in enumerate(all_topics):
        topic_word_count = {}
        topic_value = 10

        for word in topics:
            topic_word_count[word] = topic_word_count.get(word, 0) + topic_value
            topic_value -= 1
            if topic_value == 0: topic_value = 10

        top_10_words = sorted(topic_word_count.items(), key=lambda x: x[1], reverse=True)[:10]
        aggregated_topics.append(top_10_words)

    aggregated_topics_df = pd.DataFrame( {'article_id': range(len(aggregated_topics)), 'aggregated_topics': aggregated_topics})
    return aggregated_topics_df

def make_df_for_plots(df: pd.DataFrame):
    aggregated_topics = aggregate_topics(df)
    sentiment_df = calculate_mean_sentiment(df)
    final_df = pd.merge(aggregated_topics, sentiment_df, on='article_id', how='left')
    final_df = pd.merge(final_df, df[['article_id', 'published_time']].drop_duplicates(), on='article_id', how='left')
    return final_df

def make_topic_over_time_df(df: pd.DataFrame):
    plot_df = make_df_for_plots(df)

    def safe_convert_to_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                return []
        return []

    plot_df['aggregated_topics'] = plot_df['aggregated_topics'].map(safe_convert_to_list)

    exploded_df = plot_df.explode('aggregated_topics')

    exploded_df = exploded_df[
        exploded_df['aggregated_topics'].apply(lambda x: isinstance(x, tuple) and len(x) == 2)
    ]

    exploded_df[['key_word', 'occurrences']] = pd.DataFrame(
        exploded_df['aggregated_topics'].tolist(), index=exploded_df.index
    )

    topics_over_time_df = (
        exploded_df.groupby(['published_time', 'key_word'])['occurrences']
        .sum()
        .reset_index()
    )

    return topics_over_time_df

def plot_topic_over_time(df: pd.DataFrame, top_n: int = 10):

    topics_over_time_df = make_topic_over_time_df(df)
    N = top_n
    total_occurrences = topics_over_time_df.groupby('key_word')['occurrences'].sum()
    top_words = total_occurrences.nlargest(N).index
    top_topics_over_time_df = topics_over_time_df[topics_over_time_df['key_word'].isin(top_words)]

    fig = px.line(top_topics_over_time_df,
                  x="published_time",
                  y="occurrences",
                  color="key_word",
                  color_discrete_sequence=[my_orange, my_red, my_green, my_blue, my_yellow, my_gray, my_purple, my_lightblue],
                  title="Top Topics Over Time",
                  template="plotly_white")

    # fig.update_traces(visible="legendonly")  # Initially hide all lines but keep them visible in the legend

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="Deselect All",
                        method="update",
                        args=[{"visible": ["legendonly"] * len(fig.data)}]  # Hide all lines
                    ),
                    dict(
                        label="Select All",
                        method="update",
                        args=[{"visible": [True] * len(fig.data)}]  # Show all lines
                    )
                ]
            )
        ]
    )

    return fig