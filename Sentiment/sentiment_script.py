from NewsSentiment import TargetSentimentClassifier
from NewsSentiment.customexceptions import TargetNotFoundException, TooLongTextException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import spacy
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import plotly.graph_objects as go
import re


def vader_sentiment(text: str) -> str:
    vader_analyzer = SentimentIntensityAnalyzer()
    scores = vader_analyzer.polarity_scores(text)
    # define the thresholds to categorize it
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# split each article into sentences
def split_into_sentences(article):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(article)
    return [sent.text for sent in doc.sents]


def perform_sentiment_analysis(file_name: str, target_entities: list, dataset_name: str) -> tuple[pd.DataFrame,
pd.DataFrame]:
    # load dataset
    df = pd.read_csv(file_name)

    # check and remove incomplete data
    check_condition = (df['article_text'].isna()) | (df['article_text'] == "")
    df = df[~check_condition]

    tsc = TargetSentimentClassifier()
    # TODO: moving preprocessing to a different script (it recurs in NER)

    # initialize lists for comparison results
    vader_results = []
    tsc_results = []

    # threshold max sentence length
    MAX_SENTENCE_LENGTH = 100

    # process each article for both models
    for idx, row in df.iterrows():
        article_text = row['article_text']
        sentences = split_into_sentences(article_text)

        for sentence in sentences:
            original_sentence = sentence
            sentence = sentence.lower()

            # skip sentences that exceed the maximum length threshold
            if len(sentence.split()) > MAX_SENTENCE_LENGTH:
                print(f"Skipping long sentence.")
                continue

            # TSC Sentiment Analysis
            for target in target_entities:
                if target.lower() in sentence:
                    entity_start = sentence.find(target.lower())
                    entity_end = entity_start + len(target)
                    left_context = sentence[:entity_start]
                    right_context = sentence[entity_end:]

                    try:
                        sentiment_tsc = tsc.infer_from_text(left_context, target, right_context)
                    except TooLongTextException:
                        print(f"TooLongTextException: {target} - Sentence too long for TSC")
                        continue  # move on to the next target
                    except TargetNotFoundException:
                        print(f"TargetNotFoundException: {target} not found in {sentence}")
                        continue  # move on to the next target

                    sentiment_label_tsc = sentiment_tsc[0]['class_label'].lower()

                    # store TSC result (with target)
                    tsc_results.append({
                        'Model': 'TSC',
                        'Sentence': original_sentence,
                        'Target': target,
                        'Sentiment': sentiment_label_tsc,
                        'published_time': row['published_time']
                    })

            # VADER (general sentiment analysis for the entire sentence)
            sentiment_vader = vader_sentiment(sentence)

            # store VADER result
            vader_results.append({
                'Model': 'VADER',
                'Sentence': original_sentence,
                'Sentiment': sentiment_vader,
                'published_time': row['published_time']
            })

    # convert both results to DataFrames
    tsc_results_df = pd.DataFrame(tsc_results)
    vader_results_df = pd.DataFrame(vader_results)

    # save results
    tsc_results_df.to_csv(f'Results/tsc_{dataset_name}.csv', index=False)
    vader_results_df.to_csv(f'Results/vader_{dataset_name}.csv', index=False)

    return tsc_results_df, vader_results_df


def calculate_sentiment_dist(tsc_results_df: pd.DataFrame, vader_results_df: pd.DataFrame, dataset_name: str,
                             for_shiny=False):
    # aggregate sentiment counts for TSC
    tsc_sentiment_counts = tsc_results_df['Sentiment'].value_counts()

    # aggregate sentiment counts for VADER
    vader_sentiment_counts = vader_results_df['Sentiment'].value_counts()

    #comparison between TSC and VADER
    sentiment_comparison = pd.DataFrame({
        'Negative': [tsc_sentiment_counts.get('negative', 0), vader_sentiment_counts.get('negative', 0)],
        'Neutral': [tsc_sentiment_counts.get('neutral', 0), vader_sentiment_counts.get('neutral', 0)],
        'Positive': [tsc_sentiment_counts.get('positive', 0), vader_sentiment_counts.get('positive', 0)]
    }, index=['TSC', 'VADER'])

    if not for_shiny:
        print(sentiment_comparison)
    #
    # sentiment_comparison.plot(kind='bar', color=['red', 'gray', 'green'], figsize=(10, 6))
    # plt.title(f'Comparison of Overall Sentiment Distribution (TSC vs VADER) - {dataset_name}')
    # plt.xlabel('Model')
    # plt.ylabel('Count')
    # plt.xticks(rotation=0)
    # plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
    # plt.tight_layout()
    # plt.savefig(f"Plots/sentiment_dist_{dataset_name}.png")
    # plt.show()

    fig = go.Figure()
    sentiments = ['Negative', 'Neutral', 'Positive']
    colors = ['red', 'gray', 'green']

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
        title=f'Comparison of Overall Sentiment Distribution (TSC vs VADER) - {dataset_name}',
        xaxis_title='Model',
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
        height=600,
        width=800
    )

    # Save the plot as an image

    if for_shiny:
        return fig
    else:
        #fig.write_image(f"Plots/sentiment_dist_{dataset_name}.png")
        fig.show()


def calculate_sentiment_over_time(model_results_df: pd.DataFrame, dataset_name: str, for_shiny=False, model_name='tsc'):
    # # process TSC results by month
    # tsc_results_df['published_time'] = pd.to_datetime(tsc_results_df['published_time'], format='%Y-%m-%d')
    # tsc_results_df['month'] = tsc_results_df['published_time'].dt.to_period('M')  # Convert date to monthly periods
    # tsc_sentiment_counts = tsc_results_df.pivot_table(
    #     index=['month'],  # group by month
    #     columns='Sentiment',
    #     aggfunc='size',
    #     fill_value=0
    # )
    #
    # # normalize TSC sentiment proportions
    # tsc_sentiment_proportions = tsc_sentiment_counts.div(tsc_sentiment_counts.sum(axis=1), axis=0)
    #
    # # process VADER results by month
    # vader_results_df['published_time'] = pd.to_datetime(vader_results_df['published_time'], format='%Y-%m-%d')
    # vader_results_df['month'] = vader_results_df['published_time'].dt.to_period('M')  # Convert date to monthly periods
    # vader_sentiment_counts = vader_results_df.pivot_table(
    #     index=['month'],  # group by month
    #     columns='Sentiment',
    #     aggfunc='size',
    #     fill_value=0
    # )
    #
    # # normalize VADER sentiment proportions
    # vader_sentiment_proportions = vader_sentiment_counts.div(vader_sentiment_counts.sum(axis=1), axis=0)

    # Process model results by month
    model_results_df['published_time'] = pd.to_datetime(model_results_df['published_time'], format='%Y-%m-%d')
    model_results_df['month'] = model_results_df['published_time'].dt.to_period('M')  # Convert date to monthly periods
    model_sentiment_counts = model_results_df.pivot_table(
        index=['month'],  # group by month
        columns='Sentiment',
        aggfunc='size',
        fill_value=0
    )

    # Normalize model sentiment proportions
    model_sentiment_proportions = model_sentiment_counts.div(model_sentiment_counts.sum(axis=1), axis=0)

    # Plot model Sentiment Proportions
    fig = go.Figure()
    months = model_sentiment_proportions.index.astype(str)

    for sentiment, color in zip(['positive', 'neutral', 'negative'], ['green', 'gray', 'red']):
        fig.add_trace(
            go.Bar(
                name=sentiment.capitalize(),
                x=months,
                y=model_sentiment_proportions[sentiment],
                marker_color=color
            )
        )

    fig.update_layout(
        barmode='stack',
        title=f'{model_name.upper()} Sentiment Proportions Over Time (Monthly) - {dataset_name}',
        xaxis=dict(
            title='Month',
            tickmode='array',
            tickvals=months,  # Place one label per month
            ticktext=months,  # Show month labels under the grouped bars
            tickangle=60
        ),
        # yaxis=dict(title='Proportion of Sentiment'),
        yaxis=dict(
            title='Proportion of Sentiment',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        legend_title='Sentiment',
        height=600,
        width=1000
    )

    #fig.write_image(f"Plots/{model_name}_sentiment_over_time_{dataset_name}.png")

    if for_shiny:
        return fig
    else:
        fig.show()


def clean_sentences(df, sentiment_label):
    sentences = df[df['Sentiment'] == sentiment_label]['Sentence']
    sentences = " ".join(sentences)
    sentences = re.sub(r'\b[a-zA-Z]\b', '', sentences)
    sentences = re.sub(r'\s+', ' ', sentences).strip()
    return sentences


# generate word cloud for VADER sentiment (positive, negative, neutral)
def wc_vader(sentiment_label, df, dataset_name: str):
    sentences = clean_sentences(df, sentiment_label)
    if sentences:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentences)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment_label.capitalize()} Sentences (VADER) - {dataset_name}')
    return plt


# generate word cloud for TSC sentiment (positive, negative, neutral)
def wc_tsc(sentiment_label, df, dataset_name: str) -> plt:
    sentences = clean_sentences(df, sentiment_label)
    if sentences:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentences)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment_label.capitalize()} Sentences (TSC) - {dataset_name}')
    return plt


def generate_word_clouds(results_df: pd.DataFrame, dataset_name: str, for_shiny=False, model_name='tsc', sentiment='positive'):
    sentiment_to_use = sentiment if sentiment in ['positive', 'negative', 'neutral'] else 'positive'
    if model_name == 'tsc':
        plot = wc_tsc(sentiment_to_use, results_df, dataset_name)
    else:
        plot = wc_vader(sentiment_to_use, results_df, dataset_name)

    plot.tight_layout()

    # if is_tsc:
    #     if sentiment_to_use == 'positive':
    #         plot = wc_tsc('positive', tsc_results_df, dataset_name)
    #     elif sentiment_to_use == 'negative':
    #         plot = wc_tsc('negative', tsc_results_df, dataset_name)
    #     else:
    #         plot = wc_tsc('neutral', tsc_results_df, dataset_name)
    # else:
    #     if sentiment_to_use == 'positive':
    #         plot = wc_vader('positive', vader_results_df, dataset_name)
    #     elif sentiment_to_use == 'negative':
    #         plot = wc_vader('negative', vader_results_df, dataset_name)
    #     else:
    #         plot = wc_vader('neutral', vader_results_df, dataset_name)
    if for_shiny:
        return plot.gcf()
    else:
        plt.show()
    #plt.savefig(f"Plots/{sentiment_to_use}_{model_name}_wordcloud_{dataset_name}.png")


def calculate_sentiment_dist_per_target(tsc_results_df: pd.DataFrame, dataset_name: str, for_shiny=False):
    # # overall sentiment distribution per target
    #
    # # group by target and sentiment, count the occurrences
    # overall_sentiment_per_target = tsc_results_df.groupby(['Target', 'Sentiment']).size().unstack(fill_value=0)
    # print(overall_sentiment_per_target)
    #
    # # calculate sentiment proportions per target
    # overall_sentiment_per_target_proportion = overall_sentiment_per_target.div(overall_sentiment_per_target.sum(axis=1),
    #                                                                            axis=0)
    # overall_sentiment_per_target_proportion['Overall Sentiment'] = overall_sentiment_per_target_proportion[
    #     ['positive', 'negative', 'neutral']].idxmax(axis=1)
    # print(overall_sentiment_per_target_proportion)
    #
    # overall_sentiment_per_target_proportion.to_csv(f'Results/overall_sentiment_per_target_{dataset_name}.csv')
    #
    # # plot stacked bar chart for overall sentiment distribution per target
    # plt.figure(figsize=(12, 8))
    # overall_sentiment_per_target_proportion.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'])
    # plt.title(f'Overall Sentiment Distribution per Target (TSC) - {dataset_name}\n')
    # plt.xlabel('Target')
    # plt.ylabel('Proportion of Sentiment')
    # plt.xticks(rotation=75)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=8)
    # plt.tight_layout()
    # plt.savefig(f"Plots/sentiment_dist_per_target_{dataset_name}.png")
    # plt.show()
    # Overall sentiment distribution per target
    if for_shiny:
        overall_sentiment_per_target_proportion = pd.read_csv(f'../Sentiment/Results/overall_sentiment_per_target_{dataset_name}.csv')
    else:
        overall_sentiment_per_target = tsc_results_df.groupby(['Target', 'Sentiment']).size().unstack(fill_value=0)
        print(overall_sentiment_per_target)

        # Calculate sentiment proportions per target
        overall_sentiment_per_target_proportion = overall_sentiment_per_target.div(
            overall_sentiment_per_target.sum(axis=1),
            axis=0)
        overall_sentiment_per_target_proportion['Overall Sentiment'] = overall_sentiment_per_target_proportion[
            ['positive', 'negative', 'neutral']].idxmax(axis=1)
        print(overall_sentiment_per_target_proportion)

        # Save results to a CSV file
        overall_sentiment_per_target_proportion.to_csv(f'Results/overall_sentiment_per_target_{dataset_name}.csv')

    # Plot horizontal stacked bar chart
    fig = go.Figure()

    # Add traces for each sentiment
    sentiments = ['positive', 'neutral', 'negative', ]
    colors = ['green', 'gray', 'red']

    for sentiment, color in zip(sentiments, colors):
        fig.add_trace(
            go.Bar(
                name=sentiment.capitalize(),
                y=overall_sentiment_per_target_proportion["Target"],  # Targets
                x=overall_sentiment_per_target_proportion[sentiment],  # Proportions
                orientation='h',  # Horizontal bars
                marker_color=color
            )
        )

    # Update layout
    fig.update_layout(
        barmode='stack',  # Stacked bars
        title=f'Overall Sentiment Distribution per Target (TSC) - {dataset_name}',
        xaxis_title='Proportion of Sentiment',
        # yaxis_title='Target',
        yaxis=dict(
            title='Target',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        legend_title='Sentiment',
        height=800,
        width=1000
    )

    # Save and display the plot
    if for_shiny:
        return fig
    else:
        #fig.write_image(f"Plots/sentiment_dist_per_target_{dataset_name}.png")
        fig.show()


def calculate_sentiment_over_time_per_target(tsc_results_df: pd.DataFrame, dataset_name: str, for_shiny=False):
    # # sentiment over time by target - monthly
    #
    # tsc_results_df['published_time'] = pd.to_datetime(
    #     tsc_results_df['published_time'])  # TODO: check if time loads correctly
    # # convert date to monthly periods
    # tsc_results_df['month'] = tsc_results_df['published_time'].dt.to_period('M')
    #
    # # group by month and target, count sentiment occurrences
    # sentiment_over_time = tsc_results_df.groupby(['month', 'Target', 'Sentiment']).size().unstack(fill_value=0)
    #
    # # normalize to get sentiment proportions over time
    # sentiment_over_time_proportion = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0)
    #
    # # plot sentiment over time for each target (monthly) using stacked bar chart
    # for target in sentiment_over_time_proportion.index.get_level_values('Target').unique():
    #     target_data = sentiment_over_time_proportion.xs(target, level='Target')
    #
    #     # convert PeriodIndex to string for plotting
    #     months = target_data.index.astype(str)
    #
    #     target_data.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'], figsize=(10, 6))
    #     plt.title(f'Sentiment Over Time for {target} (Monthly) - {dataset_name}\n')
    #     plt.xlabel('Month')
    #     plt.ylabel('Proportion of Sentiment')
    #     plt.xticks(rotation=45)
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.065), ncol=3, fontsize=8)
    #     plt.tight_layout()
    #     plt.savefig(f"Plots/sentiment_over_time_per_target_{target}_{dataset_name}.png")
    #     plt.show()

    # Ensure published_time is in datetime format
    tsc_results_df['published_time'] = pd.to_datetime(tsc_results_df['published_time'])
    tsc_results_df['month'] = tsc_results_df['published_time'].dt.to_period('M')
    sentiment_over_time = tsc_results_df.groupby(['month', 'Target', 'Sentiment']).size().unstack(fill_value=0)

    # Normalize to get sentiment proportions over time
    sentiment_over_time_proportion = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0)

    # Plot sentiment over time for each target
    for target in sentiment_over_time_proportion.index.get_level_values('Target').unique():
        target_data = sentiment_over_time_proportion.xs(target, level='Target')

        # Convert PeriodIndex to string for plotting
        months = target_data.index.astype(str)

        # Create a Plotly figure for the target
        fig = go.Figure()

        # Add traces for each sentiment
        sentiments = ['positive', 'negative', 'neutral']
        colors = ['green', 'red', 'gray']

        for sentiment, color in zip(sentiments, colors):
            fig.add_trace(
                go.Bar(
                    name=sentiment.capitalize(),
                    x=months,  # Months
                    y=target_data[sentiment],  # Proportions
                    marker_color=color
                )
            )

        # Update layout for the chart
        fig.update_layout(
            barmode='stack',  # Stacked bars
            title=f'Sentiment Over Time for {target} (Monthly) - {dataset_name}',
            xaxis_title='Month',
            # yaxis_title='Proportion of Sentiment',
            yaxis=dict(
                title='Proportion of Sentiment',
                ticklabelposition='outside',
                ticks='outside',
                tickcolor='#fdfdfd',
                ticklen=10,
                automargin=True,
            ),
            xaxis=dict(tickangle=45),
            legend_title='Sentiment',
            height=600,
            width=1000
        )

        if for_shiny:
            return fig
            # Save and display the chart
            #fig.write_image(f"Plots/sentiment_over_time_per_target_{target}_{dataset_name}.png")
            fig.show()


def caluclate_sentiment_dist_over_time_by_target(tsc_results_df: pd.DataFrame, dataset_name: str):
    # Convert 'published_time' to datetime and extract 'month' in YYYY-MM format
    tsc_results_df['published_time'] = pd.to_datetime(tsc_results_df['published_time'])
    tsc_results_df['month'] = tsc_results_df['published_time'].dt.to_period('M').astype(str)  # Format as YYYY-MM

    # Create heatmaps for different sentiment types (positive, negative, neutral)
    heatmap_data_positive = tsc_results_df.pivot_table(index='Target', columns='month', values='Sentiment',
                                                       aggfunc=lambda x: (x == 'positive').mean())
    heatmap_data_negative = tsc_results_df.pivot_table(index='Target', columns='month', values='Sentiment',
                                                       aggfunc=lambda x: (x == 'negative').mean())
    heatmap_data_neutral = tsc_results_df.pivot_table(index='Target', columns='month', values='Sentiment',
                                                      aggfunc=lambda x: (x == 'neutral').mean())

    # Create Plotly Heatmap for Positive Sentiment Proportion
    fig_positive = go.Figure(data=go.Heatmap(
        z=heatmap_data_positive.values,
        x=heatmap_data_positive.columns,  # Use formatted month labels
        y=heatmap_data_positive.index,
        colorscale='Greens',
        colorbar=dict(title="Proportion"),
    ))

    fig_positive.update_layout(
        title=f'Proportion of Positive Sentiment by Target Over Time (Monthly) - {dataset_name}',
        xaxis_title='Month',
        # yaxis_title='Target',
        yaxis=dict(
            title='Target',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(heatmap_data_positive.columns))),
            ticktext=heatmap_data_positive.columns,  # Use formatted month labels
            tickangle=45,
        ),
        height=600,
        width=1000
    )

    #fig_positive.write_image(f"Plots/positive_sentiment_over_time_by_target_{dataset_name}.png")
    fig_positive.show()

    # Create Plotly Heatmap for Negative Sentiment Proportion
    fig_negative = go.Figure(data=go.Heatmap(
        z=heatmap_data_negative.values,
        x=heatmap_data_negative.columns,  # Use formatted month labels
        y=heatmap_data_negative.index,
        colorscale='Reds',
        colorbar=dict(title="Proportion"),
    ))

    fig_negative.update_layout(
        title=f'Proportion of Negative Sentiment by Target Over Time (Monthly) - {dataset_name}',
        xaxis_title='Month',
        # yaxis_title='Target',
        yaxis=dict(
            title='Target',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(heatmap_data_negative.columns))),
            ticktext=heatmap_data_negative.columns,  # Use formatted month labels
            tickangle=45,
        ),
        height=600,
        width=1000
    )

    #fig_negative.write_image(f"Plots/negative_sentiment_over_time_by_target_{dataset_name}.png")
    fig_negative.show()

    # Create Plotly Heatmap for Neutral Sentiment Proportion
    fig_neutral = go.Figure(data=go.Heatmap(
        z=heatmap_data_neutral.values,
        x=heatmap_data_neutral.columns,  # Use formatted month labels
        y=heatmap_data_neutral.index,
        colorscale='Greys',
        colorbar=dict(title="Proportion"),
    ))

    fig_neutral.update_layout(
        title=f'Proportion of Neutral Sentiment by Target Over Time (Monthly) - {dataset_name}',
        xaxis_title='Month',
        # yaxis_title='Target',
        yaxis=dict(
            title='Target',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(heatmap_data_neutral.columns))),
            ticktext=heatmap_data_neutral.columns,  # Use formatted month labels
            tickangle=45,
        ),
        height=600,
        width=1000
    )

    #fig_neutral.write_image(f"Plots/neutral_sentiment_over_time_by_target_{dataset_name}.png")
    fig_neutral.show()


def main():
    file_name = "../Data/gaza_textcontain_after_new_preprocessed.csv"
    dict_name = "dict_gaza.csv"
    dataset_name = "Gaza during conflict"
    target_entities = [
        "Palestinian-Israeli Conflict", "Israel", "Gaza", "Palestine", "US",
        "China", "West", "Saudi Arabia", "West Bank", "Middle East",
        "Houthi", "Hamas", "Rafah", "UN", "Wang Yi",
        "Joe Biden", "Antony Blinken", "Zhang Jun", "Xi Jinping", "Benjamin Netanyahu",
        "Antonio Guterres", "EU", "EU Union"
    ]
    tsc_results_df = pd.read_csv(f'Results/tsc_{dataset_name}.csv')
    vader_results_df = pd.read_csv(f'Results/vader_{dataset_name}.csv')
    calculate_sentiment_dist(tsc_results_df, vader_results_df, dataset_name)
    calculate_sentiment_over_time(tsc_results_df, vader_results_df, dataset_name)
    generate_word_clouds(tsc_results_df, vader_results_df, dataset_name)
    calculate_sentiment_dist_per_target(tsc_results_df, dataset_name)
    caluclate_sentiment_dist_over_time_by_target(tsc_results_df, dataset_name)


if __name__ == '__main__':
    main()
