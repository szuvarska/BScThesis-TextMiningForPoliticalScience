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


def calculate_sentiment_dist(tsc_results_df: pd.DataFrame, vader_results_df: pd.DataFrame, dataset_name: str):
    # aggregate sentiment counts for TSC
    tsc_sentiment_counts = tsc_results_df['Sentiment'].value_counts()

    # aggregate sentiment counts for VADER
    vader_sentiment_counts = vader_results_df['Sentiment'].value_counts()

    # comparison between TSC and VADER
    sentiment_comparison = pd.DataFrame({
        'Negative': [tsc_sentiment_counts.get('negative', 0), vader_sentiment_counts.get('negative', 0)],
        'Neutral': [tsc_sentiment_counts.get('neutral', 0), vader_sentiment_counts.get('neutral', 0)],
        'Positive': [tsc_sentiment_counts.get('positive', 0), vader_sentiment_counts.get('positive', 0)]
    }, index=['TSC', 'VADER'])

    print(sentiment_comparison)

    sentiment_comparison.plot(kind='bar', color=['red', 'gray', 'green'], figsize=(10, 6))
    plt.title(f'Comparison of Overall Sentiment Distribution (TSC vs VADER) - {dataset_name}')
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
    plt.tight_layout()
    plt.savefig(f"Plots/sentiment_dist_{dataset_name}.png")
    plt.show()


def calculate_sentiment_over_time(tsc_results_df: pd.DataFrame, vader_results_df: pd.DataFrame, dataset_name: str):
    # process TSC results by month
    tsc_results_df['published_time'] = pd.to_datetime(tsc_results_df['published_time'], format='%Y-%m-%d')
    tsc_results_df['month'] = tsc_results_df['published_time'].dt.to_period('M')  # Convert date to monthly periods
    tsc_sentiment_counts = tsc_results_df.pivot_table(
        index=['month'],  # group by month
        columns='Sentiment',
        aggfunc='size',
        fill_value=0
    )

    # normalize TSC sentiment proportions
    tsc_sentiment_proportions = tsc_sentiment_counts.div(tsc_sentiment_counts.sum(axis=1), axis=0)

    # process VADER results by month
    vader_results_df['published_time'] = pd.to_datetime(vader_results_df['published_time'], format='%Y-%m-%d')
    vader_results_df['month'] = vader_results_df['published_time'].dt.to_period('M')  # Convert date to monthly periods
    vader_sentiment_counts = vader_results_df.pivot_table(
        index=['month'],  # group by month
        columns='Sentiment',
        aggfunc='size',
        fill_value=0
    )

    # normalize VADER sentiment proportions
    vader_sentiment_proportions = vader_sentiment_counts.div(vader_sentiment_counts.sum(axis=1), axis=0)

    # Plot 1: TSC Sentiment Proportions Over Time (Monthly)
    plt.figure(figsize=(12, 12))
    plt.subplot(12, 6, 1)
    tsc_sentiment_proportions.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'])
    plt.title(f'TSC Sentiment Proportions Over Time (Monthly) - {dataset_name}')
    plt.xlabel('Month')
    plt.ylabel('Proportion of Sentiment')
    plt.legend(title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=60)

    # Plot 2: VADER Sentiment Proportions Over Time (Monthly)
    plt.subplot(12, 6, 2)
    vader_sentiment_proportions.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'])  # TODO: fix plots
    plt.title(f'VADER Sentiment Proportions Over Time (Monthly) - {dataset_name}')
    plt.xlabel('Month')
    plt.ylabel('Proportion of Sentiment')
    plt.legend(title='Sentiment', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=60)
    # plt.tight_layout()
    # plt.show()

    plt.tight_layout()
    plt.savefig(f"Plots/sentiment_over_time_{dataset_name}.png")
    plt.show()


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


# generate word cloud for TSC sentiment (positive, negative, neutral)
def wc_tsc(sentiment_label, df, dataset_name: str):
    sentences = clean_sentences(df, sentiment_label)
    if sentences:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentences)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment_label.capitalize()} Sentences (TSC) - {dataset_name}')


def generate_word_clouds(tsc_results_df: pd.DataFrame, vader_results_df: pd.DataFrame, dataset_name: str):
    plt.figure(figsize=(16, 14))
    plt.subplot(3, 2, 1)
    wc_vader('positive', vader_results_df, dataset_name)
    plt.subplot(3, 2, 3)
    wc_vader('negative', vader_results_df, dataset_name)
    plt.subplot(3, 2, 5)
    wc_vader('neutral', vader_results_df, dataset_name)

    plt.subplot(3, 2, 2)
    wc_tsc('positive', tsc_results_df, dataset_name)
    plt.subplot(3, 2, 4)
    wc_tsc('negative', tsc_results_df, dataset_name)
    plt.subplot(3, 2, 6)
    wc_tsc('neutral', tsc_results_df, dataset_name)
    plt.tight_layout()
    plt.savefig(f"Plots/sentiment_wordclouds_{dataset_name}.png")
    plt.show()


def calculate_sentiment_dist_per_target(tsc_results_df: pd.DataFrame, dataset_name: str):
    # overall sentiment distribution per target

    # group by target and sentiment, count the occurrences
    overall_sentiment_per_target = tsc_results_df.groupby(['Target', 'Sentiment']).size().unstack(fill_value=0)
    print(overall_sentiment_per_target)

    # calculate sentiment proportions per target
    overall_sentiment_per_target_proportion = overall_sentiment_per_target.div(overall_sentiment_per_target.sum(axis=1),
                                                                               axis=0)
    overall_sentiment_per_target_proportion['Overall Sentiment'] = overall_sentiment_per_target_proportion[
        ['positive', 'negative', 'neutral']].idxmax(axis=1)
    print(overall_sentiment_per_target_proportion)

    overall_sentiment_per_target_proportion.to_csv(f'Results/overall_sentiment_per_target_{dataset_name}.csv')

    # plot stacked bar chart for overall sentiment distribution per target
    plt.figure(figsize=(12, 8))
    overall_sentiment_per_target_proportion.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'])
    plt.title(f'Overall Sentiment Distribution per Target (TSC) - {dataset_name}\n')
    plt.xlabel('Target')
    plt.ylabel('Proportion of Sentiment')
    plt.xticks(rotation=75)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"Plots/sentiment_dist_per_target_{dataset_name}.png")
    plt.show()


def calculate_sentiment_over_time_per_target(tsc_results_df: pd.DataFrame, dataset_name: str):
    # sentiment over time by target - monthly

    tsc_results_df['published_time'] = pd.to_datetime(
        tsc_results_df['published_time'])  # TODO: check if time loads correctly
    # convert date to monthly periods
    tsc_results_df['month'] = tsc_results_df['published_time'].dt.to_period('M')

    # group by month and target, count sentiment occurrences
    sentiment_over_time = tsc_results_df.groupby(['month', 'Target', 'Sentiment']).size().unstack(fill_value=0)

    # normalize to get sentiment proportions over time
    sentiment_over_time_proportion = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0)

    # plot sentiment over time for each target (monthly) using stacked bar chart
    for target in sentiment_over_time_proportion.index.get_level_values('Target').unique():
        target_data = sentiment_over_time_proportion.xs(target, level='Target')

        # convert PeriodIndex to string for plotting
        months = target_data.index.astype(str)

        target_data.plot(kind='bar', stacked=True, color=['red', 'gray', 'green'], figsize=(10, 6))
        plt.title(f'Sentiment Over Time for {target} (Monthly) - {dataset_name}\n')
        plt.xlabel('Month')
        plt.ylabel('Proportion of Sentiment')
        plt.xticks(rotation=45)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.065), ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"Plots/sentiment_over_time_per_target_{target}_{dataset_name}.png")
        plt.show()


def caluclate_sentiment_dist_over_time_by_target(tsc_results_df: pd.DataFrame, dataset_name: str):
    tsc_results_df['published_time'] = pd.to_datetime(tsc_results_df['published_time'])
    tsc_results_df['month'] = tsc_results_df['published_time'].dt.to_period('M')

    # heatmap (target vs. month)
    heatmap_data_positive = tsc_results_df.pivot_table(index='Target', columns='month', values='Sentiment',
                                                       aggfunc=lambda x: (x == 'positive').mean())
    heatmap_data_negative = tsc_results_df.pivot_table(index='Target', columns='month', values='Sentiment',
                                                       aggfunc=lambda x: (x == 'negative').mean())
    heatmap_data_neutral = tsc_results_df.pivot_table(index='Target', columns='month', values='Sentiment',
                                                      aggfunc=lambda x: (x == 'neutral').mean())

    # heatmap showing the proportion of positive sentiment by target over time (monthly)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data_positive, cmap='Greens', annot=False, cbar=True)
    plt.title(f'Proportion of Positive Sentiment by Target Over Time (Monthly) - {dataset_name}')
    plt.xlabel('Month')
    plt.ylabel('Target')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Plots/positive_sentiment_over_time_by_target_{dataset_name}.png")
    plt.show()

    # heatmap showing the proportion of negative sentiment by target over time (monthly)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data_negative, cmap='Reds', annot=False, cbar=True)
    plt.title(f'Proportion of Negative Sentiment by Target Over Time (Monthly) - {dataset_name}')
    plt.xlabel('Month')
    plt.ylabel('Target')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Plots/negative_sentiment_over_time_by_target_{dataset_name}.png")
    plt.show()

    # heatmap showing the proportion of neutral sentiment by target over time (monthly)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data_neutral, cmap='Greys', annot=False, cbar=True)
    plt.title(f'Proportion of Neutral Sentiment by Target Over Time (Monthly) - {dataset_name}')
    plt.xlabel('Month')
    plt.ylabel('Target')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Plots/neutral_sentiment_over_time_by_target_{dataset_name}.png")
    plt.show()


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
