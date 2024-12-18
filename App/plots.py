from pathlib import Path
import sys
import ast
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from NER_and_ED.NER_ED_script import find_most_common_entity_types, find_most_common_entities_per_type_for_shiny

from Preparations.EDA_scritpt import plot_word_cout_distribution, sentance_count_distribution, plot_top_N_common_words, \
    plot_top_N_common_pos, plot_pos_wordclouds

from Sentiment.sentiment_script import calculate_sentiment_dist, calculate_sentiment_over_time, \
    generate_word_clouds, calculate_sentiment_dist_per_target, calculate_sentiment_over_time_per_target, \
    calculate_sentiment_dist_over_time_by_target_for_shiny


def generate_entity_types_plot(dataset_name: str):
    dataset_name = dataset_name.replace(' ', '_')[:-9]
    dataset_name = dataset_name.replace('during', 'after')
    ner_df = pd.read_csv(f"../NER_and_ED/Results/{dataset_name}_with_NER.csv")
    ner_df['NER'] = ner_df['NER'].apply(ast.literal_eval)
    return find_most_common_entity_types(ner_df, dataset_name, for_shiny=True)


def generate_most_common_entities_plot(dataset_name: str, entity_type: str):
    entity_type_mapping = {"Person": "PER", "Organisation": "ORG", "Location": "LOC", "Miscellaneous": "MISC"}
    entity_type_short = entity_type_mapping.get(entity_type, entity_type)
    dataset_name = dataset_name.replace(' ', '_')[:-9]
    dataset_name = dataset_name.replace('during', 'after')
    ner_df_path = f"../NER_and_ED/Results/{dataset_name}_top_40_entities.csv"
    return find_most_common_entities_per_type_for_shiny(dataset_name, ner_df_path, entity_type, entity_type_short)


def generate_sentiment_dist_plot(dataset_name: str):
    tsc_results_df = pd.read_csv(f"../Sentiment/Results/tsc_{dataset_name}.csv")
    vader_results_df = pd.read_csv(f"../Sentiment/Results/vader_{dataset_name}.csv")
    return calculate_sentiment_dist(tsc_results_df, vader_results_df, dataset_name, for_shiny=True)


def generate_sentiment_over_time_plot(dataset_name: str, model_name: str):
    results_df = pd.read_csv(f"../Sentiment/Results/{model_name}_{dataset_name}.csv")
    return calculate_sentiment_over_time(results_df, dataset_name, for_shiny=True, model_name=model_name)


def generate_sentiment_word_cloud_plot(dataset_name: str, model_name: str, sentiment: str):
    sentiment = sentiment.lower()
    results_df = pd.read_csv(f"../Sentiment/Results/{model_name}_{dataset_name}.csv")
    return generate_word_clouds(results_df, dataset_name, for_shiny=True, model_name=model_name, sentiment=sentiment)


def generate_sentiment_dist_per_target_plot(dataset_name: str):
    tsc_results_df = pd.read_csv(f"../Sentiment/Results/tsc_{dataset_name}.csv")
    return calculate_sentiment_dist_per_target(tsc_results_df, dataset_name, for_shiny=True)


def generate_sentiment_over_time_per_target_plot(dataset_name: str):
    tsc_results_df = pd.read_csv(f"../Sentiment/Results/tsc_{dataset_name}.csv")
    return calculate_sentiment_over_time_per_target(tsc_results_df, dataset_name, for_shiny=True)

def generate_sentiment_dist_over_time_by_target_plot(dataset_name: str, sentiment: str):
    sentiment = sentiment.lower()
    tsc_results_df = pd.read_csv(f"../Sentiment/Results/tsc_{dataset_name}.csv")
    return calculate_sentiment_dist_over_time_by_target_for_shiny(tsc_results_df, dataset_name, sentiment=sentiment)

# def generate_word_cout_distribution_plot(dataset_name: str):
#     df = pd.read_csv(f"../Preparations/Results/{dataset_name}.csv")
#     return plot_word_cout_distribution(df, dataset_name, for_shiny=True)
#
# def generate_sentance_count_distribution_plot(dataset_name: str):
#     df = pd.read_csv(f"../Preparations/Results/{dataset_name}.csv")
#     return sentance_count_distribution(df, dataset_name, for_shiny=True)
#
# def generate_top_N_common_words_plot(dataset_name: str, N: int):
#     df = pd.read_csv(f"../Preparations/Results/{dataset_name}.csv")
#     return plot_top_N_common_words(df, dataset_name, N, for_shiny=True)
#
# def generate_top_N_common_pos_plot(dataset_name: str, N: int):
#     df = pd.read_csv(f"../Preparations/Results/{dataset_name}.csv")
#     return plot_top_N_common_pos(df, dataset_name, N, for_shiny=True)
#
# def generate_pos_wordclouds_plot(dataset_name: str, N: int):
#     df = pd.read_csv(f"../Preparations/Results/{dataset_name}.csv")
#     return plot_pos_wordclouds(df, dataset_name, N, for_shiny=True)
