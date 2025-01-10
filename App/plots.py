from pathlib import Path
import sys
import ast
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from NER_and_ED.NER_ED_script import find_most_common_entity_types, find_most_common_entities_per_type_for_shiny

from Preparations.EDA_script import plot_word_count_distribution, sentance_count_distribution, plot_top_N_common_words, \
    plot_top_N_common_pos, plot_pos_wordclouds_for_shiny, load_pos_dict

from Sentiment.sentiment_script import calculate_sentiment_dist, calculate_sentiment_over_time, \
    generate_word_clouds, calculate_sentiment_dist_per_target, calculate_sentiment_over_time_per_target, \
    calculate_sentiment_dist_over_time_by_target_for_shiny

from Community_detection.community_graph_script import plot_community_graph

from NGrams.NGrams_script import visualize_bigrams, concordance

from Topics.topics_vis import make_topic_over_time_df, plot_topic_over_time, plot_stacked_topics_over_time


def generate_entity_types_plot(dataset_name: str):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9]
    dataset_name = dataset_name.replace('during', 'after')
    ner_df = pd.read_csv(f"NER_and_ED/Results/{dataset_name}_with_NER.csv")
    ner_df['NER'] = ner_df['NER'].apply(ast.literal_eval)
    return find_most_common_entity_types(ner_df, dataset_name_to_display, for_shiny=True)


def generate_most_common_entities_plot(dataset_name: str, entity_type: str):
    dataset_name_to_display = dataset_name
    entity_type_mapping = {"Person": "PER", "Organisation": "ORG", "Location": "LOC", "Miscellaneous": "MISC"}
    entity_type_short = entity_type_mapping.get(entity_type, entity_type)
    dataset_name = dataset_name.replace(' ', '_')[:-9]
    dataset_name = dataset_name.replace('during', 'after')
    ner_df_path = f"NER_and_ED/Results/{dataset_name}_top_40_entities.csv"
    return find_most_common_entities_per_type_for_shiny(dataset_name_to_display, ner_df_path, entity_type,
                                                        entity_type_short)


def generate_sentiment_dist_plot(dataset_name: str):
    tsc_results_df = pd.read_csv(f"Sentiment/Results/tsc_{dataset_name}.csv")
    vader_results_df = pd.read_csv(f"Sentiment/Results/vader_{dataset_name}.csv")
    return calculate_sentiment_dist(tsc_results_df, vader_results_df, dataset_name, for_shiny=True)


def generate_sentiment_over_time_plot(dataset_name: str, model_name: str):
    results_df = pd.read_csv(f"Sentiment/Results/{model_name}_{dataset_name}.csv")
    return calculate_sentiment_over_time(results_df, dataset_name, for_shiny=True, model_name=model_name)


def generate_sentiment_word_cloud_plot(dataset_name: str, model_name: str, sentiment: str):
    sentiment = sentiment.lower()
    results_df = pd.read_csv(f"Sentiment/Results/{model_name}_{dataset_name}.csv")
    return generate_word_clouds(results_df, dataset_name, for_shiny=True, model_name=model_name, sentiment=sentiment)


def generate_sentiment_dist_per_target_plot(dataset_name: str):
    tsc_results_df = pd.read_csv(f"Sentiment/Results/tsc_{dataset_name}.csv")
    return calculate_sentiment_dist_per_target(tsc_results_df, dataset_name, for_shiny=True)


def generate_sentiment_over_time_per_target_plot(dataset_name: str):
    tsc_results_df = pd.read_csv(f"Sentiment/Results/tsc_{dataset_name}.csv")
    return calculate_sentiment_over_time_per_target(tsc_results_df, dataset_name, for_shiny=True)


def generate_sentiment_dist_over_time_by_target_plot(dataset_name: str, sentiment: str):
    sentiment = sentiment.lower()
    tsc_results_df = pd.read_csv(f"Sentiment/Results/tsc_{dataset_name}.csv")
    return calculate_sentiment_dist_over_time_by_target_for_shiny(tsc_results_df, dataset_name, sentiment=sentiment)


def generate_word_count_distribution_plot(dataset_name: str):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Preparations/Data_for_EDA/df_{dataset_name}.csv")
    df = df[(df['article_category_one'] != "PHOTO") & (df['article_text'].notnull())]
    return plot_word_count_distribution(df, dataset_name_to_display)


def generate_sentence_count_distribution_plot(dataset_name: str):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Preparations/Data_for_EDA/df_{dataset_name}.csv")
    df = df[(df['article_category_one'] != "PHOTO") & (df['article_text'].notnull())]
    return sentance_count_distribution(df, dataset_name_to_display)


def generate_top_N_common_words_plot(dataset_name: str, N: int = 100):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Preparations/Data_for_EDA/df_{dataset_name}.csv")
    df = df[(df['article_category_one'] != "PHOTO") & (df['article_text'].notnull())]
    return plot_top_N_common_words(df, dataset_name_to_display, N)


def generate_top_N_common_pos_plot(dataset_name: str, N: int = 100):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Preparations/Data_for_EDA/df_pos_{dataset_name}.csv")
    return plot_top_N_common_pos(df, dataset_name_to_display, N)


def generate_pos_wordclouds_plot(dataset_name: str, N: int = 100, pos: str = 'Common Singular Nouns'):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Preparations/Data_for_EDA/df_{dataset_name}.csv")
    df = df[(df['article_category_one'] != "PHOTO") & (df['article_text'].notnull())]
    return plot_pos_wordclouds_for_shiny(df, dataset_name_to_display, N, pos)


def generate_community_graph(dataset_name: str):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9]
    dataset_name = dataset_name.replace('during', 'after')
    df_ner = pd.read_csv(f"Data/Sentances_df/sentences_{dataset_name.lower()}.csv")
    df_entities = pd.read_csv(f"NER_and_ED/Results/{dataset_name}_top_40_entities.csv")
    df_entities = df_entities[~df_entities.Word.isin(["U", "B", "N", "19", "G", "S"])].head(120)
    suptitle = f"{dataset_name_to_display}\n Co-occurrence in Same Sentence Relationship Graph"
    title = "Nodes represent entities. Edges represent co-occurrence within the same sentence.\nNodes size indicates the node strength.\nEdge width indicates the frequency of co-occurrence. Spring Layout"
    plot = plot_community_graph(df_ner, df_entities, suptitle=suptitle, title=title,
                                nodes_displayed=25, layout="spring", edge="std")
    plot.tight_layout()
    image_path = f"App/www/community_graph_{dataset_name}.png"
    plot.savefig(image_path)
    plot.close()
    return image_path


def generate_pos_choices():
    pos_dict = load_pos_dict(key='abbr')
    return list(pos_dict.values())


def generate_bigrams_plot(dataset_name: str):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Preparations/Data_for_EDA/df_{dataset_name}.csv")
    df = df[(df['article_category_one'] != "PHOTO") & (df['article_text'].notnull())]
    plot = visualize_bigrams(df, 10, dataset_name_to_display)
    plot.tight_layout()
    image_path = f"App/www/bigrams_plot_{dataset_name}.png"
    plot.savefig(image_path)
    plot.close()
    return image_path


def generate_concordance(dataset_name: str, filter: list, ngram_number: int):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Preparations/Data_for_EDA/df_{dataset_name}.csv")
    df = df[(df['article_category_one'] != "PHOTO") & (df['article_text'].notnull())]
    return concordance(df, filter, ngram_number)


def generate_keywords_over_time_plot(dataset_name: str, top_n: int = 10):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Data/Sentances_df/sentences_{dataset_name.lower()}.csv")
    return plot_topic_over_time(df, top_n, dataset_name_to_display)


def generate_stacked_keywords_over_time_plot(dataset_name: str, my_words: list[str], aggregation: str = 'monthly'):
    dataset_name_to_display = dataset_name
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Data/Sentances_df/sentences_{dataset_name.lower()}.csv")
    return plot_stacked_topics_over_time(df, my_words, aggregation, dataset_name_to_display)


def generate_keywords(dataset_name: str):
    dataset_name = dataset_name.replace(' ', '_')[:-9].lower()
    dataset_name = dataset_name.replace('during', 'after')
    df = pd.read_csv(f"Data/Sentances_df/sentences_{dataset_name}.csv")
    keywords_over_time_df = make_topic_over_time_df(df)
    keywords_to_select = keywords_over_time_df['key_word'].unique().tolist()
    keywords_to_select.sort()
    return keywords_to_select
