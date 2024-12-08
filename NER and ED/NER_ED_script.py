import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import defaultdict
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def standardize_text(text: str, word_dict: dict):
    lower_word_dict = {key.lower(): value for key, value in word_dict.items()}
    sorted_keys = sorted(lower_word_dict.keys(), key=len, reverse=True)

    # pattern that matches any of the keys
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(key) for key in sorted_keys) + r')\b', re.IGNORECASE)

    def replace(match):
        word = match.group(0)
        lower_word = word.lower()
        return lower_word_dict.get(lower_word, word)

    # replace words in the text using the pattern and the replace function
    standardized_text = pattern.sub(replace, text)
    return standardized_text


# function to merge subwords
def merge_subwords(ner_results):
    merged_results = []
    for res in ner_results:
        if res['word'].startswith('##') and merged_results:
            merged_results[-1]['word'] += res['word'][2:]
            merged_results[-1]['end'] = res['end']
        else:
            merged_results.append(res)
    return merged_results


# function to resolve inconsistent labels
def resolve_entity_labels(ner_results):
    entity_dict = defaultdict(lambda: defaultdict(int))

    for entity in ner_results:
        entity_text = entity['word']
        entity_label = entity['entity_group']
        entity_dict[entity_text][entity_label] += 1

    resolved_entities = {}
    for entity_text, labels in entity_dict.items():
        resolved_label = max(labels, key=labels.get)
        resolved_entities[entity_text] = (resolved_label, sum(labels.values()))

    return resolved_entities


# function to consolidate entities
def consolidate_entities(entities):
    consolidated = defaultdict(lambda: defaultdict(int))
    partial_names = set()

    for entity, (entity_group, count) in entities.items():
        parts = entity.split()
        if len(parts) > 1:
            consolidated[entity_group][entity] += count
            # aggregate counts from shorter forms
            for part in parts:
                if part in entities and entities[part][0] == entity_group:
                    consolidated[entity_group][entity] += entities[part][1]
                    partial_names.add(part)
        else:
            if entity not in partial_names:
                consolidated[entity_group][entity] += count

    for part in partial_names:
        if part in consolidated[entity_group]:
            del consolidated[entity_group][part]

    flat_consolidated = {}
    for entity_group, entity_dict in consolidated.items():
        for name, count in entity_dict.items():
            flat_consolidated[name] = (entity_group, count)

    return flat_consolidated


def perform_ner(file_name: str, dict_name: str, output_file: str):
    # load dataset
    df = pd.read_csv(file_name)

    # check and remove incomplete data
    check_condition = (df['article_text'].isna()) | (df['article_text'] == "")
    df = df[~check_condition]

    # Entity Disambiguation
    dict_df = pd.read_csv(dict_name)
    word_dict = pd.Series(dict_df.standard.values, index=dict_df.variation).to_dict()
    df['article_text'] = df['article_text'].apply(lambda x: standardize_text(x, word_dict))

    # load pretrained model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # create NER pipeline
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)

    ner_results_list = []
    for text in tqdm(df['article_text']):
        ner_results = nlp(text)
        merged_results = merge_subwords(ner_results)
        entities = resolve_entity_labels(merged_results)
        consolidated_entities = consolidate_entities(entities)
        ner_results_list.append(consolidated_entities)

    df['NER'] = ner_results_list
    df.to_csv(output_file, index=False)
    return df


# words distribution named entities and non-entities
def count_words_distribution(text):
    named_entity_words = sum([count for entity_type, count in text.values()])
    return named_entity_words


def calculate_entity_distribution(df: pd.DataFrame, dataset_name: str):
    # df['Named_Entity_Words'] = df['NER'].apply(count_words_distribution)
    # df['total_words'] = df['article_text'].apply(len)
    # df['Non_Entity_Words'] = df['total_words'] - df['Named_Entity_Words']
    # print(df.head())
    # total_words = df['total_words'].sum()
    # total_named_entity_words = df['Named_Entity_Words'].sum()
    # total_non_entity_words = total_words - total_named_entity_words
    # average_words_per_article = total_words / len(df)
    # percentage_named_entity_words = (total_named_entity_words / total_words) * 100
    #
    # print("Total Words: ", total_words)
    # print("Total Named Entity Words: ", total_named_entity_words)
    # print("Total Non-Named Entity Words: ", total_non_entity_words)
    # print("Average number of words per article/length:", average_words_per_article)
    # print("Percentage of words in articles that are named entities:", percentage_named_entity_words)
    #
    # labels = ['Named Entities', 'Non-Entities']
    # sizes = [total_named_entity_words, total_non_entity_words]
    # colors = ['sienna', 'khaki']
    #
    # plt.figure(figsize=(8, 6))
    # plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', explode=(0.2, 0), shadow=True,
    #         wedgeprops={'edgecolor': 'black'})
    # plt.title(f'Words Distribution between Named Entities and Non-Entities ({dataset_name})')
    # plt.axis('equal')
    # plt.savefig(f"Plots/entity_dist_{dataset_name}.png")
    # plt.show()

    # Calculate named entity words and non-entity words
    df['Named_Entity_Words'] = df['NER'].apply(count_words_distribution)
    df['total_words'] = df['article_text'].apply(len)
    df['Non_Entity_Words'] = df['total_words'] - df['Named_Entity_Words']

    print(df.head())


    total_words = df['total_words'].sum()
    total_named_entity_words = df['Named_Entity_Words'].sum()
    total_non_entity_words = total_words - total_named_entity_words


    average_words_per_article = total_words / len(df)
    percentage_named_entity_words = (total_named_entity_words / total_words) * 100


    print("Total Words: ", total_words)
    print("Total Named Entity Words: ", total_named_entity_words)
    print("Total Non-Named Entity Words: ", total_non_entity_words)
    print("Average number of words per article:", average_words_per_article)
    print("Percentage of words in articles that are named entities:", percentage_named_entity_words)

    # Data for visualization
    labels = ['Named Entities', 'Non-Entities']
    sizes = [total_named_entity_words, total_non_entity_words]

    # Create a pie chart using Plotly
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=sizes,
                hoverinfo='label+percent',
                textinfo='percent',
                textfont_size=16,
                marker=dict(colors=['sienna', 'khaki'], line=dict(color='black', width=1)),
                pull=[0.2, 0],  # Explode only the 'Named Entities' slice
            )
        ]
    )

    # Update layout with title
    fig.update_layout(
        title_text=f'Words Distribution between Named Entities and Non-Entities ({dataset_name})',
        title_x=0.5,  # Center the title
    )

    # Save the chart as an HTML file and show it
    fig.write_image(f"Plots/entity_dist_{dataset_name}.png")  # Save to file
    fig.show()  # Display the chart




def count_entity_types(text: str):
    entity_types = [entity_type for entity_type, _ in text.values()]
    return entity_types


def find_most_common_entity_types(df: pd.DataFrame, dataset_name: str):
    # entity_types = df['NER'].apply(count_entity_types).explode().value_counts()
    # colors = sns.color_palette('viridis', len(entity_types))
    #
    # plt.figure(figsize=(10, 6))
    # entity_types.plot(kind='bar', color=colors)
    # plt.title(f'Most Frequently Mentioned Named Entity Types ({dataset_name})')
    # plt.xlabel('Entity Type')
    # plt.ylabel('Frequency')
    # plt.xticks(rotation=45)
    # plt.savefig(f"Plots/entity_types_{dataset_name}.png")
    # plt.show()
    entity_types = df['NER'].apply(count_entity_types).explode().value_counts()
    entity_types_df = entity_types.reset_index()
    entity_types_df.columns = ['Entity Type', 'Frequency']

    # Create a bar chart using Plotly Express
    fig = px.bar(
        entity_types_df,
        x='Entity Type',
        y='Frequency',
        title=f'Most Frequently Mentioned Named Entity Types ({dataset_name})',
        labels={'Entity Type': 'Entity Type', 'Frequency': 'Frequency'},
        color='Frequency',
        color_continuous_scale='Viridis'  # Similar to sns.color_palette('viridis')
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis=dict(title="Entity Type", tickangle=45),  # Rotate x-axis labels
        yaxis_title="Frequency",
        title=dict(x=0.5),  # Center-align the title
        coloraxis_showscale=False,  # Hide the color scale legend
    )

    # Save the chart as an image and display it
    fig.write_image(f"Plots/entity_types_{dataset_name}.png")  # Save as PNG
    fig.show()  # Show the chart


def find_most_common_entities_per_type(df: pd.DataFrame, dataset_name: str, output_file: str):
    # entity_words_dict = defaultdict(Counter)
    #
    # for ner_dict in df['NER']:
    #     for word, (entity_type, count) in ner_dict.items():
    #         entity_words_dict[entity_type][word] += count
    #
    # colors = sns.color_palette('viridis', 15)
    # plt.figure(figsize=(16, 12))
    #
    # data = {'Entity Type': [], 'Word': [], 'Count': []}
    #
    # for i, (entity_type, words_counter) in enumerate(entity_words_dict.items(), 1):
    #     top_words = dict(words_counter.most_common(15))
    #     plt.subplot(2, 2, i)
    #     plt.bar(top_words.keys(), top_words.values(), color=colors)
    #     plt.title(f'Top 15 words for {entity_type} ({dataset_name})', fontsize=14, fontweight='bold')
    #     plt.xlabel('Word', fontsize=12)
    #     plt.ylabel('Frequency', fontsize=12)
    #     plt.xticks(rotation=60, fontsize=9, ha="right")
    #
    #     top_words = words_counter.most_common(40)
    #     for word, count in top_words:
    #         data['Entity Type'].append(entity_type)
    #         data['Word'].append(word)
    #         data['Count'].append(count)
    #
    # plt.tight_layout()
    # plt.savefig(f"Plots/most_common_entities_{dataset_name}.png")
    # plt.show()
    #
    # df_top_words = pd.DataFrame(data)
    # df_top_words.to_csv(output_file, index=False)
    #
    # print(f"Top words by entity type have been saved to {output_file}.")
    entity_words_dict = defaultdict(Counter)

    # Aggregate word counts by entity type
    for ner_dict in df['NER']:
        for word, (entity_type, count) in ner_dict.items():
            entity_words_dict[entity_type][word] += count

    # Prepare data for visualization
    data = {'Entity Type': [], 'Word': [], 'Count': []}
    for entity_type, words_counter in entity_words_dict.items():
        top_words = words_counter.most_common(40)  # Get the top 40 words for each entity type
        for word, count in top_words:
            data['Entity Type'].append(entity_type)
            data['Word'].append(word)
            data['Count'].append(count)

    df_top_words = pd.DataFrame(data)

    df_top_words.to_csv(output_file, index=False)
    print(f"Top words by entity type have been saved to {output_file}.")

    # Generate separate interactive visualizations for each entity type
    for entity_type in entity_words_dict:
        entity_data = df_top_words[df_top_words['Entity Type'] == entity_type].nlargest(15, 'Count')
        fig = px.bar(
            entity_data,
            x='Word',
            y='Count',
            title=f'Top 15 Words for {entity_type} ({dataset_name})',
            labels={'Word': 'Word', 'Count': 'Frequency'},
            color='Count',
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            xaxis=dict(title='Word', tickangle=60),
            yaxis_title='Frequency',
            title=dict(x=0.5),  # Center-align the title
            coloraxis_showscale=False,  # Hide color scale
        )

        fig.show()

    print(f"Visualizations for the top words per entity type have been displayed.")


def main():
    file_name = "../Data/gaza_textcontain_before_new_preprocessed.csv"
    dict_name = "dict_gaza.csv"
    ner_file = "Results/Gaza_before_with_NER.csv"
    dataset_name = "Gaza before conflict"
    entities_file = "Results/Gaza_before_top_40_entities.csv"
    df = perform_ner(file_name, dict_name, ner_file)
    calculate_entity_distribution(df, dataset_name)
    find_most_common_entity_types(df, dataset_name)
    find_most_common_entities_per_type(df, dataset_name, entities_file)


if __name__ == "__main__":
    main()
