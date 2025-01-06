import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import csv
from colors import main_color,my_red,my_blue,my_gray,my_green,my_yellow

project_dir = os.path.dirname(os.path.abspath(__file__))
nltk_data_path = os.path.join(project_dir, 'App/nltk_data')
os.environ['NLTK_DATA'] = nltk_data_path
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')

stop_words = stopwords.words('english')
for w in stopwords.words('english'):
    stop_words.append(w.capitalize())
stop_words = set(stop_words)


def load_pos_dict(file_path='Preparations/Data_for_EDA/pos_dict.csv', key='abbr'):
    pos_dict = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        if key == 'abbr':
            pos_dict = {rows[0]: rows[1] for rows in reader}
        elif key == 'name':
            pos_dict = {rows[1]: rows[0] for rows in reader}
    return pos_dict


def perpare_df_for_eda(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    df['sentence_count'] = df['article_text'].apply(lambda x: len(x.split('.')) - 1)
    df['word_count'] = df['article_text'].apply(lambda x: len([word for word in word_tokenize(x) if word.isalpha()]))
    nltk.download('averaged_perceptron_tagger')
    df['pos_count'] = df['article_text'].apply(lambda x: FreqDist(
        [tag for word, tag in nltk.pos_tag(word_tokenize(x)) if word.isalpha() and word not in stop_words]))

    df_pos = pd.DataFrame(df['pos_count'].tolist())
    df_pos.fillna(0, inplace=True)
    df_pos['sentence_count'] = df['sentence_count']
    df_pos['word_count'] = df['word_count']

    return df, df_pos


def plot_word_count_distribution(df: pd.DataFrame, df_name: str):
    # plt.figure(figsize=(12, 10))
    # sns.histplot(df.loc[(df['article_category_one'] != "PHOTO") & (df['word_count'] < 2000), 'word_count'], bins=25, kde=True)
    # #plt.xlim(0, 2000)
    # # mean line
    # plt.axvline(df['word_count'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
    # plt.legend()
    # plt.title(f'{df_name} word count distribution')
    # plt.show()
    #plt.savefig(f"EDA_plots/{df_name}_word_count_dist_.png")
    # Filtr danych
    filtered_data = df.loc[(df['article_category_one'] != "PHOTO") & (df['word_count'] < 2000), 'word_count']

    # Histogram z Plotly Express
    fig = px.histogram(
        filtered_data,
        nbins=25,
        title=f'Word Count Distribution - {df_name}',
        labels={'value': 'Word Count'},
        height=600,
        width=800,
        color_discrete_sequence=[my_blue]
    )

    # Dodanie linii średniej
    mean_value = filtered_data.mean()
    fig.add_shape(
        type="line",
        x0=mean_value, x1=mean_value,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color=my_red, dash="dash"),
        name="Mean",
    )

    # Dodanie legendy dla średniej
    fig.add_trace(go.Scatter(
        x=[mean_value],
        y=[0],
        mode="markers",
        marker=dict(color=my_red),
        name="Mean"
    ))

    # Aktualizacja układu wykresu
    fig.update_layout(
        xaxis_title="Word Count",
        yaxis_title="Frequency",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )

    )
    #add theme
    fig.update_layout(
        template="plotly_white"
    )


    return fig


def sentance_count_distribution(df: pd.DataFrame, df_name: str):
    # plt.figure(figsize=(12, 10))
    # sns.histplot(df.loc[(df['article_category_one'] != "PHOTO") & (df['word_count'] < 2000), 'sentence_count'], bins=20, kde=True)
    # #plt.xlim(0, 100)
    # # mean line
    # plt.axvline(df['sentence_count'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
    # plt.legend()
    # plt.title(f'{df_name} sentence count distribution')
    # plt.show()
    #plt.savefig(f"EDA_plots/{df_name}_sentace_count_dist_.png")

    # Filtr danych
    filtered_data = df.loc[(df['article_category_one'] != "PHOTO") & (df['word_count'] < 2000), 'sentence_count']

    # Histogram z Plotly Express
    fig = px.histogram(
        filtered_data,
        nbins=20,
        title=f'Sentence Count Distribution - {df_name}',
        labels={'value': 'Sentence Count'},
        height=600,
        width=800,
        color_discrete_sequence=[my_blue]
    )

    # Obliczenie średniej
    mean_value = filtered_data.mean()

    # Dodanie linii średniej
    fig.add_shape(
        type="line",
        x0=mean_value, x1=mean_value,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color=my_red, dash="dash"),
        name="Mean"
    )

    # Dodanie legendy dla średniej
    fig.add_trace(go.Scatter(
        x=[mean_value],
        y=[0],
        mode="markers",
        marker=dict(color=my_red),
        name="Mean"
    ))

    # Aktualizacja układu wykresu
    fig.update_layout(
        xaxis_title="Sentence Count",
        yaxis_title="Frequency",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    #add theme
    fig.update_layout(
        template="plotly_white"
    )

    return fig


def plot_top_N_common_words(df: pd.DataFrame, df_name: str, N=100):
    stop_words = set(stopwords.words('english'))
    fdist = FreqDist(
        [word for word in word_tokenize(' '.join(df['article_text'])) if
         word.lower() not in stop_words and word.isalpha()])
    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=lambda *args, **kwargs: my_blue).generate_from_frequencies(
        dict(fdist.most_common(N)))
    # plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Top {N} Most Common Words - {df_name}')
    # plt.show()
    plt.tight_layout()
    return plt.gcf()


def plot_top_N_common_pos(df_pos: pd.DataFrame, df_name: str, N=10):
    # pos_dict = {
    #     "NN": "Common Singular Nouns",
    #     "NNS": "Common Plural Nouns",
    #     "NNP": "Proper Singular Nouns",
    #     "NNPS": "Proper Plural Nouns",
    #     "JJ": "Adjectives in Positive Form",
    #     "JJR": "Adjectives in Comparative Form",
    #     "JJS": "Adjectives in Superlative Form",
    #     "VB": "Verbs in Base Form",
    #     "VBD": "Verbs in Past Tense",
    #     "VBG": "Verbs in Present Participle",
    #     "VBN": "Verbs in Past Participle",
    #     "VBP": "Verbs in Non-3rd Person Singular Present Form",
    #     "VBZ": "Verbs in 3rd Person Singular Present Form",
    #     "RB": "Adverbs in Positive Form",
    #     "RBR": "Adverbs in Comparative Form",
    #     "RBS": "Adverbs in Superlative Form",
    #     "WDT": "Wh-determiners",
    #     "WP": "Wh-pronouns",
    #     "WRB": "Wh-adverbs",
    #     "IN": "Prepositions",
    #     "CC": "Conjunctions",
    #     "DT": "Determiners",
    #     "EX": "Existential There",
    #     "FW": "Foreign Words",
    #     "LS": "List Item Marker",
    #     "MD": "Modal",
    #     "CD": "Cardinal Numbers",
    #     "POS": "Possessive Ending",
    #     "PRP": "Personal Pronouns",
    #     "PRP$": "Possessive Pronouns",
    #     "RP": "Particles",
    #     "TO": "To",
    #     "UH": "Interjection",
    #     "SYM": "Symbol",
    #     "WP$": "Possessive Wh-pronouns",
    #     "PDT": "Predeterminers",
    #     '$': "Dollar Sign",
    #     '``': "Curly Quotation Mark",
    #     "''": "Quotation Mark",
    # }

    pos_dict = load_pos_dict(key='abbr')

    # Calculate the sum for each POS tag and sort them
    pos_counts = df_pos.drop(columns=['sentence_count', 'word_count']).sum().sort_values(ascending=False).head(N)

    # Convert the Series to a DataFrame and reset the index
    pos_counts = pos_counts.reset_index()

    # Rename the columns
    pos_counts.columns = ['POS', 'Count']

    # Map the short POS tags to their longer names
    pos_counts['POS'] = pos_counts['POS'].map(pos_dict)

    # Replace NaN values with 0
    pos_counts['Count'] = pos_counts['Count'].fillna(0)

    # Ensure there are no infinite values
    pos_counts['Count'] = pos_counts['Count'].replace([np.inf, -np.inf], 0)

    # Create a bar plot using Plotly
    fig = px.bar(
        pos_counts,
        x='Count',
        y='POS',
        orientation='h',
        title=f'Top {N} Most Common Part of Speech - {df_name}',
        labels={'Count': 'Count', 'POS': 'Part of Speech'},
        color_discrete_sequence=[my_blue]
    )

    # Update the layout of the plot
    fig.update_layout(
        xaxis=dict(title="Count", tickangle=45),
        yaxis=dict(
            title='Part of Speech',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        title=dict(x=0.5),  # Center the title
        showlegend=False,  # Remove the legend as it is not needed
        coloraxis_showscale=False,
    )

    #add theme
    fig.update_layout(
        template="plotly_white"
    )

    return fig


def plot_pos_wordclouds(df: pd.DataFrame, df_name: str, N=100):
    #if we want to comate eg diffrents categories, jutro put a piltered df as input

    NN = FreqDist([word for word, tag in nltk.pos_tag(word_tokenize(' '.join(df['article_text']))) if
                   tag == "NN" and word.isalpha() and word not in stop_words])
    NNP = FreqDist([word for word, tag in nltk.pos_tag(word_tokenize(' '.join(df['article_text']))) if
                    tag == "NNP" and word.isalpha() and word not in stop_words])
    JJ = FreqDist([word for word, tag in nltk.pos_tag(word_tokenize(' '.join(df['article_text']))) if
                   tag == "JJ" and word.isalpha() and word not in stop_words])
    NNS = FreqDist([word for word, tag in nltk.pos_tag(word_tokenize(' '.join(df['article_text']))) if
                    tag == "NNS" and word.isalpha() and word not in stop_words])
    VB = FreqDist([word for word, tag in nltk.pos_tag(word_tokenize(' '.join(df['article_text']))) if
                   tag == "VB" and word.isalpha() and word not in stop_words])
    RB = FreqDist([word for word, tag in nltk.pos_tag(word_tokenize(' '.join(df['article_text']))) if
                   tag == "RB" and word.isalpha() and word not in stop_words])

    plt.figure(figsize=(6, 36))
    plt.subplot(6, 1, 1)
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(
        dict(NN.most_common(100)))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{df_name} Top {N} Most Common Nouns')

    plt.subplot(6, 1, 2)
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(
        dict(NNP.most_common(100)))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{df_name} Top {N} Most Common Nouns Proper')

    plt.subplot(6, 1, 3)
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(
        dict(JJ.most_common(100)))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{df_name} Top {N} Most Common Adjectives')

    plt.subplot(6, 1, 4)
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(
        dict(NNS.most_common(100)))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{df_name} Top {N}Most Common Nouns Plural')

    plt.subplot(6, 1, 5)
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(
        dict(VB.most_common(100)))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{df_name} Top {N} Most Common Verbs')

    plt.subplot(6, 1, 6)
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(
        dict(RB.most_common(100)))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{df_name} Top {N} Most Common Adverbs')
    return plt


def plot_pos_wordclouds_for_shiny(df: pd.DataFrame, df_name: str, N=100, pos="Common Singular Nouns"):
    # pos_dict = {
    #     "Common Singular Nouns": "NN",
    #     "Common Plural Nouns": "NNS",
    #     "Proper Singular Nouns": "NNP",
    #     "Proper Plural Nouns": "NNPS",
    #     "Adjectives in Positive Form": "JJ",
    #     "Adjectives in Comparative Form": "JJR",
    #     "Adjectives in Superlative Form": "JJS",
    #     "Verbs in Base Form": "VB",
    #     "Verbs in Past Tense": "VBD",
    #     "Verbs in Present Participle": "VBG",
    #     "Verbs in Past Participle": "VBN",
    #     "Verbs in Non-3rd Person Singular Present Form": "VBP",
    #     "Verbs in 3rd Person Singular Present Form": "VBZ",
    #     "Adverbs in Positive Form": "RB",
    #     "Adverbs in Comparative Form": "RBR",
    #     "Adverbs in Superlative Form": "RBS",
    #     "Wh-determiners": "WDT",
    #     "Wh-pronouns": "WP",
    #     "Wh-adverbs": "WRB",
    #     "Prepositions": "IN",
    #     "Conjunctions": "CC",
    #     "Determiners": "DT",
    #     "Existential There": "EX",
    #     "Foreign Words": "FW",
    #     "List Item Marker": "LS",
    #     "Modal": "MD",
    #     "Cardinal Numbers": "CD",
    #     "Possessive Ending": "POS",
    #     "Personal Pronouns": "PRP",
    #     "Possessive Pronouns": "PRP$",
    #     "Particle": "RP",
    #     "To": "TO",
    #     "Interjection": "UH",
    #     "Symbol": "SYM"
    # }

    pos_dict = load_pos_dict(key='name')

    pos_short = pos_dict[pos]
    tagged_data = nltk.pos_tag(word_tokenize(' '.join(df['article_text'])))
    pos_data = FreqDist([word for word, tag in tagged_data if
                         tag == pos_short and word.isalpha() and word not in stop_words])
    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=lambda *args, **kwargs: my_blue).generate_from_frequencies(
        dict(pos_data.most_common(N)))
    # plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Top {N} Most Common {pos} - {df_name}')
    plt.tight_layout()
    return plt.gcf()
