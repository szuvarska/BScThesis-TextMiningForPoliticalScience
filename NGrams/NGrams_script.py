import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
import nltk
from wordcloud import WordCloud
import networkx as nx
from colors import main_color,my_red,my_blue,my_gray,my_green,my_yellow


def concordance(df: pd.DataFrame, filter: list, ngram_number: int):
    # n >= 2
    n = ngram_number
    # tokenization
    df['tokenized'] = df['article_text'].apply(str).apply(word_tokenize)
    # remove punctuation
    df['tokenized'] = df['tokenized'].apply(lambda x: [i for i in x if i.isalnum()])
    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    df['tokenized'] = df['tokenized'].apply(lambda x: [i for i in x if i not in stopwords])

    # ngrams with count
    ngrams_list = []
    for i in range(len(df)):
        ngrams_i = ngrams(df['tokenized'].iloc[i], n)
        ngrams_list += ngrams_i
    ngrams_counter = Counter(ngrams_list)

    # corcordance making and filtering
    ngrams_counted_list = []
    for ngram, count in ngrams_counter.items():
        ngram_center = ngram[int(n / 2)]
        ngrams_str = ' '.join(ngram)
        ngrams_counted_list.append((ngrams_str, count, ngram_center))
    ngrams_df = pd.DataFrame(ngrams_counted_list, columns=['ngram', 'count', 'center'])
    ngrams_df = ngrams_df[ngrams_df['center'].isin(filter)]

    # formating
    ngrams_df_table = ngrams_df.sort_values(by='count', ascending=False).head(20)
    max_words_len = [0, 0, 0]
    lefts = []
    centers = []
    rights = []

    for i in range(len(ngrams_df_table)):
        words = ngrams_df_table['ngram'].iloc[i].split()
        words_len = [0, 0, 0]
        center = str(ngrams_df_table['center'].iloc[i])
        left = ''
        right = ''
        j = 0
        while words[j] != center:
            words_len[0] += len(words[j]) + 1
            left += words[j] + ' '
            j += 1
        words_len[1] = len(words[j]) + 1
        center = words[j] + ' '
        j += 1
        while j < len(words):
            words_len[2] += len(words[j]) + 1
            right += words[j] + ' '
            j += 1
        right = right[:-1]
        # print(right)
        for j in range(3):
            if words_len[j] > max_words_len[j]:
                max_words_len[j] = words_len[j]

        lefts.append(left)
        centers.append(center)
        rights.append(right)

    formated_ngrams = []
    formeted_len = []
    for i in range(len(ngrams_df_table)):
        formated_ngram = ''
        formated_ngram += lefts[i].rjust(max_words_len[0], " ")
        formated_ngram += centers[i].center(max_words_len[1], " ")
        formated_ngram += rights[i].ljust(max_words_len[2] - 1, " ")
        formated_ngrams.append(formated_ngram)
        formeted_len.append(len(formated_ngram))

    ngrams_df_table['formated_ngram'] = formated_ngrams
    ngrams_df_table['formated_len'] = formeted_len
    ngrams_df_table['lefts'] = lefts
    ngrams_df_table['rights'] = rights
    ngrams_df_table = ngrams_df_table.reset_index(drop=True)

    return ngrams_df_table


def visualize_bigrams(df: pd.DataFrame, top_n: int = 10, dataset_name: str = ''):
    df['tokenized'] = df['article_text'].apply(str).apply(word_tokenize)
    # remove punctuation
    df['tokenized'] = df['tokenized'].apply(lambda x: [i for i in x if i.isalnum()])
    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    df['tokenized'] = df['tokenized'].apply(lambda x: [i for i in x if i not in stopwords])

    # ngrams with count
    n = 2
    ngrams_list = []

    for i in range(len(df)):
        ngrams_i = ngrams(df['tokenized'].iloc[i], n)
        ngrams_list += ngrams_i
    ngrams_counter = Counter(ngrams_list)
    word_0 = []
    word_1 = []
    ngrams_counted_list = []
    for ngram, count in ngrams_counter.items():
        word_0.append(ngram[0])
        word_1.append(ngram[1])
        ngrams_counted_list.append(count)
    ngrams_df = pd.DataFrame({'word_0': word_0, 'word_1': word_1, 'count': ngrams_counted_list})
    ngrams_df = ngrams_df.sort_values(by='count', ascending=False)

    ngrams_df = ngrams_df[ngrams_df['count'] > 15]
    ngrams_df = ngrams_df[ngrams_df['word_0'] != 'The']
    ngrams_df = ngrams_df[ngrams_df['word_1'] != 'Times']
    top_ngrams = ngrams_df.head(20)

    word_freq = pd.concat([top_ngrams['word_0'], top_ngrams['word_1']]).value_counts()
    top_words = set(word_freq.index)
    filtered_df_0 = ngrams_df[ngrams_df['word_0'].isin(top_words) | ngrams_df['word_1'].isin(top_words)]

    word_freq2 = pd.concat([filtered_df_0['word_0'], filtered_df_0['word_1']]).value_counts()
    top_words2 = set(word_freq2.nlargest(top_n).index)
    filtered_df = filtered_df_0[filtered_df_0['word_0'].isin(top_words2) | filtered_df_0['word_1'].isin(top_words2)]
    filtered_df = filtered_df.sort_values(by='count', ascending=False).head(50)

    filtered_df['count'] = filtered_df['count'] / filtered_df['count'].max()

    # crating a graph
    G = nx.DiGraph()
    for _, row in filtered_df.iterrows():
        G.add_edge(row['word_0'], row['word_1'], weight=(row['count']) * 10)

    # Compute node size based on degree (incoming + outgoing edges)
    node_size = {node: (G.in_degree(node, weight='weight') + G.out_degree(node, weight='weight')) * 50 for node in
                 G.nodes}
    if node_size == {} or max(node_size.values()) == 0:
        return None

    # Plot the graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=0.9)  # Adjust k
    nx.draw_networkx_nodes(G, pos, node_size=[node_size[n] for n in G.nodes], node_color=main_color,
                           edgecolors=None)
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='-|>', edge_color='gray', alpha=0.4,
                                   width=[G[u][v]['weight'] for u, v in G.edges], connectionstyle='arc3,rad=0.05',
                                   arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add title and subtitle
    plt.title(f"Most common words in bigrams graph - {dataset_name} \n\n\n", fontsize=25, fontweight='bold')
    plt.suptitle("Edges represent bigram connections strength. Arrows indicate the order of words in the bigram",
                 fontsize=20, fontweight='regular', y=0.92)

    return plt
