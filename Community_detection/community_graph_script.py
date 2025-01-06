import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from community import community_louvain
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.colors as mcolors
from colors import main_color,my_red,my_blue,my_gray,my_green,my_yellow


#from Sentiment.sentiment_script import vader_sentiment

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



def plot_community_graph(df_ner: pd.DataFrame, df_entities: pd.DataFrame, suptitle: str, title: str,
                         weight_threshold=10, nodes_displayed=20, layout='spring', edge='std'):
    df_entities.sort_values(by='Count', ascending=False, inplace=True)
    entities = df_entities.Word.tolist()
    co_occurrence = defaultdict(int)
    co_sentiment = defaultdict(int)

    # iterate through the articles and process with SpaCy to get sentences
    # for article in tqdm(df_ner['article_text']):
    #     sentences = article.split('.')
    #     # iterate through each sentence in the article
    #     for sentence in sentences:
    #         present_entities = [entity for entity in entities if entity in sentence]
    #         vader_sentiment_score = vader_sentiment(sentence)
    #         # print(present_entities)
    #         for i in range(len(present_entities)):
    #             for j in range(i + 1, len(present_entities)):
    #                 co_occurrence[(present_entities[i], present_entities[j])] += 1
    #                 if vader_sentiment_score == 'positive':
    #                     co_sentiment[(present_entities[i], present_entities[j])] += 1
    #                 elif vader_sentiment_score == 'negative':
    #                     co_sentiment[(present_entities[i], present_entities[j])] -= 1

    for sentence, sentiment in tqdm(zip(df_ner['article_text'], df_ner['sentiment'])):
        present_entities = [entity for entity in entities if entity in str(sentence)]
        for i in range(len(present_entities)):
            for j in range(i + 1, len(present_entities)):
                co_occurrence[(present_entities[i], present_entities[j])] += 1
                if sentiment == 'positive':
                    co_sentiment[(present_entities[i], present_entities[j])] += 1
                elif sentiment == 'negative':
                    co_sentiment[(present_entities[i], present_entities[j])] -= 1



    #standarize sentiment
    for (entity1, entity2), sentiment in co_sentiment.items():
        co_sentiment[(entity1, entity2)] = sentiment / co_occurrence[(entity1, entity2)]

    # creata a graph
    G = nx.Graph()
    for entity in entities:
        G.add_node(entity)

    # add edges with weights to the graph, filtering by the threshold
    selected_edges = []
    for (entity1, entity2), weight in co_occurrence.items():
        if weight >= weight_threshold:
            G.add_edge(entity1, entity2, weight=weight)
            selected_edges.append((entity1, entity2, weight))

    # calculate node strength -- the sum of the weights of edges connected to them
    node_strength = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes}

    # select top displayed nodes
    nodes_above_top = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[nodes_displayed:]
    G.remove_nodes_from([node for node, _ in nodes_above_top])

    # apply the Louvain method for community detection
    partition = community_louvain.best_partition(G, weight='weight')
    for node, community in partition.items():
        G.nodes[node]['community'] = community

    #add sentiment to the edges
    for (entity1, entity2), sentiment in co_sentiment.items():
        if entity1 in G.nodes and entity2 in G.nodes and G.has_edge(entity1, entity2):
            G[entity1][entity2]['sentiment'] = sentiment

    # draw the graph
    plt.figure(figsize=(12, 9))
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'arf':
        pos = nx.arf_layout(G)

    # get the community color map
    communities = set(partition.values())
    colors = [plt.cm.rainbow(i / len(communities)) for i in range(len(communities))]
    node_colors = [colors[partition[node]] for node in G.nodes]

    # node sizes with standarized node strength
    max_strength = max(node_strength.values())
    # normalize node sizes based on their strength
    node_sizes = [node_strength[node] / max_strength * 1500 for node in G.nodes]

    # edges with normalized weights
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    max_weight = max(weights)
    min_width = 0.2
    std_treshold = 0

    mean_weight = sum(weights) / len(weights)
    std_weight = (sum((weight - mean_weight) ** 2 for weight in weights) / len(weights)) ** 0.5

    if edge == 'std':
        weights_to_plot = [(weight / max_weight * 10) for weight in weights]  # standarize weights
    else:
        weights_to_plot = [((weight - mean_weight) / std_weight) for weight in weights]  # normalize weights

    # draw nodes with community colors and egdes with weights
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    #add efges with sentiment as color
    edge_colors = []
    for edge in edges:
        if len(edge[2]) < 2:
            edge_colors.append(0)
        else:
            edge_colors.append(edge[2]['sentiment'])

    # edge_colors = [edge[2]['sentiment'] if edge[2]['sentiment'] is not None else 0 for edge in edges]
    my_cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", [my_red, my_yellow, my_green])
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights_to_plot, edge_color=edge_colors, edge_cmap=my_cmap)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    #add legend for the sentiment
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, label='Sentiment')


    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    # title
    plt.suptitle(suptitle, fontsize=20)
    plt.title(title, fontsize=15)
    return plt
