import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from community import community_louvain


def plot_community_graph(df_ner: pd.DataFrame, df_entities: pd.DataFrame, suptitle: str, title: str,
                         weight_threshold=10, nodes_displayed=20, layout='spring', edge='std'):
    df_entities.sort_values(by='Count', ascending=False, inplace=True)
    entities = df_entities.Word.tolist()
    co_occurrence = defaultdict(int)

    # iterate through the articles and process with SpaCy to get sentences
    for article in df_ner['article_text']:
        sentences = article.split('.')
        # iterate through each sentence in the article
        for sentence in sentences:
            present_entities = [entity for entity in entities if entity in sentence]
            # print(present_entities)
            for i in range(len(present_entities)):
                for j in range(i + 1, len(present_entities)):
                    co_occurrence[(present_entities[i], present_entities[j])] += 1

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

    # draw the graph
    plt.figure(figsize=(15, 15))
    if layout == 'spring':
        pos = nx.spring_layout(G)
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
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, cmap=plt.cm.rainbow, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights_to_plot, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)

    # title
    plt.suptitle(suptitle, fontsize=20)
    plt.title(title, fontsize=15)
    plt.show()
