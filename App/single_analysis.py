import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import spacy
from spacy import displacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from NewsSentiment import TargetSentimentClassifier
from NewsSentiment.customexceptions import TargetNotFoundException, TooLongTextException
import re
import warnings
import nltk
from nltk import FreqDist, word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from colors import main_color,my_red,my_blue,my_gray,my_green,my_yellow

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Load the spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en-core-web-sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize the VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize the TSC model
tsc = TargetSentimentClassifier()

# Define the legend for entity categories
entity_legend = {
    "ORG": "Organizations: companies, agencies, institutions, etc.",
    "PERSON": "People (including fictional)",
    "GPE": "Geopolitical entities: countries, cities, states",
    "LOC": "Non-geopolitical locations: mountain ranges, bodies of water, etc.",
    "NORP": "Nationalities or religious or political groups",
    "EVENT": "Named events: hurricanes, battles, wars, sports events, etc."
}


def vader_sentiment(text: str) -> str:
    scores = vader_analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def tsc_sentiment(sentence: str, target: str) -> str:
    try:
        entity_start = sentence.find(target)
        entity_end = entity_start + len(target)
        left_context = sentence[:entity_start]
        right_context = sentence[entity_end:]
        sentiment_tsc = tsc.infer_from_text(left_context, target, right_context)
        return sentiment_tsc[0]['class_label'].lower()
    except (TooLongTextException, TargetNotFoundException):
        return 'unknown'


def analyse_single_article(article_text: str):
    positive_emoji = "emoji_1"
    negative_emoji = "emoji_2"
    neutral_emoji = "emoji_3"
    sentences = [sent.text.strip() for sent in nlp(article_text).sents]
    sentiment_sentences = []

    # Analyze sentiment for each sentence
    for i, sentence in enumerate(sentences):
        sentiment = vader_sentiment(sentence)
        sentiment_emoji = positive_emoji if sentiment == "positive" else negative_emoji if sentiment == "negative" else neutral_emoji
        sentences[i] = sentence + ' ' + sentiment_emoji
        sentiment_sentences.append((sentence, sentiment))

    modified_text = " ".join(sentences)
    # Process the article text with the spacy model for visualization
    doc = nlp(modified_text)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in entity_legend.keys()]

    # Analyze sentiment for each entity
    entity_sentiments = []
    for entity, label in entities:
        for sentence in sentences:
            if entity in sentence:
                sentiment = tsc_sentiment(sentence, entity)
                entity_sentiments.append((entity, label, sentiment))
                break

    # Visualize the named entities using displacy
    options = {"ents": list(entity_legend.keys())}
    html = displacy.render(doc, style="ent", jupyter=False, options=options)
    html = html.replace(
        "emoji_1",
        '<a href="https://img.icons8.com/ios-filled/50/00FF00/happy--v1.png" target="_blank">'
        '<img width="16" height="16" src="https://img.icons8.com/ios-filled/50/00FF00/happy--v1.png" '
        'alt="Positive sentiment" style="background-color: black; border-radius: 50%; padding: 0px;"/>'
        '<span class="tooltip"><b>Positive</b> sentence</span>'
        '</a><br>'
    )
    html = html.replace(
        "emoji_2",
        '<a href="https://img.icons8.com/ios-filled/50/FF0000/sad.png" target="_blank">'
        '<img width="16" height="16" src="https://img.icons8.com/ios-filled/50/FF0000/sad.png" '
        'alt="Negative sentiment" style="background-color: black; border-radius: 50%; padding: 0px;"/>'
        '<span class="tooltip"><b>Negative</b> sentence</span>'
        '</a><br>'
    )
    html = html.replace(
        "emoji_3",
        '<a href="https://img.icons8.com/ios-filled/50/FCC419/neutral-emoticon--v1.png" target="_blank">'
        '<img width="16" height="16" src="https://img.icons8.com/ios-filled/50/FCC419/neutral-emoticon--v1.png" '
        'alt="Neutral sentiment" style="background-color: black; border-radius: 50%; padding: 0px;"/>'
        '<span class="tooltip"><b>Neutral</b> sentence</span>'
        '</a><br>'
    )

    # Add the legend to the HTML output
    legend_html = "<div class='legend'><h3>Entity Legend</h3><ul>"
    for entity, description in entity_legend.items():
        legend_html += f"<li><strong>{entity}</strong>: {description}</li>"
    legend_html += "</ul></div>"
    legend_html += "<div class='legend'><h3>Sentiment Legend</h3><ul>"
    legend_html += f'<li><strong style="color: {my_green};">Green</strong>: Positive sentiment</li>'
    legend_html += f'<li><strong style="color: {my_yellow};">Gray</strong>: Neutral sentiment</li>'
    legend_html += f'<li><strong style="color: {my_red};">Red</strong>: Negative sentiment</li>'
    legend_html += "</ul></div>"

    # Combine the legend, NER visualization, and sentiment analysis results
    full_html = legend_html + html

    # Add inline styles for entity colors based on sentiment
    for entity, label, sentiment in entity_sentiments:
        color = my_green if sentiment == "positive" else my_red if sentiment == "negative" else my_yellow
        tooltip_value = f'<b>Value:</b> {entity}<br><b>Entity type:</b> {entity_legend[label]}<br><b>Sentiment:</b> {sentiment}'
        tooltip = f'><span class="tooltip">{tooltip_value}</span>'
        title = f'"<b>Value:</b> {entity}<br><b>Entity type:</b> {entity_legend[label]}<br><b>Sentiment:</b> {sentiment}">'
        pattern = fr'(<mark class="entity" style="background: )#[a-fA-F0-9]+(;[^>]*?>\s*{re.escape(entity)})'
        full_html = re.sub(pattern, lambda
            match: f'{match.group(1)}{color}{match.group(2).replace(">", tooltip, 1)}', full_html)

    entity_sentiments = pd.DataFrame(entity_sentiments, columns=["Entity", "Entity Type", "Sentiment"])
    sentiment_sentences = pd.DataFrame(sentiment_sentences, columns=["Sentence", "Sentiment"])

    return full_html, entity_sentiments, sentiment_sentences


def entity_types_plot_single(entity_sentiments: pd.DataFrame):
    entity_types = entity_sentiments["Entity Type"].value_counts()
    entity_types = entity_types.reindex(entity_legend.keys()).fillna(0)
    entity_types = entity_types.reset_index()
    entity_types.columns = ["Entity Type", "Frequency"]

    # Create a bar chart using Plotly Express
    fig = px.bar(
        entity_types,
        x='Entity Type',
        y='Frequency',
        title=f'Distribution of Named Entity Types',
        labels={'Entity Type': 'Entity Type', 'Frequency': 'Frequency'},
        color_discrete_sequence=[my_blue]  # Similar to sns.color_palette('viridis')
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis=dict(title="Entity Type", tickangle=45),  # Rotate x-axis labels
        yaxis_title="Frequency",
        title=dict(x=0.5),  # Center-align the title
        coloraxis_showscale=False,  # Hide the color scale legend
    )
    #temmplate
    fig.update_layout(template='plotly_white')

    return fig


def most_common_entities_plot_single(entity_sentiments: pd.DataFrame):
    entity_counts = entity_sentiments["Entity"].value_counts().head(15)
    entity_counts = entity_counts.reset_index()
    entity_counts.columns = ["Entity", "Frequency"]

    # Create a bar chart using Plotly Express
    fig = px.bar(
        entity_counts,
        y='Entity',
        x='Frequency',
        orientation='h',
        title=f'Most Common Named Entities',
        labels={'Entity': 'Entity', 'Frequency': 'Frequency'},
        color_discrete_sequence=[my_blue],
    )

    fig.update_layout(
        xaxis=dict(title='Frequency'),
        yaxis=dict(
            title='Entity',
            ticklabelposition='outside',
            ticks='outside',
            tickcolor='#fdfdfd',
            ticklen=10,
            automargin=True,
        ),
        # margin=dict(l=150),
        title=dict(x=0.5),  # Center-align the title
        coloraxis_showscale=False,  # Hide the color scale
    )
    fig.update_layout(template='plotly_white')

    return fig


def sentiment_dist_plot_single(entity_sentiments: pd.DataFrame, sentiment_sentences: pd.DataFrame):
    sentence_sentiment = sentiment_sentences["Sentiment"].value_counts()
    entity_sentiment = entity_sentiments["Sentiment"].value_counts()
    sentiment_comparison = pd.DataFrame({
        'Positive': [sentence_sentiment.get('positive', 0), entity_sentiment.get('positive', 0)],
        'Neutral': [sentence_sentiment.get('neutral', 0), entity_sentiment.get('neutral', 0)],
        'Negative': [sentence_sentiment.get('negative', 0), entity_sentiment.get('negative', 0)]
    }, index=['Sentence-based', 'Entity-based'])

    fig = go.Figure()
    sentiments = ['Positive', 'Neutral', 'Negative']
    colors = [my_green, my_yellow, my_red]

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
        title=f'Comparison of Sentiment Distribution Based on Sentences and Named Entities',
        xaxis_title='Type of Sentiment Analysis',
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
        legend=dict(traceorder='normal'),
        height=600,
        width=800
    )
    fig.update_layout(template='plotly_white')

    return fig


def most_common_words_plot_single(sentiment_sentences: pd.DataFrame, N=100, article: str = ''):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    fdist = FreqDist(
        [word for word in word_tokenize(' '.join(sentiment_sentences["Sentence"])) if
         word.lower() not in stop_words and word.isalpha()])
    wordcloud = WordCloud(width=800, height=400, background_color='white', color_func=lambda *args, **kwargs: my_blue).generate_from_frequencies(
        dict(fdist.most_common(N)))
    plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')
    plt.axis('off')
    if article != '':
        plt.title(f'Top {N} Most Common Words from {article}')
    else:
        plt.title(f'Top {N} Most Common Words')
    plt.tight_layout()
    return plt.gcf()
