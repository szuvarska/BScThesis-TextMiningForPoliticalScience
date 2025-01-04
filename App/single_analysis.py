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


# def perform_ner_single_article(article_text: str):
#     # Process the article text with the spacy model for visualization
#     doc = nlp_spacy(article_text)
#
#     # Visualize the named entities using displacy
#     html = displacy.render(doc, style="ent", jupyter=False,
#                            options={"ents": ["ORG", "PERSON", "GPE", "LOC", "NORP", "EVENT"]})
#
#     # Add the legend to the HTML output
#     legend_html = "<div class='legend'><h3>Entity Legend</h3><ul>"
#     for entity, description in entity_legend.items():
#         legend_html += f"<li><strong>{entity}</strong>: {description}</li>"
#     legend_html += "</ul></div>"
#
#     # Combine the legend and the NER visualization
#     full_html = legend_html + html
#
#     return full_html


def analyse_single_article(article_text: str):
    # Process the article text with the spacy model for visualization
    doc = nlp(article_text)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in entity_legend.keys()]

    # Split article into sentences
    sentences = [sent.text for sent in doc.sents]

    # Analyze sentiment for each entity
    entity_sentiments = []
    for entity, label in entities:
        for sentence in sentences:
            if entity in sentence:
                sentiment = tsc_sentiment(sentence, entity)
                entity_sentiments.append((entity, label, sentiment))
                break

    # Analyze sentiment for each sentence
    sentence_sentiments = []
    for sentence in sentences:
        sentiment = vader_sentiment(sentence)
        sentence_sentiments.append((sentence, sentiment))

    # Visualize the named entities using displacy
    options = {"ents": list(entity_legend.keys())}
    html = displacy.render(doc, style="ent", jupyter=False, options=options)

    # Add the legend to the HTML output
    legend_html = "<div class='legend'><h3>Entity Legend</h3><ul>"
    for entity, description in entity_legend.items():
        legend_html += f"<li><strong>{entity}</strong>: {description}</li>"
    legend_html += "</ul></div>"

    # Add sentiment analysis results to the HTML output
    sentiment_html = "<div class='sentiment-analysis'><h3>Sentiment Analysis</h3><ul>"
    for entity, label, sentiment in entity_sentiments:
        sentiment_html += f"<li><strong>{entity} ({label})</strong>: {sentiment}</li>"
    sentiment_html += "</ul><h4>Sentence Sentiments</h4><ul>"
    for sentence, sentiment in sentence_sentiments:
        sentiment_html += f"<li>{sentence}: {sentiment}</li>"
    sentiment_html += "</ul></div>"

    # Combine the legend, NER visualization, and sentiment analysis results
    full_html = legend_html + html + sentiment_html

    # Add inline styles for entity colors based on sentiment
    for entity, label, sentiment in entity_sentiments:
        color = "green" if sentiment == "positive" else "red" if sentiment == "negative" else "yellow"
        tooltip_value = f'<b>Value:</b> {entity}<br><b>Entity type:</b> {entity_legend[label]}<br><b>Sentiment:</b> {sentiment}'
        # tooltip_value = sentiment
        tooltip = f'><span class="tooltip">{tooltip_value}</span>'
        title = f'"<b>Value:</b> {entity}<br><b>Entity type:</b> {entity_legend[label]}<br><b>Sentiment:</b> {sentiment}">'
        pattern = fr'(<mark class="entity" style="background: )#[a-fA-F0-9]+(;[^>]*?>\s*{re.escape(entity)})'
        full_html = re.sub(pattern, lambda
            match: f'{match.group(1)}{color}{match.group(2).replace(">", tooltip, 1)}', full_html)
    return full_html


if __name__ == '__main__':
    article_text = pd.read_csv("../Preparations/Data_for_EDA/df_gaza_after.csv")['article_text'][4]
    html_output = analyse_single_article(article_text)
    with open("article_analysis.html", "w") as file:
        file.write(html_output)
