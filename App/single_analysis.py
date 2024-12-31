import spacy
from spacy import displacy

# Load the spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en-core-web-sm")
    nlp = spacy.load("en_core_web_sm")

# Define the legend for entity categories
entity_legend = {
    "ORG": "Organizations: companies, agencies, institutions, etc.",
    "PERSON": "People (including fictional)",
    "GPE": "Geopolitical entities: countries, cities, states",
    "LOC": "Non-geopolitical locations: mountain ranges, bodies of water, etc.",
    "NORP": "Nationalities or religious or political groups",
    "EVENT": "Named events: hurricanes, battles, wars, sports events, etc."
}

def perform_ner_single_article(article_text: str):
    # Process the article text with the spacy model for visualization
    doc = nlp_spacy(article_text)

    # Visualize the named entities using displacy
    html = displacy.render(doc, style="ent", jupyter=False,
                           options={"ents": ["ORG", "PERSON", "GPE", "LOC", "NORP", "EVENT"]})

    # Add the legend to the HTML output
    legend_html = "<div class='legend'><h3>Entity Legend</h3><ul>"
    for entity, description in entity_legend.items():
        legend_html += f"<li><strong>{entity}</strong>: {description}</li>"
    legend_html += "</ul></div>"

    # Combine the legend and the NER visualization
    full_html = legend_html + html

    return full_html
