import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the spacy model
nlp_spacy = spacy.load("en_core_web_sm")

# Load pretrained model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
nlp_transformers = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=-1)


def perform_ner_single_article(article_text: str):
    # Process the article text with the transformers NER pipeline
    ner_results = nlp_transformers(article_text)

    # Process the article text with the spacy model for visualization
    doc = nlp_spacy(article_text)

    # Visualize the named entities using displacy
    html = displacy.render(doc, style="ent", jupyter=False)

    return html
