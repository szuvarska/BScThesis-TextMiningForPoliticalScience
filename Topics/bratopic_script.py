import pandas as pd
from bertopic import BERTopic
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech

def perform_bratopic(df: pd.DataFrame):
    sentences_tokenized = df['article_text']

    # for text, date in zip(df['text'], df['published_time']):
    #     sentences_tokenized.extend(word_tokenize(str(text)))

    sentences_tokenized = [re.sub(r"[^a-zA-Z]+", " ", str(s)) for s in sentences_tokenized]
    print(len(sentences_tokenized))
    #prapare models
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(sentences_tokenized, show_progress_bar=True)
    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english")
    representation_model = MaximalMarginalRelevance(diversity=0.2)

    # Create BERTopic model
    topic_model = BERTopic(verbose=True, vectorizer_model=vectorizer_model, hdbscan_model=hdbscan_model,
                           embedding_model=embedding_model, representation_model=representation_model)
    topics, probs = topic_model.fit_transform(sentences_tokenized)

    topic_detailes_df = topic_model.get_topic_info()

    return topics, probs, topic_detailes_df, topic_model


