import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

def celan_articles(articles: pd.DataFrame) -> pd.DataFrame:
    articles = articles.drop(articles[articles['published_time'] == 'Published time not found'].index)
    articles['published_time'] = articles['published_time'].str[11:]
    articles['published_time'] = articles['published_time'].str[:12]
    articles['published_time'] = pd.to_datetime(articles['published_time'], format='%b %d, %Y')
    articles['author'] = articles['author'].apply(lambda x: x[3:] if x.startswith("By ") else x)
    articles['article_text'] = articles.apply(lambda x: x['article_text'].replace(str(x['picture_description']), ''), axis=1)
    return articles

def merge_articles_authors(articles: pd.DataFrame, authors: pd.DataFrame) -> pd.DataFrame:
    authors = authors.drop_duplicates()
    df = articles.join(authors.set_index('author_name'), on='author', how='left')
    df.replace(pd.NA, 'Author details not found', inplace=True)
    return df

