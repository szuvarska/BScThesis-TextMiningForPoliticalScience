import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

stop_words = stopwords.words('english')
for w in stopwords.words('english'):
    stop_words.append(w.capitalize())
stop_words = set(stop_words)

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

def plot_word_cout_distribution(df: pd.DataFrame, df_name: str):
    plt.figure(figsize=(12, 10))
    sns.histplot(df.loc[(df['article_category_one'] != "PHOTO") & (df['word_count'] < 2000), 'word_count'], bins=25, kde=True)
    #plt.xlim(0, 2000)
    # mean line
    plt.axvline(df['word_count'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()
    plt.title(f'{df_name} word count distribution')
    plt.show()
    #plt.savefig(f"EDA_plots/{df_name}_word_count_dist_.png")

def sentance_count_distribution(df: pd.DataFrame, df_name: str):
    plt.figure(figsize=(12, 10))
    sns.histplot(df.loc[(df['article_category_one'] != "PHOTO") & (df['word_count'] < 2000), 'sentence_count'], bins=20, kde=True)
    #plt.xlim(0, 100)
    # mean line
    plt.axvline(df['sentence_count'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.legend()
    plt.title(f'{df_name} sentence count distribution')
    plt.show()
    #plt.savefig(f"EDA_plots/{df_name}_sentace_count_dist_.png")


def plot_top_N_common_words(df: pd.DataFrame, df_name: str, N = 100):
    stop_words = set(stopwords.words('english'))
    fdist = FreqDist(
        [word for word in word_tokenize(' '.join(df['article_text'])) if word.lower() not in stop_words and word.isalpha()])
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate_from_frequencies(
        dict(fdist.most_common(N)))
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'{df_name} Top {N} Most Common Words')
    plt.show()

def plot_top_N_common_pos(df_pos: pd.DataFrame, df_name: str, N = 10):
    plt.figure(figsize=(12, 10))
    df_pos.drop(columns=['sentence_count', 'word_count']).sum().sort_values(ascending=False).head(N).plot(kind='bar')
    plt.title(f'{df_name} Top {N} Most Common Part of Speech')
    # turn x labels
    plt.xticks(rotation=45)
    plt.show()

def plot_pos_wordclouds(df: pd.DataFrame, df_name: str, N = 100):
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
    plt.title(f'{df_name} Top {N}Most Common Nouns Proper')

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
    plt.show()


