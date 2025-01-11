import pandas as pd
from Sentiment.sentiment_script import vader_sentiment, split_into_sentences
from tqdm import tqdm
from NER_and_ED.NER_ED_script import perform_ner_for_sentences
from Topics.bratopic_script import perform_bratopic
dict_name = "../../NER_and_ED/dict_ukraine.csv"

def make_sentences_df(df: pd.DataFrame, dict_name: str = None):
    article_text = []
    article_id = []
    dates = []
    #prepare ID
    for i in range(len(df)):
        article_text.append(str(df['article_text'][i]))
        date = str(df['published_time'][i])
        title = str(df['article_text'][i]).replace(" ", "_")
        id = str(i)
        article_id.append(id)

    senteces_all = []
    sentences_id_all = []
    senteces_date_all = []

    #split article into sentences
    for article, id, date in tqdm(zip(article_text, article_id, df['published_time'])):
        sentences = split_into_sentences(article)
        sentences_id = []
        for i in range(len(sentences)):
            # sentence number 3digits with 0 padding
            sentences_id.append(str(id).zfill(3) + "-" + str(i).zfill(3))
            senteces_date_all.append(date)
        senteces_all += sentences
        sentences_id_all += sentences_id

    articles_id = [x.split("-")[0] for x in sentences_id_all]
    sentences_df = pd.DataFrame({'sentence_id': sentences_id_all, 'article_id': articles_id, 'article_text': senteces_all,
                                 'published_time': senteces_date_all})

    #perform srntiment analysis
    sentences_df['sentiment'] = sentences_df['article_text'].apply(vader_sentiment)
    #perform NER
    ner_df = perform_ner_for_sentences(sentences_df, dict_name)
    #perform topics
    topics, probs, topic_detailes_df, topic_model = perform_bratopic(ner_df)
    topics_df = ner_df
    topics_df['topic_id'] = topics
    topics_df = topics_df.merge(topic_detailes_df[['Topic', 'Representation']], left_on='topic_id',
                                right_on='Topic')
    topics_df = topics_df.drop(columns=['Topic'], inplace=False)
    topics_df = topics_df.rename(columns={"Representation": "topic_representation"})
    topics_df = topics_df.sort_values(by=['sentence_id'], inplace=False)

    return topics_df, topic_detailes_df, topic_model
