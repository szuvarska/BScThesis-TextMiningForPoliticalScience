import pandas as pd
import os
from Preparations.EDA_script import prepare_df_for_eda
from Data.Sentances_df.sentences_script import make_sentences_df
from NER_and_ED.NER_ED_script import perform_ner, calculate_entity_distribution, find_most_common_entity_types, find_most_common_entities_per_type
from Sentiment.sentiment_script import perform_sentiment_analysis


def Prepare_dataset(directory_path: str):
    dataset_name = directory_path.split('\\')[-1]
    articles = []
    
    print("Reading articles from text files...")
    #input
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(directory_path+'\\'+filename, 'r', encoding="utf-8") as file:
                txt = file.read()
                articles.append(txt)
    print(f"Loaded {len(articles)} articles.")

    published_time = []
    article_text = []
    article_title = []
    author = []

    for txt in articles:
        is_text = False
        text_content = []
        for line in txt.splitlines():
            if line.startswith("Published Time:"):
                published_time.append(line.replace("Published Time: ", ""))
            elif line.startswith("Title:"):
                article_title.append(line.replace("Title: ", ""))
            elif line.startswith("Author:"):
                author.append(line.replace("Author: ", ""))
            elif line.startswith("Text:"):
                is_text = True
            elif is_text:
                text_content.append(line)
        article_text.append("\n".join(text_content))

    name_textcontain_new_preprocessed = pd.DataFrame({'published_time': published_time, 'article_title': article_title, 'author': author, 'article_text': article_text})
    print("Saving preprocessed dataset to CSV file...")
    name_textcontain_new_preprocessed.to_csv(f"..\\Data\\{dataset_name}_textcontain_new_preprocessed.csv", index=False)
    print("Dataset saved successfully.")

    #prepare EDA
    print("Preparing EDA dataframes...")
    df_name, df_pos_name = prepare_df_for_eda(name_textcontain_new_preprocessed)
    print("EDA preparation complete. Saving dataframes to CSV files...")
    df_name.to_csv(f"..\\Preparations\\Data_for_EDA\\df_{dataset_name}.csv", index=False)
    df_pos_name.to_csv(f"..\\Preparations\\Data_for_EDA\\df_pos_{dataset_name}.csv", index=False)
    print("EDA dataframes saved successfully.")

    #prepare sentences_df
    print("Creating sentences dataframe...")
    sencences_name, name_topics_details, name_topics_model = make_sentences_df(name_textcontain_new_preprocessed)
    print("Sentences dataframe created. Saving to CSV file...")
    sencences_name.to_csv(f"..\\Data\\Sentances_df\\sentences_{dataset_name}.csv", index=False)
    print("Sentences dataframe saved successfully.")

    #prepare NER
    print("Performing NER analysis...")
    find_most_common_entities_per_type(sencences_name, dataset_name,
                                       f"..\\NER_and_ED\\Results\\{dataset_name}_top_40_entities.csv")
    print("NER analysis complete.")
    name_top_40_entities = pd.read_csv(f"..\\NER_and_ED\\Results\\{dataset_name}_top_40_entities.csv")

    name_target_entities = list(name_top_40_entities.sort_values(by='Count', ascending=False)["Word"].head(20))

    filename_textcontain = f"..\\Data\\{dataset_name}_textcontain_new_preprocessed.csv"

    tsc_results_df, vader_results_df = perform_sentiment_analysis(filename_textcontain, name_target_entities,
                                                                  dataset_name)

    print("Performing sentiment analysis...")

    tsc_results_df, vader_results_df = perform_sentiment_analysis(filename_textcontain, name_target_entities,
                                                                  dataset_name)
    print("Sentiment analysis complete.")


if __name__ == '__main__':
    dir_path = "C:\\Users\\lukas\\Desktop\\PRACA INŻYNIERSKA\\data_test"
    Prepare_dataset(dir_path)