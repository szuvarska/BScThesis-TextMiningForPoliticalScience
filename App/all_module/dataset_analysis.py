from shiny import ui
from Preparations.EDA_script import prepare_df_for_eda
from Data.Sentances_df.sentences_script import make_sentences_df
from NER_and_ED.NER_ED_script import (perform_ner, calculate_entity_distribution, find_most_common_entity_types,
                                      find_most_common_entities_per_type)
from Sentiment.sentiment_script import perform_sentiment_analysis, calculate_sentiment_dist_per_target_without_plot
import os
import asyncio
import zipfile
import pandas as pd
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


def analyze_dataset(file_paths: list, dataset_name: str, not_enough_data: bool, progress_callback=None):
    print("Reading articles from text files...")
    if progress_callback:
        progress_callback(1, message="Reading articles from text files...")
    articles = []
    for file_path in file_paths:
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding="utf-8") as file:
                txt = file.read()
                articles.append(txt)
    print(f"Loaded {len(articles)} articles.")
    if progress_callback:
        progress_callback(3, message=f"Loaded {len(articles)} articles.")

    published_time = []
    article_text = []
    article_title = []
    author = []

    original_dataset_name = dataset_name
    dataset_name = dataset_name.replace(' ', '_')

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

    name_textcontain_new_preprocessed = pd.DataFrame(
        {'published_time': published_time, 'article_title': article_title, 'author': author,
         'article_text': article_text})
    print("Saving preprocessed dataset to CSV file...")
    name_textcontain_new_preprocessed.to_csv(f"Data/{dataset_name.lower()}_textcontain_new_preprocessed.csv",
                                             index=False)
    print("Dataset saved successfully.")
    if progress_callback:
        progress_callback(5, message="Dataset preprocessed and saved successfully.")

    # prepare EDA
    print("Preparing EDA dataframes...")
    df_name, df_pos_name = prepare_df_for_eda(name_textcontain_new_preprocessed)
    print("EDA preparation complete. Saving dataframes to CSV files...")
    df_name.to_csv(f"Preparations/Data_for_EDA/df_{dataset_name.lower()}.csv", index=False)
    df_pos_name.to_csv(f"Preparations/Data_for_EDA/df_pos_{dataset_name.lower()}.csv", index=False)
    print("EDA dataframes saved successfully.")
    if progress_callback:
        progress_callback(7, message="EDA dataframes saved successfully.")

    # prepare sentences_df
    print("Creating sentences dataframe...")
    if len(articles) < 50:
        print("Less than 50 articles. Topics will not be calculated.")
        sencences_name, _, _ = make_sentences_df(name_textcontain_new_preprocessed, perform_topics=False,
                                                 progress_callback=progress_callback)
    else:
        sencences_name, _, _ = make_sentences_df(name_textcontain_new_preprocessed, progress_callback=progress_callback)
    print("Sentences dataframe created. Saving to CSV file...")
    sencences_name.to_csv(f"Data/Sentances_df/sentences_{dataset_name.lower()}.csv", index=False)
    print("Sentences dataframe saved successfully.")
    if progress_callback:
        progress_callback(70, message="Sentences dataframe saved successfully.")

    # prepare NER
    print("Performing NER analysis...")
    if progress_callback:
        progress_callback(75, message="Performing additional NER analysis...")
    find_most_common_entities_per_type(sencences_name, dataset_name,
                                       f"NER_and_ED/Results/{dataset_name}_top_40_entities.csv", for_shiny=True)
    print("NER analysis complete.")
    name_top_40_entities = pd.read_csv(f"NER_and_ED/Results/{dataset_name}_top_40_entities.csv")

    name_target_entities = list(name_top_40_entities.sort_values(by='Count', ascending=False)["Word"].head(20))

    filename_textcontain = f"Data/{dataset_name.lower()}_textcontain_new_preprocessed.csv"

    print("Performing sentiment analysis...")
    if progress_callback:
        progress_callback(80, message="Performing additional sentiment analysis...")

    tsc_results_df, _ = perform_sentiment_analysis(filename_textcontain, name_target_entities, original_dataset_name,
                                                   output_directory_path="Sentiment/Results")
    calculate_sentiment_dist_per_target_without_plot(tsc_results_df, original_dataset_name)

    if len(articles) < 50:
        progress_callback(99, message="Some plots might not be available due to the small dataset size.")
        not_enough_data.set(True)

    print("Sentiment analysis complete.")
    if progress_callback:
        progress_callback(100, message="Sentiment analysis complete.")

    return True


async def analyze_dataset_reactive(files, dataset_choices, dataset_filter_value, dataset_name, not_enough_data):
    if files:
        # Extract file paths
        file_paths = [file["datapath"] for file in files]

        if len(file_paths) == 1 and file_paths[0].endswith(".zip"):
            folder_name = "Uploaded Dataset"
            # Extract zip contents
            extracted_files_dir = f"/tmp/{folder_name}"
            os.makedirs(extracted_files_dir, exist_ok=True)
            with zipfile.ZipFile(file_paths[0], 'r') as zip_ref:
                zip_ref.extractall(extracted_files_dir)

            # Check if the extracted files are in a subdirectory
            subdirs = [os.path.join(extracted_files_dir, d) for d in os.listdir(extracted_files_dir) if os.path.isdir(os.path.join(extracted_files_dir, d))]
            if subdirs:
                extracted_files_dir = subdirs[0]

            file_paths = [
                os.path.join(extracted_files_dir, f) for f in os.listdir(extracted_files_dir) if f.endswith(".txt")
            ]
            if file_paths:
                if dataset_name is None or dataset_name == "":
                    if subdirs:
                        dataset_name = os.path.basename(subdirs[0])
                    else:
                        dataset_name = folder_name
            else:
                ui.notification_show("No .txt files found in the uploaded zip file.", type="error")
        elif dataset_name is None or dataset_name == "":
            folder_path = os.path.dirname(file_paths[0])
            dataset_name = os.path.basename(folder_path)

        with ui.Progress(min=1, max=100) as p:
            try:
                # Set progress at the beginning
                p.set(message="Starting analysis...")

                # Blocking function (ensure progress updates are in the main thread)
                if analyze_dataset(file_paths, dataset_name, not_enough_data, progress_callback=p.set):
                    dataset_choices.set(dataset_choices.get() + [dataset_name])
                    dataset_filter_value.set(dataset_name)
            except Exception as e:
                ui.notification_show(str(e), type="error")
