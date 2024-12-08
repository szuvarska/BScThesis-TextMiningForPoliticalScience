from pathlib import Path
import sys
import ast
import pandas as pd
import mpld3
from io import BytesIO
import base64

sys.path.append(str(Path(__file__).resolve().parent.parent))
from NER_and_ED.NER_ED_script import find_most_common_entity_types, find_most_common_entities_per_type_for_shiny


def generate_entity_types_plot(dataset_name: str):
    dataset_name = dataset_name.replace(' ', '_')[:-9]
    dataset_name = dataset_name.replace('during', 'after')
    ner_df = pd.read_csv(f"../NER_and_ED/Results/{dataset_name}_with_NER.csv")
    ner_df['NER'] = ner_df['NER'].apply(ast.literal_eval)
    return find_most_common_entity_types(ner_df, dataset_name, for_shiny=True)


def generate_most_common_entities_plot(dataset_name: str, entity_type: str = 'ORG'):
    dataset_name = dataset_name.replace(' ', '_')[:-9]
    dataset_name = dataset_name.replace('during', 'after')
    ner_df_path = f"../NER_and_ED/Results/{dataset_name}_top_40_entities.csv"
    return find_most_common_entities_per_type_for_shiny(dataset_name, ner_df_path, entity_type)
