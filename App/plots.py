from pathlib import Path
import sys
import ast
import pandas as pd
import mpld3
from io import BytesIO
import base64

sys.path.append(str(Path(__file__).resolve().parent.parent))
from NER_and_ED.NER_ED_script import find_most_common_entity_types


def generate_entity_types_plot(dataset_name: str):
    ner_df = pd.read_csv(f"../NER_and_ED/Results/{dataset_name.replace(' ', '_')[:-9]}_with_NER.csv")
    ner_df['NER'] = ner_df['NER'].apply(ast.literal_eval)
    fig = find_most_common_entity_types(ner_df, dataset_name, save_plot=False)
    return find_most_common_entity_types(ner_df, dataset_name, save_plot=False)

