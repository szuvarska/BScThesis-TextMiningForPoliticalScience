import pandas as pd
import spacy


def read_txt_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read()


def read_sentences(txt):
    sentences = []
    start = 0
    end = 0
    for i in range(len(txt)):
        if txt[i] == '\n':
            end = i
            sentences.append((start, end, txt[start:end]))
            start = i+1
    sentences = sentences[7:]
    sentences = pd.DataFrame(sentences, columns=['start', 'end', 'text'])
    return sentences


def clean_ann(ann):
    ids, categories, arg1s, arg2s, texts = [], [], [], [], []
    ann = ann.split('\n')
    while ann[-1] == '':
        ann = ann[:-1]
    ann = [a for a in ann if a[0][0] != 'A']
    for a in ann:
        ann_split = a.split('\t')
        ids.append(ann_split[0])
        args = ann_split[1].split()
        categories.append(args[0])

        if args[0] == 'refers':
            arg1s.append(args[1].split(':')[1])
            arg2s.append(args[2].split(':')[1])
        else:
            arg1s.append(args[1])
            arg2s.append(args[2])
        texts.append(ann_split[2])

    return pd.DataFrame({'id': ids, 'category': categories, 'arg1': arg1s, 'arg2': arg2s, 'text': texts})


def make_relations_df(ann_df):
    relations = ann_df[ann_df['category'] == 'refers']
    relations = relations.drop(columns=['category', 'text'])
    relations = relations.reset_index(drop=True)
    return relations


def make_final_df(relations, ann_df, sentences):
    categories2 = []
    arg1s = []
    arg2s = []
    whole_sentences = []

    for _, row in relations.iterrows():
        arg1 = row['arg1']
        arg2 = row['arg2']
        arg1_row = ann_df[ann_df['id'] == arg1]
        arg2_row = ann_df[ann_df['id'] == arg2]

        if not arg1_row['category'].values[0] in ['blame', 'praise']:
            arg1_row, arg2_row = arg2_row, arg1_row

        #check if both arg1 and arg2 are in the same sentence
        arg1_start = int(arg1_row['arg1'])
        arg1_end = int(arg1_row['arg2'])
        arg2_start = int(arg2_row['arg1'])
        arg2_end = int(arg2_row['arg2'])
        arg1_sentence = sentences[(sentences['start'] <= arg1_start) & (sentences['end'] >= arg1_end)]
        arg2_sentence = sentences[(sentences['start'] <= arg2_start) & (sentences['end'] >= arg2_end)]
        if arg1_sentence.empty or arg2_sentence.empty or arg1_sentence.index[0] != arg2_sentence.index[0]:
            continue
        else:
            categories2.append(arg1_row['category'].values[0])
            arg1s.append(arg1_row['text'].values[0])
            arg2s.append(arg2_row['text'].values[0])
            whole_sentences.append(arg1_sentence['text'].values[0])

    return pd.DataFrame({'annotation_category': categories2, 'annotation': arg1s, 'entity': arg2s, 'sentence': whole_sentences})


def add_ner_to_sentence(final_df):
    nlp = spacy.load('en_core_web_sm')
    all_entities = []
    all_entities_annotated = []

    entity_legend = {
        "ORG": "Organizations: companies, agencies, institutions, etc.",
        "PERSON": "People (including fictional)",
        "GPE": "Geopolitical entities: countries, cities, states",
        "LOC": "Non-geopolitical locations: mountain ranges, bodies of water, etc.",
        "NORP": "Nationalities or religious or political groups",
        "EVENT": "Named events: hurricanes, battles, wars, sports events, etc."
    }

    for i in range(len(final_df)):
        sentence = final_df['sentence'][i]
        doc = nlp(sentence)
        entities = [(ent.text) for ent in doc.ents if ent.label_ in entity_legend.keys()]
        entities = list(set(entities))
        entities_annotated = [ ent for ent in entities if ent in final_df['entity'][i]]

        all_entities.append(entities)
        all_entities_annotated.append(entities_annotated)

    final_df['entities_annotated'] = all_entities_annotated

    return final_df


def atomize_entities(final_df):
    atomized_df = pd.DataFrame(columns=['annotation_category', 'annotation', 'entity', 'sentence', 'entity_atomized'])
    for i in range(len(final_df)):
        # for every value in entities_annotated make a seperate row and add it to a new df
        if len(final_df['entities_annotated'][i]) > 0:
            for ent in final_df['entities_annotated'][i]:
                new_row = pd.DataFrame([{
                    'annotation_category': final_df['annotation_category'][i],
                    'annotation': final_df['annotation'][i],
                    'entity': final_df['entity'][i],
                    'sentence': final_df['sentence'][i],
                    'entity_atomized': ent
                }])
                atomized_df = pd.concat([atomized_df, new_row], ignore_index=True)
        else:
            new_row = pd.DataFrame([{
                    'annotation_category': final_df['annotation_category'][i],
                    'annotation': final_df['annotation'][i],
                    'entity': final_df['entity'][i],
                    'sentence': final_df['sentence'][i],
                    'entity_atomized': final_df['entity'][i]
                }])
            atomized_df = pd.concat([atomized_df, new_row], ignore_index=True)

    return atomized_df


def perform_preprocessing(path):
    txt = read_txt_file(path)
    sentences = read_sentences(txt)
    ann = read_txt_file(path.replace('.txt', '.ann'))
    if len(ann) < 10: return None
    ann_df = clean_ann(ann)
    relations = make_relations_df(ann_df)
    final_df = make_final_df(relations, ann_df, sentences)
    ner_df = add_ner_to_sentence(final_df)
    atomized_df = atomize_entities(ner_df)
    return atomized_df