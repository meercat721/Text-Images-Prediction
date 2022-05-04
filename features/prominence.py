"""
This file contains features regarding the prominence of an article, i.e. the popularity, number of wikified entries, etc.
"""
from collections import Counter
import pandas as pd
import numpy as np
import spacy
from scipy import spatial

nlp = spacy.load('de_core_news_lg')

def wiki_entries(text: str) -> int:
    """
    extract the total number of wikified entries from the text
    :param text: the text to analyse
    :return: the cumulative number of wikipedia entries for the given text
    """
    return 0  # TODO


def entity_count(text: str) -> int:
    """
    extract the number of entities in a text
    :param text: the text to analyse
    :return: the cumulative number of entities in a text
    """
    doc = nlp(text)
        
    return len(doc.ents)


def get_all_entities_counter(frame: pd.DataFrame, column: str) -> dict:
    """
    extract all entities from all articles
    :param frame: the whole dataframe
    :param column: the column with the titles
    :return: dict with the entity and the corresponding occurrence
    """
    entities = []
    for doc in nlp.pipe(frame[column], batch_size=50):
        if doc.has_annotation("DEP"):
            entities.append([ent.text for ent in doc.ents])
        else:
            # We want to make sure that the lists of parsed results have the
            # same number of entries of the original Dataframe, so add some blanks in case the parse fails
            entities.append(None)
    
    all_entities_flat = []
    for sublist in entities:
        for item in sublist:
            all_entities_flat.append(item)
    counter = Counter(all_entities_flat)
    return counter


def _get_title_entities(text: str) -> list():
    """
    extract all entities from a title
    :param text: the text to analyse
    :return: list with the entities in the text
    """
    doc = nlp(text)
    entities = []
    if doc.has_annotation("DEP"):
            entities.extend([ent.text for ent in doc.ents])
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        entities.append(None)
    return entities


def get_entities_count(text: str, counter) -> list():
    """
    get a list of the occurrences of all entities in a text 
    :param text: the text to analyse
    :return: list with the entitie counts in the text
    """
    title_entities = _get_title_entities(text)
    counts = []
    for item in title_entities:
        counts.append(counter[item])
    counts.sort(reverse=True)
    return counts


def get_entities_count_as_column(entities, rank: int) -> int:
    """
    number of historical occurrences of one of the mentioned entities
    :param entities: list of occurences of the single entities in a title
    :param rank: the rank of the entity in the sorted list of top 5 entities.
    :return: number of entitity mentions in the past 
    """
    if 0 <= rank < len(entities):
        return entities[rank]
    else: 
        return 0 
    

def get_similarity(text_1, text_2) -> float:
    """
    similarity score fo an article compared to another article
    :param text : text of an article 
    :param text_of_top_article: text of a top ranked article
    :return: similarity score
    """
    doc1 = nlp(text_1)
    doc2 = nlp(text_2)
    # Similarity of two documents
    similarity = doc1.similarity(doc2)
    return similarity    


def _create_word_vector_array(training_df):
    arr = np.empty((0, 300), float)
    for title in training_df["title_without_stopwords"]:
        doc = nlp(title)
        arr = np.append(arr, np.array([doc.vector]), axis=0)
        return arr

def get_ctr_of_most_similar_article(text, training_df):
    arr = _create_word_vector_array(training_df)
    doc = nlp(text)
    index_counter = 0
    highest_similarity = 0.0
    highest_sim_index = None
    for vector in arr:
        similarity = 1 - spatial.distance.cosine(doc.vector, vector)
        if similarity >= 0.98: 
            pass
        else:
            if similarity > highest_similarity: 
                highest_similarity = similarity
                highest_sim_index = index_counter
        index_counter += 1
    print("highest_similarity = " + str(highest_similarity))
    print("highest_sim_index = " + str(highest_sim_index))
    print("most similar article: " + training_df.loc[training_df.index == highest_sim_index, 'title_without_stopwords'].item() + ", with ctr = " + str(df.loc[df.index == highest_sim_index, 'ctr'].item()))
    if highest_sim_index is None:
        print("no value found, similarity = 0.0 ")
        return 0.0
    else:
        return training_df.loc[training_df.index == highest_sim_index, 'ctr'].item()
