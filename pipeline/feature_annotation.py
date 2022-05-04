from pandas import DataFrame
from tqdm import tqdm

from features.category import get_category, get_category_id
from features.feature_list import *
from features.sentiment import (
    text_polarity,
    text_subjectivity,
    negative_word_proportion,
)
from features.text_properties import (
    word_count,
    symbol_count,
    character_count,
    avg_word_length,
    stop_word_count,
    non_stop_word_count,
    non_stop_word_rate,
    remove_stop_words
)
from features.prominence import(
    entity_count,
    get_all_entities_counter,
    get_entities_count,
    get_entities_count_as_column,
    get_similarity,
    get_ctr_of_most_similar_article,
)


def add_features_without_reference(data: DataFrame) -> DataFrame:
    """
    Adds all possible features to the dataframe
    :param data: the article data
    :return: the article data with all features
    """
    add_sentiment_features(data)
    add_text_property_features(data)
    add_categories(data)
    return data

def add_features_with_reference(data: DataFrame,
                                training_data: DataFrame,
                                complete_category_list,
                                complete_subcategory_1_list,
                                complete_subcategory_2_list) -> DataFrame:
    """
    Adds all possible features to the dataframe
    :param data: the article data
    :return: the article data with all features
    """
    data = add_category_labels(data,
                               complete_category_list,
                               complete_subcategory_1_list,
                               complete_subcategory_2_list)
    data = add_prominence_features(data, training_data)
    return data


def add_sentiment_features(data: DataFrame) -> DataFrame:
    """
    Adds sentiment-related feature to the dataset
    :param data: the article dataset
    :return: the article dataset with the new features
    """
    tqdm.pandas(desc="Title polarity")
    data[TITLE_POLARITY] = data.progress_apply(
        lambda row: text_polarity(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title subjectivity")
    data[TITLE_SUBJECTIVITY] = data.progress_apply(
        lambda row: text_subjectivity(row["title"]), axis=1
    )
    tqdm.pandas(desc="Negative word proportion in title")
    data[TITLE_NEGATIVE_WORD_PROPORTION] = data.progress_apply(
        lambda row: negative_word_proportion(row["title"]), axis=1
    )
    return data


def add_text_property_features(data: DataFrame) -> DataFrame:
    """
    Add text-related features like word length or similar to the dataset
    :param data: the article dataset
    :return: the article dataset with the new features
    """
    tqdm.pandas(desc="Title word count")
    data[TITLE_WORD_COUNT] = data.progress_apply(
        lambda row: word_count(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title exclamation mark count")
    data[TITLE_EXCLAMATION_COUNT] = data.progress_apply(
        lambda row: symbol_count(row["title"], "!"), axis=1
    )
    tqdm.pandas(desc="Title question mark count")
    data[TITLE_QUESTION_COUNT] = data.progress_apply(
        lambda row: symbol_count(row["title"], "\?"), axis=1
    )
    tqdm.pandas(desc="Title hyphen mark count")
    data[TITLE_HYPHEN_COUNT] = data.progress_apply(
        lambda row: symbol_count(row["title"], "-"), axis=1
    )
    tqdm.pandas(desc="Title character count")
    data[TITLE_CHARACTER_COUNT] = data.progress_apply(
        lambda row: character_count(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title average word length")
    data[TITLE_AVG_WORD_LENGTH] = data.progress_apply(
        lambda row: avg_word_length(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title stop word count")
    data[TITLE_STOP_WORD_COUNT] = data.progress_apply(
        lambda row: stop_word_count(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title non stop word count")
    data[TITLE_NON_STOP_WORD_COUNT] = data.progress_apply(
        lambda row: non_stop_word_count(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title non stop word rate")
    data[TITLE_NON_STOP_WORD_RATE] = data.progress_apply(
        lambda row: non_stop_word_rate(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title without stopwords")
    data[TITLE_WITHOUT_STOPWORDS] = data.progress_apply(
        lambda row: remove_stop_words(row["title"]), axis=1
    )
    tqdm.pandas(desc="Title without stopwords")
    data[TITLE_WITHOUT_STOPWORDS_AND_PUNCT] = data.progress_apply(
        lambda row: remove_stop_words(row[TITLE_WITHOUT_STOPWORDS])
        , axis=1
    )
    return data

def add_prominence_features(data: DataFrame, training_data: DataFrame) -> DataFrame:
    """
    Add text-related prominence features to the dataset
    :param data: the article dataset
    :param training_data: the dataset used to obtain whole dataset metrics from
    :return: the article dataset with the new features
    """
    tqdm.pandas(desc="Title entity count")
    data[TITLE_ENTITY_COUNT] = data.progress_apply(
        lambda row: entity_count(row["title"]), axis=1
    )
    # get all entitites of all articles with thier corresponding occurrence
    all_entities = get_all_entities_counter(training_data, "title")

    tqdm.pandas(desc="Recent entity count of all entities in title")
    data[TITLE_RECENT_ENTITY_COUNT_LIST] = data.progress_apply(
        lambda row: get_entities_count(row["title"], all_entities), axis=1
    )
    tqdm.pandas(desc="Recent entity occurrences of entity 1 ")
    data[TITLE_RECENT_ENTITY_COUNT_1] = data.progress_apply(
        lambda row: get_entities_count_as_column(row[TITLE_RECENT_ENTITY_COUNT_LIST],0), axis=1
    )
    tqdm.pandas(desc="Recent entity occurrences of entity 2 ")
    data[TITLE_RECENT_ENTITY_COUNT_2] = data.progress_apply(
        lambda row: get_entities_count_as_column(row[TITLE_RECENT_ENTITY_COUNT_LIST],1), axis=1
    )
    
    top_ctr_articles = training_data[['title', "ctr"]].sort_values('ctr', ascending=False).head(10)
    
    tqdm.pandas(desc="Similarity to nr. 1 article")
    data[TITLE_SIMILARITY_TO_TOP_1] = data.progress_apply(
        lambda row: get_similarity(
            row["title"],
            top_ctr_articles["title"][top_ctr_articles.index[0]]), axis=1
    )
    tqdm.pandas(desc="Similarity to nr. 2 article")
    data[TITLE_SIMILARITY_TO_TOP_2] = data.progress_apply(
        lambda row: get_similarity(
            row["title"],
            top_ctr_articles["title"][top_ctr_articles.index[1]]), axis=1
    )
    tqdm.pandas(desc="Similarity to nr. 3 article")
    data[TITLE_SIMILARITY_TO_TOP_3] = data.progress_apply(
        lambda row: get_similarity(
            row["title"],
            top_ctr_articles["title"][top_ctr_articles.index[2]]), axis=1
    )
    tqdm.pandas(desc="Similarity to nr. 4 article")
    data[TITLE_SIMILARITY_TO_TOP_4] = data.progress_apply(
        lambda row: get_similarity(
            row["title"],
            top_ctr_articles["title"][top_ctr_articles.index[3]]), axis=1
    )
    tqdm.pandas(desc="Similarity to nr. 5 article")
    data[TITLE_SIMILARITY_TO_TOP_5] = data.progress_apply(
        lambda row: get_similarity(
            row["title"],
            top_ctr_articles["title"][top_ctr_articles.index[4]]), axis=1
    )
    
    # tqdm.pandas(desc="get_ctr_of_most_similar_article")
    # data[TITLE_CTR_OF_MOST_SIMILAR] = data.progress_apply(
    #     lambda row: get_ctr_of_most_similar_article(
    #         row[TITLE_WITHOUT_STOPWORDS_AND_PUNCT],
    #         training_data), axis=1
    # )

    return data

def add_categories(data: DataFrame) -> DataFrame:
    """
    Add category-related features like the category or its numerical value
    :param data: the article dataset
    :return: the article dataset with category features
    """
    tqdm.pandas(desc="Category")
    data[CATEGORY] = data.progress_apply(lambda row: get_category(row["url"],category_type="category"), axis=1)
    category_feature_list = data[CATEGORY].unique()
    
    tqdm.pandas(desc="subcategory_1")
    data[SUBCATEGORY_1] = data.progress_apply(lambda row: get_category(row["url"],category_type="subcategory_1"), axis=1)

    tqdm.pandas(desc="subcategory_2")
    data[SUBCATEGORY_2] = data.progress_apply(lambda row: get_category(row["url"],category_type="subcategory_2"), axis=1)

    return data

def add_category_labels(data: DataFrame,
                        complete_category_list,
                        complete_subcategory_1_list,
                        complete_subcategory_2_list) -> DataFrame:
    """
    Add Category as one hot encoded features
    :param data: the article dataset
    :param training_data: training_data for reference
    :return: the article dataset with category features
    """
    print("add Category as one hot encoded features")
    data = get_category_id(
        data=data,
        category_list=complete_category_list,
        column_name=CATEGORY,
        column_short="MC")
    data = get_category_id(
        data=data,
        category_list=complete_subcategory_1_list,
        column_name=SUBCATEGORY_1,
        column_short="SC_1_")
    data = get_category_id(
        data=data,
        category_list=complete_subcategory_2_list,
        column_name=SUBCATEGORY_2,
        column_short="SC_2_")

    return data
