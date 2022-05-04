import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import numpy as np
from textblob import TextBlob
import string

from features.feature_list import *


def get_unique_article_categories(column_name,
                                  jan19_articles,
                                  feb19_articles,
                                  mar19_articles,
                                  may19_articles):
    cat_list = []
    cat_list.extend(jan19_articles[column_name].unique())
    cat_list.extend(feb19_articles[column_name].unique())
    cat_list.extend(mar19_articles[column_name].unique())
    cat_list.extend(may19_articles[column_name].unique())
    
    cat_list = list(set(cat_list))
    return cat_list

def get_ctr(df):
    
    tqdm.pandas(desc="calculate ctr")
    df[CTR] = df.progress_apply(
        lambda row: calculate_ctr(row[N_CLICKS], row[N_RECS]), axis=1,
    )
    return df 

def calculate_ctr(clicks, recs):
    if recs != 0:
        ctr = clicks / recs 
    else:
        ctr = 0.0
    return ctr

def label_txt_to_pandas(location: str) -> DataFrame:
    """
    Converts a label txt file to a pandas dataframe
    :param location: the file location
    :return: a pandas dataframe with the columns 'id','imageId','labelId', and so on...
    """
    with open(location) as f:
        content = f.readlines()
    labels = [x.strip() for x in content]

    def to_list(line: str):
        return [x.replace("'", "") for x in line.split("','")]

    labels = list(map(lambda line: to_list(line), labels))
    df = DataFrame(labels, columns=labels.pop(0))
    # convert strings to floats/ints
    df["labelProbability"] = df["labelProbability"].astype(float)
    df["labelId"] = df["labelId"].astype(int)
    df["configurationId"] = df["configurationId"].astype(int)
    df["imageId"] = df["imageId"].astype(int)

    return df


def map_labels_to_articles(articles: DataFrame, labels: DataFrame) -> DataFrame:
    """
    Maps the labels to the articles of a dataset
    :param labels: the total label set
    :param articles: the article set
    :return: the articles with additional label columns
    """
    # make queries faster by filtering only within image id range
    max_article_iid = articles["iid"].max()
    min_article_iid = articles["iid"].min()
    label_range = labels.loc[
        (labels["imageId"] >= min_article_iid) & (labels["imageId"] <= max_article_iid)
    ]

    # assign human/not human value for each article
    def find_human_delta(image_id):
        human = label_range.loc[
            (label_range["imageId"] == image_id) & (label_range["configurationId"] == 1)
        ]
        if human.shape[0] < 2:  # not all labels available
            return 0
        return (
            human.loc[human["labelValue"] == "human", "labelProbability"].iloc[0]
            - human.loc[
                human["labelValue"] == "without human", "labelProbability"
            ].iloc[0]
        )

    # def get_labels_sentiment(image_id):
    #     arr = label_range.loc[(label_range["imageId"] == image_id)
    #                            & (label_range["configurationId"] == 2)]["labelValue"].values.flatten()
    #     punctuation_string = string.punctuation
    #     for i in punctuation_string:
    #         arr = str(arr).replace(i, '').replace('\n', '')
    #         blob = TextBlob(arr)
    #         p = blob.sentences[0].sentiment[0]
    #         s = blob.sentences[0].sentiment[1]
    #
    #     return p, s


    tqdm.pandas(desc="Human/not human delta labels")
    articles["human_delta"] = articles.progress_apply(
        lambda row: find_human_delta(row["iid"]), axis=1,
    )

    # articles[["label_polarity"], ["label_subjectivity"]] = articles.progress_apply(
    #     lambda row: get_labels_sentiment(row["iid"]), axis=1,
    # )

    return articles


# def map_march_articles_to_labels(articles: DataFrame, labels: DataFrame) -> DataFrame:
#     """
#     This maps the labels of march to the articles. this method is separate because we use 201903-item-Image-Mapping
#     :param labels: the total label set
#     :param articles: the article set
#     :return: the articles with additional label columns
#     """
#     mappings = pd.read_csv(r"data/201903-item-Image-Mapping.tsv", sep="\t", header=0)
#     articles["iid"] = mappings["iid"]
#     map_labels_to_articles(labels=labels, articles=articles)
#     return articles


def cleanup_labels(labels: DataFrame, probability_threshold=0.1) -> DataFrame:
    """
    Removes non unique or faulty labels.
    :param probability_threshold: the filter threshold for the probability of a label
    :param labels: the label dataset
    :return: a new, clean label dataset
    """
    labels.drop_duplicates(subset="id", keep=False, inplace=True)
    # drop labels below probability
    labels = labels[labels["labelProbability"] >= probability_threshold].dropna()
    # drop latter suggestions
    labels = labels[labels["labelId"] <= 3].dropna()

    return labels
