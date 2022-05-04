"""
This file contains features related to the article category (contained within the article URL)
"""
import re
from typing import List
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import pandas as pd


def get_category(url: str, category_type: str) -> str:
    """
    Extracts the category from the url
    :param url: the url as a string
    :return: the category as a string
    """
    
    if category_type == "category": 
        category = re.findall("https://www.ksta.de/(.*?)/", url)
    if category_type == "subcategory_1": 
        category = re.findall("https://www.ksta.de/.*?/(.*?)/", url)
    if category_type == "subcategory_2": 
        category = re.findall("https://www.ksta.de/.*?/.*?/(.*?)/", url)
    
    category_str = ",".join(category)
    
    if category_str == "":
        return "no-category"
    else:
        return category_str


def get_category_id(data: pd.DataFrame,
                    category_list, 
                    column_name: str, 
                    column_short: str):
    """
    Get the category numerical value (0,1,..)
    :param article_category: the article category to get the id for
    :param categories: the categories list. 0 is always no-category
    :return: the category id as an integer
    """
    category_encoder = LabelEncoder()
    
    
    #category_encoder = LabelBinarizer()
    category_encoder.fit(category_list)
    
    data[column_name + "_enc"] = category_encoder.transform(data[column_name])
    
    #transformed = category_encoder.transform(data[column_name])
    #category_features = [column_short + str(i) 
     #                    for i in range(0, len(category_list))]

    #ohe_df = pd.DataFrame(transformed, columns=category_features)
    #data = pd.concat([data, ohe_df], axis=1)
    
    return data
