"""
This file includes feature calculations regarding the title (and text) sentiment.
"""
from textblob_de import TextBlobDE


def text_polarity(text: str) -> float:
    """
    Computes the polarity of a text using textblob_de
    :param text: the text string
    :return: the polarity in a decimal value, ranging from -1 (negative) to 1 (positive)
    """
    blob = TextBlobDE(text)
    return blob.polarity


def text_subjectivity(text: str) -> float:
    """
    Computes the subjectivity of a text using textblob_de
    :param text: the text string
    :return: the subjectivity in a decimal value, ranging from 0 (very neutral/objective) to 1 (very subjective)
    """
    blob = TextBlobDE(text)
    return blob.subjectivity


def biased_word_proportion(text: str) -> float:
    """
    Calculates the proportion of biased words in a text using the subjectivity. "Subjective" words are those with >0.5
    :param text: the text string to analyse
    :return: percentage of biased words in the text, ranging from 0-1 (0 is 0%, 1 is 100% biased words)
    """
    words = text.split()
    negative = 0
    for word in words:
        if text_subjectivity(word) > 0.5:
            negative += 1
    try:
        return negative / len(words)
    except ZeroDivisionError:
        return 0


def negative_word_proportion(text: str) -> float:
    """
    Calculates the proportion of negative words in a text using the polarity
    :param text: the text string to analyse
    :return: percentage of negative words in the text, ranging from 0-1 (0 is 0%, 1 is 100% negative words)
    """
    words = text.split()
    negative = 0
    for word in words:
        if text_polarity(word) < 0:
            negative += 1
    try:
        return negative / len(words)
    except ZeroDivisionError:
        return 0
