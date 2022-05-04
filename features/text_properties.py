"""
This file contains low-level text properties, with the categories punctuation, brevity,
"""
from nltk.corpus import stopwords
import string


def character_count(text: str) -> int:
    """
    Get the character count in a string
    :param text: the text to count characters in
    :return: the amount of characters
    """
    return len(text)


def word_count(text: str) -> int:
    """
    Get the word count of a text
    :param text: the text to analyse
    :return: amount of words
    """
    return len(text.split())


def symbol_count(text: str, symbol="!") -> int:
    """
    Amount of symbols in a text.
    :param symbol: the symbol to count. The default is the exclamation mark (!)
    :param text: the text to analyse
    :return: the amount of exclamation marks
    """
    return text.count(symbol)


def avg_word_length(text: str) -> float:
    """
    Average word length. 
    :param text: the text to analyse
    :return: the average word length
    get by df["avg_wrd_length_title"] = df["title"].apply(lambda x: avg_word_legnth(x))
    """
    words = text.split()
    average = sum(len(word) for word in words) / len(words)
    return average


def stop_word_count(text: str) -> int:
    """
    Stop word count. 
    :param text: the text to analyse
    :return: number of stop words
    """
    stop_words_catalogue = set(stopwords.words("german"))
    words = text.split()
    return len(set(words) & stop_words_catalogue)


def non_stop_word_count(text: str) -> int:
    """
    non Stop word count. 
    :param text: the text to analyse
    :return: number of non stop words
    """
    stop_words_count = stop_word_count(text)
    all_words_count = word_count(text)
    return all_words_count - stop_words_count


def non_stop_word_rate(text: str) -> float:
    """
    non Stop word rate. 
    :param text: the text to analyse
    :return: rate of non stop words
    """
    stop_words_count = stop_word_count(text)
    all_words_count = word_count(text)
    non_stop_words_count = all_words_count - stop_words_count
    return non_stop_words_count / all_words_count

def remove_stop_words(text: str):
    stop_words = set(stopwords.words('german'))
    return ' '.join([word for word in text.split() if word not in (stop_words)])

def remove_punctuation(text: str):
    punct = '!"#$%&\'()*+,„“./:;<=>?@[\\]^_`{}~'   # `|` is not present here
    transtab = str.maketrans(dict.fromkeys(punct, ''))

    text =  '|'.join(text.tolist()).translate(transtab).split('|')

    punct = '-'   # hyphen between words
    transtab = str.maketrans(dict.fromkeys(punct, ' '))
    text = '|'.join(text.tolist()).translate(transtab).split('|')

    punct = '–'   # hyphen between sentences
    transtab = str.maketrans(dict.fromkeys(punct, ''))
    text = '|'.join(text.tolist()).translate(transtab).split('|')
    
    return text

    
