import re 
import string

from nltk.tokenize import word_tokenize


def general_preprocess(text: str, punctuations_to_remove=string.punctuation, unify_commas=False):
    """ Applying general preprocessing on the given text. It removes: html tags, numbers, punctuations. 
    I makes it lower case and has a flag to unify commas, which may be used if we change punctuations_to_remove.
    """
    # removing html tags
    text_no_html = re.sub('<.*>', '', text)

    # Removing numbers
    text_number = re.sub(r'\d+', '', text_no_html)

    # Lowering text
    text_lower = text_number.lower()

    # removing punctuation
    text_no_punctuations = text_lower.translate(str.maketrans('', '', punctuations_to_remove))

    # making one type of commas
    if unify_commas:
        text_no_punctuations = text_no_punctuations.replace('.', ',')

    return text_no_punctuations


def tokenize(text: str) -> str:
    return word_tokenize(text)


def remove_stopwords(tokens: list, stopwords:list):
    return [token for token in tokens if token not in stopwords]


def total_preprocess(text, stopwords=None):
    """ Lets to redo all processing that was applied. Maybe a bit unsafe since it is hardcoded and one might customize preprocessing. 
    """
    text = general_preprocess(text)
    tokens = word_tokenize(text)
    
    if stopwords is not None:
        tokens = remove_stopwords(tokens, stopwords)

        tokens = [token for token in tokens if token not in stopwords]
    
    return tokens
