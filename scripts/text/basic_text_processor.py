from typing import List
from spacy.tokens import Token
import spacy


class BasicTextProcessor:
    """Class that provides basic methods for text processing"""

    def __init__(self, text_model: str = 'en_core_web_sm'):
        self.nlp = spacy.load(text_model)

    def token_list(self, text: str):
        doc = self.nlp(text)
        return doc

    @staticmethod
    def has_numbers(token: Token) -> bool:
        text = token.text.lower()
        return any(char.isdigit() for char in text)

    @staticmethod
    def token_filter(token: Token) -> bool:
        """
        Filter punctuation and spaces
        :param token:
        :return:
        """
        return not (token.is_punct | token.is_space)

    @staticmethod
    def token_filter_stopword(token: Token):
        """Filter stopwords"""
        return not token.is_stop

    @staticmethod
    def filter_noisy_characters(token: Token) -> bool:
        """
        Remove @, # and urls.
        :param token:
        :return:
        """
        text = token.text.lower()
        wild_char = ["\n", "'s", "’s", "com", "https", "twitter", "@", "#", ".", "pa",
                     "la", "aug", "nil", "boo", "espn", "wcq", "cf", "ebb",
                     "afc", "fc"]
        return not any(char in text for char in wild_char)

    @staticmethod
    def token_pos_filter_noun(token: Token) -> bool:
        """
        Retain nouns and proper nouns
        :param token:
        :return:
        """
        pos = token.tag_
        return pos in ['NN', 'NNP']

    @staticmethod
    def token_pos_filter_adj(token: Token) -> bool:
        """
        Retain adjectives
        :param token:
        :return:
        """
        pos = token.tag_
        return pos == 'ADJ'

    def get_sentences(self, text: str) -> List:
        """
        Use spacy to divide text into sentences
        :return:
        """
        # Minus
        doc = self.nlp(text)
        sentences = doc.sents
        return sentences

    @staticmethod
    def get_chunks(sent) -> List:
        """
        Returns chunks within a given sentence
        :param sent:
        :return:
        """
        chunk_list = [chunk for chunk in sent.noun_chunks]
        return chunk_list

    def entity_names(self, text):
        doc = self.nlp(text)
        ents = [ent.text for ent in doc.ents]
        return ents

    def entity_names_labels(self, doc):
        if type(doc) is str:
            doc = self.nlp(doc)
        ents_labels = [(ent.text, ent.label_) for ent in doc.ents]
        return ents_labels

