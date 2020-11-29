from typing import List
from spacy.tokens import Token
import spacy
import nltk

# sklearn stuff
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline


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
                     "afc", "fc", "|"]
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

    def entity_names_events(self, list_events: List[str]):
        return [self.entity_names_labels(event) for event in list_events]

    @staticmethod
    def count_stopwords(doc):
        return len([token for token in doc if token.is_stop])

    @staticmethod
    def train_tfidf(text_list: List[str], **count_vec_kwargs):
        pipe = Pipeline([('count', CountVectorizer(**count_vec_kwargs)),
                         ('tfid', TfidfTransformer())])
        x = pipe.fit_transform(text_list)
        return {'x': x, 'pipeline': pipe}

    def get_numbers(self, doc):
        return [token.text.lower() for token in doc if self.has_numbers(token)]

    def preprocess_word_emb(self, text: str, word_emb: str):
        sent_list = [sent for sent in nltk.sent_tokenize(text)]
        if word_emb == "glove":
            IDs = [[self.nlp.vocab.strings[t.text.lower()] for t in self.nlp(sent) if
                    t.text.isalpha() and t.text.lower() not in stop_words] for sent in sent_list]
        if word_emb == "elmo":
            # no word IDs, just use spacy ids, but without lower/stop words
            # IDs = [[nlp.vocab.strings[t.text] for t in nlp(sent)] for sent in sent_list]
            IDs = [[self.nlp.vocab.strings[t.text] for t in self.nlp(sent)] for sent in sent_list]
        id_list = [x for x in IDs if x != []]  # get rid of empty sents
        text_list = [[token.text for token in self.nlp(x)] for x in sent_list if x != []]
