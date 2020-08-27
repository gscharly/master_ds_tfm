import spacy
import pandas as pd
import networkx as nx
from typing import Dict, List
from spacy.tokens import Token
from collections import OrderedDict
import warnings


class SemanticGraph:
    """
    Class that creates a semantic graph given a text
    """
    def __init__(self, events_list: List[str]):
        self.events_list = events_list
        self.n_sentences = 0
        self.nlp = spacy.load('en_core_web_sm')
        self.sources = []
        self.targets = []
        self.already_in_graph = set()
        # Init empty graph
        self.g = None
        # Sentence-node dictionary
        self.node_dict = dict()

    @staticmethod
    def _token_filter(token: Token) -> bool:
        """
        Filter punctuation, spaces and stopwords
        :param token:
        :return:
        """
        return not (token.is_punct | token.is_space | token.is_stop)

    @staticmethod
    def _token_pos_filter_noun(token):
        """
        Retain nouns and proper nouns
        :param token:
        :return:
        """
        pos = token.tag_
        return pos in ['NN', 'NNP']

    @staticmethod
    def _token_pos_filter_adj(token):
        """
        Retain adjectives
        :param token:
        :return:
        """
        pos = token.tag_
        return pos == 'ADJ'

    @staticmethod
    def _remove_noise(token: Token) -> bool:
        return token.text not in ['\n']

    @staticmethod
    def _get_chunks(sent) -> List:
        """
        Returns chunks within a given sentence
        :param sent:
        :return:
        """
        chunk_list = [chunk for chunk in sent.noun_chunks]
        return chunk_list

    def _add_relation_to_graph(self, source: str, target: str):
        """
        Update source, target and aux list with new relation
        :param source:
        :param target:
        :return:
        """
        tuple_tok = tuple((source, target))
        self.already_in_graph.add(tuple_tok)
        self.already_in_graph.add(tuple_tok[::-1])
        self.sources.append(source)
        self.targets.append(target)

    def _traverse_chunks(self, chunk_list: List) -> List:
        """
        First iteration: roots of chunks will be related to the other parts of the chunk.
        The root will be kept to be related with the other roots of the sentences
        :param chunk_list:
        :return:
        """
        root_list = list()
        for chunk in chunk_list:
            # If it's an EN, keep it. If it's a chunk name, create edges and only select the root
            chunk_root = chunk.ents[0] if chunk.ents else chunk.root.lemma_.lower()
            # print("Root:", chunk_root)
            # print("Root label:", chunk.label_)
            # chunk_root = chunk.root.text.lower()
            if len(chunk) > 1 and not chunk.ents:
                root_list.append(chunk_root)
                for c in chunk:
                    # Create edges between roots and other words of chunk
                    c_text = c.lemma_.lower()
                    if c_text != chunk_root and self._token_filter(c):
                        # print("Other:", c_text)
                        # print("Adding {} to {}".format(c_text, chunk_root))
                        self._add_relation_to_graph(c_text, chunk_root)
            else:
                # Only add if the individual chunk is a noun
                if self._token_pos_filter_noun(chunk[0]):
                    root_list.append(chunk_root)
        return root_list

    def _create_relations_sentence(self, root_list: List):
        """
        Second iteration: relate chunk roots with other roots in the sentence
        :param root_list:
        :return:
        """
        i = 0
        for root in root_list:
            # print('Chunk:', root)
            i += 1
            list_without_token = root_list[i:]
            if len(list_without_token) == 0:
                continue
            # Relate with other tokens in sentence
            for t in list_without_token:
                tuple_tok = tuple((root, t))
                if tuple_tok not in self.already_in_graph and tuple_tok[::-1] not in self.already_in_graph:
                    # print('Adding relation: {}-{}'.format(root, t))
                    self._add_relation_to_graph(root, t)

    def _add_to_node_dict(self, node: str, event: str):
        """
        Add a node as a key of a dictionary, and append sentence as list value
        :param node:
        :param event:
        :return:
        """
        # print(sentence)
        # whole_sentence = ' '.join([token.text for token in sentence])
        tuple_to_add = (self.n_sentences, event)
        # print(whole_sentence)
        if self.node_dict.get(node):
            self.node_dict[node].append(tuple_to_add)
        else:
            self.node_dict[node] = [tuple_to_add]

    @staticmethod
    def _preprocess_event_text(event: str):
        event = event.replace(' (', ', ')
        event = event.replace(')', '')
        return event

    def create_relations(self):
        """
        Create graph traversing all of the sentences
        :return:
        """
        # Traverse events
        for event in self.events_list:
            # print("Event:", event)
            proc_event = self._preprocess_event_text(event)
            # print("Preprocessed event:", proc_event)
            event_doc = self.nlp(proc_event)
            # For each sentence, grab noun chunks
            chunk_list = self._get_chunks(event_doc)
            # print("Chunks: {}".format(chunk_list))
            # First iteration
            # root_list = self._traverse_chunks(chunk_list)
            root_list = [chunk.text if chunk.ents else chunk.text.lower() for chunk in chunk_list]
            # print("List: {}".format(root_list))
            # Add chunk roots to node dict
            for root in root_list:
                self._add_to_node_dict(root, event)
            # Second iteration
            self._create_relations_sentence(root_list)
            # Add 1 to sentence counter
            self.n_sentences += 1
        # print('The provided text has {} sentences'.format(self.n_sentences))

    def create_graph(self):
        # Fill sources and targets
        self.create_relations()
        kg_df = pd.DataFrame({'source': self.sources, 'target': self.targets})
        self.g = nx.from_pandas_edgelist(kg_df, "source", "target")
        return self.g

    def _get_n_hubs(self, n: int):
        """
        Get a list of n ordered nodes using each node's degree.
        :return:
        """
        # assert self.g, "Graph must be created. Please execute create_graph"
        if not self.g:
            warnings.warn("Graph could not be created. Returning empty list")
            return list()
        degree_list = list(self.g.degree())
        return sorted(degree_list, key=lambda x: x[1], reverse=True)[:n]

    def get_n_hubs_sentences(self, n: int) -> Dict:
        """
        Returns a dictionary where keys are the n first nodes with more edges, and values are lists of sentences
        where each node appears.
        :param n:
        :return:
        """
        n_hubs_dict = OrderedDict()
        hubs_list = self._get_n_hubs(n)
        if not hubs_list:
            warnings.warn("Could not create hubs out of graph. Returning empty dict")
            return dict()
        print("Hubs:", hubs_list)
        for hub in hubs_list:
            node = hub[0]
            # Only search for nodes that are root of noun chunks
            if self.node_dict.get(node):
                sentences = self.node_dict[node]
                n_hubs_dict[node] = sentences
            # print('Node {}: {}'.format(node, sentences))
        return n_hubs_dict
