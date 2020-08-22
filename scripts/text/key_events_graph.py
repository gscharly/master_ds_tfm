import spacy
import pandas as pd
import networkx as nx
from typing import Dict, List
from spacy.tokens import Token
from collections import OrderedDict


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

    def _get_sentences(self) -> List:
        """
        Use spacy to divide text into sentences
        :return:
        """
        # Minus
        doc = self.nlp(self.text)
        sentences = doc.sents
        return sentences

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
        assert self.g, "Graph must be created. Please execute create_graph"
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
        print("Hubs:", hubs_list)
        for hub in hubs_list:
            node = hub[0]
            # Only search for nodes that are root of noun chunks
            if self.node_dict.get(node):
                sentences = self.node_dict[node]
                n_hubs_dict[node] = sentences
            # print('Node {}: {}'.format(node, sentences))
        return n_hubs_dict


def _homogeneous_summary(hubs_sentences, n_sentences_summary):
    """
    Performs an homogeneous summary over sentences, with a given length. It iterates through hubs, extracting all
    of the sentences of each hub, until the summary is completed.
    :param hubs_sentences:
    :param n_sentences_summary:
    :return:
    """
    summary_set = set()
    summary_list = list()
    n_sentences_show = 0
    for node, list_pos_sent in hubs_sentences.items():
        # print(node)
        for pos_sent in list_pos_sent:
            # print(pos_sent)
            if n_sentences_show < n_sentences_summary and pos_sent not in summary_set:
                summary_set.add(pos_sent)
                summary_list.append(pos_sent)
                n_sentences_show += 1
            else:
                continue
    summary = '\n'.join(sent[1] for sent in sorted(summary_list))
    return summary


def _heterogeneous_summary(hubs_sentences, n_sentences_summary, hub_percentage=0.1):
    """
    Performs an heterogeneous summary over sentences, with a given length. It iterates through hubs, extracting a
    percentage of sentences of each hub. If all hubs are traversed and still more sentences are needed, it starts
    again from the first hub.
    :param hubs_sentences:
    :param n_sentences_summary:
    :param hub_percentage: % of sentences to grab from each hub
    :return:
    """
    assert 0 < hub_percentage <= 1, "Hub percentage must be greater than 0 and less or equal than 1"
    summary_set = set()
    summary_list = list()
    n_sentences_show, summary_list = _iterate_heterogeneous_hubs(hubs_sentences, n_sentences_summary,
                                                                 summary_list, summary_set, hub_percentage)
    summary = '\n'.join(sent[1] for sent in sorted(summary_list))
    return summary


def _iterate_heterogeneous_hubs(hubs_sentences, n_sentences_summary,
                                summary_list, summary_set, hub_percentage):
    """
    Recursive function that allows to traverse a dictionary of nodes/sentences, extracting a number of sentences from each,
    until the summary is completed. If the last hub is reached and still more sentences are needed, it starts again from
    the start.
    :param hubs_sentences:
    :param n_sentences_summary:
    :param summary_list:
    :param summary_set:
    :param hub_percentage:
    :return:
    """
    n_sentences_show = 0
    for node, list_pos_sent in hubs_sentences.items():
        n_sentences_hub = round(len(list_pos_sent) * hub_percentage)
        # For nodes with little number of sentences (testing with short texts)
        if n_sentences_hub == 0:
            n_sentences_hub = 1
        # print('Including {} sentences from node {}'.format(n_sentences_hub, node))
        iterated_sentences_hub = 0
        for pos_sent in list_pos_sent:
            # print('Shown', n_sentences_show)
            if n_sentences_show >= n_sentences_summary:
                break
            elif pos_sent not in summary_set and iterated_sentences_hub < n_sentences_hub:
                summary_set.add(pos_sent)
                summary_list.append(pos_sent)
                n_sentences_show += 1
                # print('Adding {} sentence from node {}'.format(pos_sent, node))
                iterated_sentences_hub += 1
            else:
                continue
    if n_sentences_show < n_sentences_summary:
        n_sentences_left = n_sentences_summary - n_sentences_show
        # Update hubs_sentences dict for next round
        for node, list_pos_sent in hubs_sentences.items():
            new_list_pos_sent = [pos_sent for pos_sent in list_pos_sent if
                                 pos_sent not in summary_set]
            hubs_sentences[node] = new_list_pos_sent
        n_sentences_show, summary_list = _iterate_heterogeneous_hubs(hubs_sentences, n_sentences_left,
                                                                     summary_list, summary_set, hub_percentage)
    return n_sentences_show, summary_list


def _get_n_sentences_from_hubs(hubs_sentences):
    """
    Calculate total number of different sentences given a dictionary of nodes-> sentences.
    :param hubs_sentences:
    :return:
    """
    summary_set = set()
    for node, list_pos_sent in hubs_sentences.items():
        for pos_sent in list_pos_sent:
            summary_set.add(pos_sent)
    return len(summary_set)


def get_summary_from_text(text, n_hubs, fc, mode='homogeneous', hub_percentage=0.1):
    """
    Performs a summary from the text, selecting the n_hubs nodes with more edges from a semantic graph, with a compression
    factor of fc.
    :param text: text read from .txt
    :param fc: compression factor, indicates the percentage of the text to be shown in the summary
    :param n_hubs: number of nodes of the semantic graph used to build the summary
    :param mode: homogeneous or heterogeneous
    :param hub_percentage: only for heterogeneous, indicates the percentage of sentences selected from each hub
    :return:
    """
    assert 0 <= fc <= 1, "Compression factor must be between 0 and 1"
    # if fc == 1:
    #     print('Returning the whole text')
    #     return text
    # elif fc == 0:
    #     print('Returning an empty summary')
    #     return ''
    # Initialize and create graph
    semantic_graph = SemanticGraph(text)
    g = semantic_graph.create_graph()
    n_nodes = len(g.nodes)
    assert n_nodes >= n_hubs, "Hubs must be lower than the total number of nodes. This graph has {} nodes".format(n_nodes)
    # Get n hubs with more edges
    hubs_sentences = semantic_graph.get_n_hubs_sentences(n=n_hubs)
    print("Hubs with sentences:", {hub: len(v) for hub, v in hubs_sentences.items()})
    # Number of sentences to show
    n_sentences_text = semantic_graph.n_sentences
    n_sentences_summary = round(n_sentences_text * fc)
    n_sentences_hubs = _get_n_sentences_from_hubs(hubs_sentences)
    total_n_sentences_text = min(n_sentences_hubs, n_sentences_summary)
    print('The text has {} sentences'.format(n_sentences_text))
    print("The semantic graph has {} nodes".format(n_nodes))
    print('The summary should have {} sentences with a compression factor of {}'.format(n_sentences_summary, fc))
    print('There are {} sentences in the {} nodes with more degree'.format(n_sentences_hubs, n_hubs))

    if mode == 'homogeneous':
        summary = _homogeneous_summary(hubs_sentences, total_n_sentences_text)
    elif mode == 'heterogeneous':
        summary = _heterogeneous_summary(hubs_sentences, total_n_sentences_text, hub_percentage)
    else:
        raise ValueError("Mode must be one of: homogeneous, heterogeneous")
    # print('Showing {} sentences out of {}, with a compression factor of {}'.format(len(summary.split('\n')), n_sentences_text, fc))
    print('Showing {} sentences'.format(len(summary.split('\n'))))
    return summary


if __name__ == "__main__":
    text_path = 'texts/lor.txt'
    summary_path = 'summary.txt'
    # Read text
    with open(text_path, 'r', encoding='utf-8', newline='') as f:
        text = f.read()

    proc_text = ' '.join(text.split('\n'))
    summary = get_summary_from_text(proc_text, n_hubs=50, fc=0.1, mode='homogeneous')
    print("Summary\n")
    print(summary)
    with open(summary_path, "w") as summary_file:
        summary_file.write(summary)