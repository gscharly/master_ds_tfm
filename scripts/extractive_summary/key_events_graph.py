from scripts.text.semantic_graph import SemanticGraph
from .key_events import KeyEvents

from scripts.conf import TEAMS

from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from functools import reduce
import warnings

import pandas as pd


class KeyEventsSummaryGraph(KeyEvents):

    def __init__(self, drop_teams: bool = False, lemma: bool = False):
        super().__init__(drop_teams=drop_teams, lemma=lemma)
        # Event-sentence relations
        self.global_event_sentence_dict = dict()

    def _join_summary(self, summary_pos_list: List[Tuple[int, str]]) -> str:
        summary_list = list()
        for pos_sent in sorted(summary_pos_list):
            # Create summary and update mapping with real events
            self.events_mapping_list.append(pos_sent[0])
            summary_list.append(pos_sent[1])
        summary = '\n'.join(summary_list)
        return summary

    def _homogeneous_summary(self, hubs_sentences, n_sentences_summary) -> str:
        """
        Performs an homogeneous summary over sentences, with a given length. It iterates through hubs, extracting all
        of the sentences of each hub, until the summary is completed.
        :param hubs_sentences:
        :param n_sentences_summary:
        :return:
        """
        summary_set = set()
        summary_pos_list = list()
        n_sentences_show = 0
        for node, list_pos_sent in hubs_sentences.items():
            # print(node)
            for pos_sent in list_pos_sent:
                # print(pos_sent)
                if n_sentences_show < n_sentences_summary and pos_sent not in summary_set:
                    summary_set.add(pos_sent)
                    summary_pos_list.append(pos_sent)
                    n_sentences_show += 1
                else:
                    continue
        return self._join_summary(summary_pos_list)

    def _heterogeneous_summary(self, hubs_sentences, n_sentences_summary, hub_percentage=0.1):
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
        summary_pos_list = list()
        n_sentences_show, summary_pos_list = self._iterate_heterogeneous_hubs(hubs_sentences, n_sentences_summary,
                                                                              summary_pos_list, summary_set,
                                                                              hub_percentage)
        return self._join_summary(summary_pos_list)

    def _iterate_heterogeneous_hubs(self, hubs_sentences, n_sentences_summary,
                                    summary_list, summary_set, hub_percentage):
        """
        Recursive function that allows to traverse a dictionary of nodes/sentences, extracting a number of sentences
        from each, until the summary is completed. If the last hub is reached and still more sentences are needed,
        it starts again from the start.
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
            n_sentences_show, summary_list = self._iterate_heterogeneous_hubs(hubs_sentences, n_sentences_left,
                                                                              summary_list, summary_set, hub_percentage)
        return n_sentences_show, summary_list

    @staticmethod
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

    def process_match_events(self, events: List[str], n_hubs: int = 10, fc: float = 0.5, mode: str = 'homogeneous',
                             hub_percentage: Optional[float] = 0.1) -> List[str]:
        """
        Performs a summary from the events, selecting the n_hubs nodes with more edges from a semantic graph, with a
        compression factor of fc.
        :param events:
        :param fc: compression factor, indicates the percentage of the text to be shown in the summary
        :param n_hubs: number of nodes of the semantic graph used to build the summary
        :param mode: homogeneous or heterogeneous
        :param hub_percentage: only for heterogeneous, indicates the percentage of sentences selected from each hub
        :return:
        """
        assert 0 <= fc <= 1, "Compression factor must be between 0 and 1"
        if fc == 1:
            print('Returning the whole text')
            return events
        elif fc == 0:
            print('Returning an empty summary')
            return list()
        # Initialize and create graph
        semantic_graph = SemanticGraph(events)
        g = semantic_graph.create_graph()
        n_nodes = len(g.nodes)

        if n_nodes < n_hubs:
            warnings.warn('Required hubs are higher than available nodes. Using {} hubs'.format(n_nodes))
            n_hubs = n_nodes

        # Get n hubs with more edges
        hubs_sentences = semantic_graph.get_n_hubs_sentences(n=n_hubs)

        if not hubs_sentences:
            warnings.warn("Hubs sentences is empty. Returning empty list")
            return list()

        print("Hubs with sentences:", {hub: len(v) for hub, v in hubs_sentences.items()})
        # Number of sentences to show
        n_sentences_text = semantic_graph.n_sentences
        n_sentences_summary = round(n_sentences_text * fc)
        n_sentences_hubs = self._get_n_sentences_from_hubs(hubs_sentences)
        total_n_sentences_text = min(n_sentences_hubs, n_sentences_summary)
        print('The text has {} events'.format(n_sentences_text))
        print("The semantic graph has {} nodes".format(n_nodes))
        print('The summary should have {} sentences with a compression factor of {}'.format(n_sentences_summary, fc))
        print('There are {} sentences in the {} nodes with more degree'.format(n_sentences_hubs, n_hubs))

        self.events_mapping_list = list()

        if mode == 'homogeneous':
            summary = self._homogeneous_summary(hubs_sentences, total_n_sentences_text)
        elif mode == 'heterogeneous':
            summary = self._heterogeneous_summary(hubs_sentences, total_n_sentences_text, hub_percentage)
        else:
            raise ValueError("Mode must be one of: homogeneous, heterogeneous")
        summary_list = summary.split('\n')
        print('Number of original events:', len(events))
        print('Number of processed events:', len(summary_list))

        return [' '.join(self.process_match_text(sum_event)) for sum_event in summary_list]

    def match_summary(self, match_dict: Dict, count_vec_kwargs: Dict, save_relations: bool = False,
                      verbose=False, **key_events_properties) -> Dict:
        """
        Performs a summary based on key events. These key events are selected performing an extractive summary of the
        events of a match.

        match_dict contains both events and article body from a match. Events and article text are processed,
        and then cosine distances are computed between a key event and each of the text's sentences.
        Therefore, the summary is built with the nearest sentences in the text for each key event.
        :param match_dict:
        :param save_relations: whether to save event-sentence relation while performing the summary
        :param verbose:
        :param count_vec_kwargs: feeded to CountVectorizer
        :return:
        """
        match_summary_info = self._match_summary(match_dict, count_vec_kwargs, **key_events_properties)
        if save_relations and match_summary_info:
            for event_ix, sentence_ix in enumerate(match_summary_info['sentences_ixs']):
                event = match_dict['events'][self.events_mapping_list[event_ix]]
                sentence = match_summary_info['article_sents_list'][sentence_ix]
                self.global_event_sentence_dict[event] = sentence
                if verbose:
                    print('Event:')
                    print(event)
                    print('Nearest sentence in article:')
                    print(sentence)
        return match_summary_info

    def run(self, save_events_sentences: bool, path_csv: str, path_mapping: str, count_vec_kwargs: Dict,
            **key_events_properties):
        """
        Performs a summary for every available match. If save_events_sentences is True, a dictionary will be created
        with mappings between key events and article sentences. This dictionary will have the following structure:
        {event: sentence, ...}
        :param save_events_sentences:
        :param path_csv: path to save the summaries
        :param path_mapping: path to save event-article mappings
        :param count_vec_kwargs:
        :return:
        """
        all_files = self.processor.load_json()
        list_pd_matches = list()
        for season_file, season_values in tqdm(all_files.items()):
            self.league_season_teams = TEAMS[season_file.split('.')[0]]
            for match_url, match_dict in season_values.items():
                if len(match_dict['events']) == 0 or len(match_dict['article']) == 0:
                    print('Could not perform summary for {}'.format(match_url))
                    continue

                match_summary_info = self.match_summary(match_dict, count_vec_kwargs,
                                                        save_relations=save_events_sentences,
                                                        **key_events_properties)
                if not match_summary_info:
                    warnings.warn('Could not perform summary for {}'.format(match_url))
                    continue
                elif len(match_summary_info['article_summary']) == 0 and len(match_summary_info['sentences_ixs']) == 0:
                    warnings.warn('Could not perform summary for {}'.format(match_url))
                    continue

                # Horrible pero es lo único que funciona para meter una lista en un df
                pd_summary = pd.DataFrame(columns=['json_file', 'url', 'summary', 'article_sentences_ix',
                                                   'article_sentences', 'summary_events', 'events_mapping'])
                pd_summary.loc[0, 'json_file'] = season_file
                pd_summary.loc[0, 'url'] = match_url
                pd_summary.loc[0, 'summary'] = match_summary_info['article_summary']
                pd_summary.loc[0, 'article_sentences_ix'] = match_summary_info['sentences_ixs']
                pd_summary.loc[0, 'article_sentences'] = match_summary_info['article_sents_list']
                pd_summary.loc[0, 'summary_events'] = list(map(match_dict['events'].__getitem__,
                                                               self.events_mapping_list))
                pd_summary.loc[0, 'events_mapping'] = self.events_mapping_list

                list_pd_matches.append(pd_summary)
        pd_df_matches = reduce(lambda df1, df2: pd.concat([df1, df2]), list_pd_matches)
        if path_csv:
            print('Saving summaries in', path_csv)
            pd_df_matches.to_csv(path_csv, index=False)
        if path_mapping:
            print('Saving mappings in', path_mapping)
            map_dict = {'event': list(), 'article_sentence': list()}
            for event, article_sentence in self.global_event_sentence_dict.items():
                map_dict['event'].append(event)
                map_dict['article_sentence'].append(article_sentence)
            pd_map = pd.DataFrame(data=map_dict)
            pd_map.to_csv(path_mapping, index=False)

        return pd_df_matches
