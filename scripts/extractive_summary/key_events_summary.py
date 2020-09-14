from .key_events import KeyEvents

from scripts.conf import TEAMS

import warnings
from typing import List, Dict, Optional
import pandas as pd
from functools import reduce
from collections import Counter
from tqdm import tqdm


class KeyEventsSummary(KeyEvents):
    RED_CARD = 'red_card'
    GOAL = 'goal'
    ATTEMPT = 'attempt'
    SPECIAL_EVENTS = [GOAL, RED_CARD]

    def __init__(self, key_events: List[str], drop_teams: bool = False, lemma: bool = False):
        """
        :param key_events: list of key events to consider. For example: ['goal', 'var']. Red cards need special token:
        red_card
        """
        super().__init__(drop_teams=drop_teams, lemma=lemma)
        self.key_events = key_events
        # Event-sentence relations
        self.event_sentence_dict = dict()
        self.global_event_sentence_dict = {key_event: dict() for key_event in self.key_events}

    @staticmethod
    def filter_red_cards(token_list: List[str]) -> bool:
        return 'red' in token_list and 'card' in token_list

    def filter_attempts_in_goals(self, token_list: List[str]) -> bool:
        return self.GOAL in token_list and self.ATTEMPT not in token_list

    def _update_event_mapping(self, ix_event, n_processed):
        self.events_mapping_list.append(ix_event)
        n_processed += 1
        return n_processed

    def _add_top_players_events(self, events: List[str], processed_events_dict: Dict, n_players: int):
        """
        Add players events
        :param events:
        :param processed_events_dict:
        :param n_players:
        :return:
        """
        players_appearences = Counter(self.match_players)
        top_players = [pl[0] for pl in players_appearences.most_common(n_players)]
        print('Adding events for', top_players)
        n_processed_events = 0
        for ix_event, event in enumerate(events):
            tokens_en = self.process_match_text(event)
            if any(player in tokens_en for player in top_players) and ix_event not in processed_events_dict.keys():
                processed_events_dict[ix_event] = ' '.join(tokens_en)
                n_processed_events = self._update_event_mapping(ix_event, n_processed_events)
        # Required if we reorder using new players' events
        self.events_mapping_list = sorted(self.events_mapping_list)
        return processed_events_dict

    def _save_key_event(self, processed_events, tokens_en, ix_event, n_processed_events, event=None, key_event=None):
        """
        Saves necessary information. It updates the mapping between processed events and real events, and optionally
        initializes a dictionary that will contain the mapping between events and article sentences.
        :param processed_events:
        :param tokens_en:
        :param ix_event:
        :param n_processed_events:
        :param event:
        :param key_event:
        :return:
        """
        processed_events[ix_event] = ' '.join(tokens_en)
        # We save the mapping between event index and new processed event index
        n_processed_events = self._update_event_mapping(ix_event, n_processed_events)
        if event:
            # We save the event index attached to the key event
            self.event_sentence_dict[key_event].update({ix_event: None})
        return n_processed_events

    def process_match_events(self, events: List[str], keep_key_events: bool = False,
                             keep_top_players: Optional[int] = None) -> List[str]:
        """
        Processes a list of match events, cleaning tokens and identifying entity names.

        :param events:
        :param keep_key_events: only keep events with key words
        :param keep_top_players: whether to keep events of N most mentioned players. This value is the number of
        most mentioned players.
        :return:
        """
        self._check_league_season_teams()

        processed_events = dict()
        n_processed_events = 0
        self.events_mapping_list = list()
        self.event_sentence_dict = {key_event: dict() for key_event in self.key_events}
        for ix_event, event in enumerate(events):
            tokens_en = self.process_match_text(event)
            if keep_key_events:
                # print(tokens_en)
                # if any(key_event in tokens_en for key_event in self.key_events) or self._filter_red_cards(tokens_en):
                # Look for key events
                for key_event in self.key_events:
                    # Do not include attempts that have the word "goal" (referido a porteria, no a gol)
                    if key_event == self.GOAL and self.filter_attempts_in_goals(tokens_en):
                        self._save_key_event(processed_events, tokens_en, ix_event, n_processed_events, event,
                                             key_event)
                        break
                    # Look for red cards if no key events are found (red cards need special treatment)
                    elif key_event == self.RED_CARD and self.filter_red_cards(tokens_en):
                        self._save_key_event(processed_events, tokens_en, ix_event, n_processed_events, event,
                                             key_event)
                        break
                    elif key_event in tokens_en and key_event not in self.SPECIAL_EVENTS:
                        self._save_key_event(processed_events, tokens_en, ix_event, n_processed_events, event,
                                             key_event)
                        break
                    else:
                        continue
            else:
                self._save_key_event(processed_events, tokens_en, ix_event, n_processed_events)

        # Only do this if we are keeping a subset of events
        if keep_top_players and keep_key_events:
            old_length = len(processed_events)
            processed_events = self._add_top_players_events(events, processed_events, keep_top_players)
            new_length = len(processed_events)
            if old_length != new_length:
                print('Added {} new events for {} player(s)'.format(new_length - old_length, keep_top_players))

        if len(events) != len(processed_events) and not keep_key_events:
            warnings.warn('Length of generated events list is different from original. Some events have been lost')

        print('Number of original events:', len(events))
        print('Number of processed events:', len(processed_events))

        # Return sorted events
        # print(processed_events)
        return [it[1] for it in sorted(processed_events.items())]

    def match_summary(self, match_dict: Dict, count_vec_kwargs: Dict, save_relations: bool = False,
                      verbose=False, **key_events_properties) -> Dict:
        """
        Performs a summary based on key events. match_dict contains both events and article body from a match. Events
        and article text are processed, and then cosine distances are computed between a key event and each of
        the text's sentences. Therefore, the summary is built with the nearest sentences in the text for each key event.
        :param match_dict:
        :param save_relations: whether to save event-sentence relation while performing the summary
        :param verbose:
        :param count_vec_kwargs: feeded to CountVectorizer
        :param key_events_properties: properties for process_match_events
        :return:
        """

        match_summary_info = self._match_summary(match_dict, count_vec_kwargs, **key_events_properties)
        if save_relations and match_summary_info:
            for event_ix, sentence_ix in enumerate(match_summary_info['sentences_ixs']):
                event = match_dict['events'][self.events_mapping_list[event_ix]]
                sentence = match_summary_info['article_sents_list'][sentence_ix]
                for event_type, events_ixs_dict in self.event_sentence_dict.items():
                    # Events and article sentences are stored for an article
                    if self.events_mapping_list[event_ix] in events_ixs_dict.keys():
                        self.global_event_sentence_dict[event_type][event] = sentence
                if verbose:
                    print('Event:')
                    print(event)
                    print('Nearest sentence in article:')
                    print(sentence)
        return match_summary_info

    def run(self, save_events_sentences: bool, path_csv: str,
            path_mapping: str, count_vec_kwargs: Dict, **key_events_properties):
        """
        Performs a summary for every available match. If save_events_sentences is True, a dictionary will be created
        with mappings between key events and article sentences. This dictionary will have the following structure:
        {event_type: {event: sentence, ...}, ...}
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
                pd_summary.loc[0, 'summary_events'] = ' '.join(list(map(match_dict['events'].__getitem__,
                                                              self.events_mapping_list)))
                pd_summary.loc[0, 'events_mapping'] = self.events_mapping_list

                list_pd_matches.append(pd_summary)
        pd_df_matches = reduce(lambda df1, df2: pd.concat([df1, df2]), list_pd_matches)
        if path_csv:
            print('Saving summaries in', path_csv)
            pd_df_matches.to_csv(path_csv, index=False)
        if path_mapping:
            print('Saving mappings in', path_mapping)
            map_dict = {'event_type': list(), 'event': list(), 'article_sentence': list()}
            for event_type, mapping_dict in self.global_event_sentence_dict.items():
                for event, article_sentence in mapping_dict.items():
                    map_dict['event_type'].append(event_type)
                    map_dict['event'].append(event)
                    map_dict['article_sentence'].append(article_sentence)
            pd_map = pd.DataFrame(data=map_dict)
            pd_map.to_csv(path_mapping, index=False)

        return pd_df_matches
