from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.text.teams_players import TeamPlayers

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scripts.conf import TEAMS

import warnings
import unidecode
from typing import List, Dict, Tuple, Optional
import pandas as pd
from functools import reduce
from collections import Counter


class KeyEventsSummary:

    def __init__(self, key_events: List[str]):
        """
        :param key_events: list of key events to consider. For example: ['goal', 'var']
        """
        self.key_events = key_events
        self.processor = ArticleTextProcessor()
        self.text_proc = BasicTextProcessor()
        self.team_players = TeamPlayers()
        # Necessary to keep track of chosen events
        self.events_mapping_list = list()
        # Will be updated with the different articles of each season
        self.league_season_teams = None
        # Players involved in a match
        self.match_players = None

    @staticmethod
    def _filter_red_cards(token_list: List[str]) -> bool:
        return 'red' in token_list and 'card' in token_list

    def _clean_tokens(self, doc):
        return [token.text.lower() for token in doc if self.text_proc.token_filter(token)
                and self.text_proc.filter_noisy_characters(token)
                and not self.text_proc.has_numbers(token) and self.text_proc.token_filter_stopword(token)]

    def _process_match_text(self, text: str, text_type: str = 'event'):
        """
        Cleans tokens and retrieve entity names for a given text
        :param text:
        :param text_type:
        :return:
        """
        if text_type == 'event':
            doc = self.text_proc.token_list(text)
        elif text_type == 'article':
            doc = text
        else:
            raise ValueError("text_type available values are event and article")

        # Token cleaning
        clean_tokens = self._clean_tokens(doc)
        # Entity names
        en_text = self.text_proc.entity_names_labels(text)
        en_proc = self.team_players.team_players_event(en_text, self.league_season_teams)
        en_list = [event_proc[0] for event_proc in en_proc]
        # print(en_list)
        # Save players
        if not self.match_players:
            self.match_players = list()
        self.match_players.extend([en for en in en_list if en in self.team_players.players_set])
        # Filter repeated words
        filtered_tokens = [t for t in clean_tokens if not any([t in en.lower() for en in en_list])]
        # Join
        tokens_en = filtered_tokens + en_list
        # Strip accents
        tokens_en = [unidecode.unidecode(t) for t in tokens_en]
        return tokens_en

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
            tokens_en = self._process_match_text(event)
            if any(player in tokens_en for player in top_players) and ix_event not in processed_events_dict.keys():
                processed_events_dict[ix_event] = ' '.join(set(tokens_en))
                n_processed_events = self._update_event_mapping(ix_event, n_processed_events)
        # Required if we reorder using new players' events
        self.events_mapping_list = sorted(self.events_mapping_list)
        return processed_events_dict

    def process_match_events(self, events: List[str], keep_key_events: bool = False,
                             keep_top_players: Optional[int] = None,
                             league_season_teams: Optional[str] = None) -> List[str]:
        """
        Processes a list of match events, cleaning tokens and identifying entity names.

        :param league_season_teams:
        :param events:
        :param keep_key_events: only keep events with key words
        :param keep_top_players: whether to keep events of N most mentioned players. This value is the number of
        most mentioned players.
        :return:
        """
        if not self.league_season_teams and league_season_teams:
            print('league_season_teams is empty. Initializing it to', league_season_teams)
            self.league_season_teams = league_season_teams
        if not self.league_season_teams:
            raise ValueError('league_season_teams is empty')

        processed_events = dict()
        n_processed_events = 0
        self.events_mapping_list = list()
        for ix_event, event in enumerate(events):
            tokens_en = self._process_match_text(event)
            if keep_key_events:
                # print(tokens_en)
                if any(key_event in tokens_en for key_event in self.key_events) or self._filter_red_cards(tokens_en):
                    processed_events[ix_event] = ' '.join(set(tokens_en))
                    # We save the mapping between event index and new processed event index
                    n_processed_events = self._update_event_mapping(ix_event, n_processed_events)
                else:
                    continue
            else:
                processed_events[ix_event] = ' '.join(set(tokens_en))
                n_processed_events = self._update_event_mapping(ix_event, n_processed_events)

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

    def process_match_article(self, article: str) -> List[str]:
        doc_sents = self.text_proc.get_sentences(article)
        processed_sentences = list()
        for sentence in doc_sents:
            tokens_en = self._process_match_text(sentence, text_type='article')
            # print(tokens_en)
            processed_sentences.append(' '.join(set(tokens_en)))
        return processed_sentences

    def match_summary(self, match_dict: Dict, print_relations: bool = False,
                      keep_top_players: Optional[int] = None, **count_vec_kwargs) -> Tuple[str, List]:
        """
        Performs a summary based on key events. match_dict contains both events and article body from a match. Events
        and article text are processed, and then cosine distances are computed between a key event and each of the text's
        sentences. Therefore, the summary is built with the nearest sentences in the text for each key event.
        :param match_dict:
        :param print_relations:
        :param keep_top_players:
        :param count_vec_kwargs: feeded to CountVectorizer
        :return:
        """
        processed_events = self.process_match_events(match_dict['events'], keep_key_events=True,
                                                     keep_top_players=keep_top_players)
        processed_article_sentences = self.process_match_article(match_dict['article'])
        if len(processed_events) == 0 or len(processed_article_sentences) == 0:
            warnings.warn('Could not perform tfidf')
            return '', list()
        # Train tfidf with article sentences
        vectorizer = CountVectorizer(**count_vec_kwargs)
        X = vectorizer.fit_transform(processed_article_sentences).toarray()
        tfidfconverter = TfidfTransformer()
        X = tfidfconverter.fit_transform(X).toarray()
        # Events
        X_events = vectorizer.transform(processed_events)
        X_events = tfidfconverter.transform(X_events).toarray()
        # Distances
        distances = cosine_similarity(X_events, X)
        sentences_ixs = distances.argmax(axis=1)
        # We need to save article's sentences to build the summary
        article_sents_list = [str(sent) for sent in self.text_proc.get_sentences(match_dict['article'])]
        summary_sents_list = [article_sents_list[ix] for ix in sorted(list(set(sentences_ixs)))]
        article_summary = ''.join(summary_sents_list)
        print('Number of sentences in original article:', len(article_sents_list))
        print('Number of sentences in summary:', len(summary_sents_list))
        if print_relations:
            for event_ix, sentence_ix in enumerate(sentences_ixs):
                print('Event:')
                print(match_dict['events'][self.events_mapping_list[event_ix]])
                print('Nearest sentence in article:')
                print(article_sents_list[sentence_ix])
        return article_summary, sentences_ixs

    def run(self):
        """
        Performs a summary for every available match.
        :return:
        """
        all_files = self.processor.load_json()
        list_pd_matches = list()
        for season_file, season_values in all_files.items():
            self.league_season_teams = TEAMS[season_file.split('.')[0]]
            for match_url, match_dict in season_values.items():
                summary, sentences_ixs = self.match_summary(match_dict, ngram_range=(1, 2), strip_accents='unicode')
                if len(summary) == 0 and len(sentences_ixs) == 0:
                    print('Could not perform summary for {}'.format(match_url))
                    continue

                # Horrible pero es lo único que funciona para meter una lista en un df
                pd_summary = pd.DataFrame(columns=['season_file', 'match_url', 'summary', 'article_sentences_ix', 'events_mapping'])
                pd_summary.loc[0, 'json_file'] = season_file
                pd_summary.loc[0, 'url'] = match_url
                pd_summary.loc[0, 'summary'] = summary
                pd_summary.loc[0, 'article_sentences_ix'] = sentences_ixs
                pd_summary.loc[0, 'events_mapping'] = self.events_mapping_list

                list_pd_matches.append(pd_summary)
        pd_df_matches = reduce(lambda df1, df2: pd.concat([df1, df2]), list_pd_matches)
        return pd_df_matches
