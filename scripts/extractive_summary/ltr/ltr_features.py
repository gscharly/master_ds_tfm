# Scripts
from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.extractive_summary.key_events_summary import KeyEventsSummary
from scripts.extractive_summary.ltr.learn_to_rank import LearnToRank

# sklearn stuff
from sklearn.metrics.pairwise import cosine_similarity

# DS imports
import numpy as np
import pandas as pd

# Other imports
from typing import Dict, List, Optional
from collections import OrderedDict, Counter
import warnings
import os


class LTRFeatures(LearnToRank):
    LTR_TYPE = 'features'

    def __init__(self, key_events: List[str], count_vec_kwargs: Dict, lags: List[int] = None, drop_teams: bool = False,
                 lemma: bool = False, processor: Optional[ArticleTextProcessor] = None):
        self.processor = processor if processor else ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma)
        super().__init__(processor=self.processor)

        self.key_events_sum = KeyEventsSummary(key_events=key_events)
        self.text_proc = BasicTextProcessor()
        self.lemma = lemma
        self.drop_teams = drop_teams
        self.key_events = key_events
        self.count_vec_kwargs = count_vec_kwargs

        if lags is None:
            self.lags = [1, 3, 5]
        else:
            self.lags = lags

    def config(self) -> Dict:
        return {
            'key_events': self.key_events,
            'lags': self.lags,
            'drop_teams': self.drop_teams,
            'lemma': self.lemma,
            'count_vec_kwargs': self.count_vec_kwargs
        }

    @property
    def ltr_type(self) -> str:
        return self.LTR_TYPE

    @property
    def file_path(self) -> str:
        return '{}/{}.csv'.format(self.path, self.LTR_TYPE)

    def _is_key_event(self, proc_event: List[str]) -> int:
        """
        Returns 1 is a key word is present in a processed event; otherwise returns 0
        :param proc_event:
        :return:
        """
        for key_event in self.key_events_sum.key_events:
            # Do not include attempts that have the word "goal" (referido a porteria, no a gol)
            if key_event == self.key_events_sum.GOAL and self.key_events_sum.filter_attempts_in_goals(proc_event):
                return 1
            # Look for red cards if no key events are found (red cards need special treatment)
            elif key_event == self.key_events_sum.RED_CARD and self.key_events_sum.filter_red_cards(proc_event):
                return 1
            elif key_event in proc_event and key_event not in self.key_events_sum.SPECIAL_EVENTS:
                return 1
            else:
                return 0

    def _get_players(self, event: str) -> List:
        en_text = self.text_proc.entity_names_labels(event)
        en_proc = self.processor.team_players.team_players_event(en_text, self.processor.league_season_teams)
        player_list = [t for t in en_proc if t[1] == 'PLAYER']
        return player_list

    def _event_changes(self, event_doc: str, proc_event: List[str]):
        """
        Returns a dict if an event has a goal, indicating if the goal gives advantage to a team or equalizes the match.
        If the event doesn't contain a goal, an empty dict will be returned.
        :param event_doc:
        :param proc_event:
        :return:
        """
        change_dict = {
            'equalize': 0,
            'advantage': 0
        }
        if self.key_events_sum.filter_attempts_in_goals(proc_event):
            result_list = self.text_proc.get_numbers(event_doc)
            if len(result_list) != 2:
                warnings.warn("There are more than 2 results")
                print(result_list)
                return change_dict
            if result_list[0] == result_list[1]:
                change_dict['equalize'] += 1
            else:
                change_dict['advantage'] += 1

        return change_dict

    @staticmethod
    def _lag_similarities(x: np.array, lags: List[int]) -> np.array:
        """
        Calculate similarities with lagged events.
        :param x:
        :param lags:
        :return:
        """
        cos_sim_matrix = cosine_similarity(x, x)
        event_lag_matrix = np.zeros(shape=(x.shape[0], len(lags)))
        for i in range(x.shape[0]):
            for lag_ix, lag in enumerate(lags):
                if i < lag:
                    # print(i, 0)
                    event_lag_matrix[i][lag_ix] = 0
                else:
                    # print(i, i-lag)
                    # print(cos_sim_matrix[i][i - lag])
                    event_lag_matrix[i][lag_ix] = cos_sim_matrix[i][i-lag]
        return event_lag_matrix

    def _event_level_features(self, event: str, proc_event: str, players_importance_dict: Dict) -> OrderedDict:
        """
        Create features for an event, only using information from that event. The following features are created:
        - Event length (without stopwords)
        - Number of stopwords
        - Key events
        - Number of players
        - Changes in result
        :param event:
        :param proc_event: processed event from before, so that we don't need to preprocess it again.
        :return:
        """
        event_feature_dic = OrderedDict()
        event_doc = self.text_proc.token_list(event)
        # Event length
        proc_event_list = proc_event.split(' ')
        event_feature_dic['length'] = len(proc_event_list)
        # Number of stopwords
        event_feature_dic['n_stop'] = self.text_proc.count_stopwords(event_doc)
        # Key events
        proc_event_list = proc_event.split(' ')
        event_feature_dic['is_key_event'] = self._is_key_event(proc_event_list)
        # Number of players
        players_list = self._get_players(event)
        event_feature_dic['n_players'] = len(players_list)
        # Players importance
        event_feature_dic['players_importance'] = sum([players_importance_dict[player_tuple[0]]
                                                       for player_tuple in players_list])
        # Result changes
        changes_dict = self._event_changes(event_doc, proc_event_list)
        event_feature_dic['advantage'] = changes_dict['advantage']
        event_feature_dic['equalize'] = changes_dict['equalize']

        return event_feature_dic

    def _get_players_importance(self) -> Dict:
        """
        Returns a dict with the percentage of events where each player appears.
        :return:
        """
        players_appearences = Counter(self.processor.match_players)
        n_events = sum(players_appearences.values())
        return {player_tuple[0]: player_tuple[1]/n_events for player_tuple in players_appearences.most_common()}

    def _match_level_features(self, events: List[str]) -> Dict:
        """
        This function calculates and returns the following features:
        - Sum of TFIDF weights for every event
        - Similarity with previous events
        - Players importances
        It also returns the processed events, so that we don't have to process them again in event level stage.
        :param events:
        :return:
        """
        processed_events = [' '.join(self.processor.process_match_text(event)) for event in events]
        # TFIDF sum
        tfidf_dict = self.text_proc.train_tfidf(processed_events, **self.count_vec_kwargs)
        tfidf_sum_list = np.sum(tfidf_dict['x'], axis=1)
        assert len(tfidf_sum_list) == len(events), "Length of events does not match length of tfidf"
        # Similarities with lagged events
        sim_lag_matrix = self._lag_similarities(tfidf_dict['x'], self.lags)
        assert sim_lag_matrix.shape == (len(events), len(self.lags)),\
            "Length of sim matrix does not match".format((len(events), len(self.lags)))
        # Players importance
        players_importance = self._get_players_importance()

        return {'tfidf_sum': tfidf_sum_list, 'sim_lag': sim_lag_matrix, 'players_importance': players_importance,
                'processed_events': processed_events}

    def _add_lags_to_event(self, features_dict: Dict, match_level_features: Dict, event_pos: int):
        for lag_ix, lag in enumerate(self.lags):
            features_dict['sim_previous_{}'.format(lag)] = match_level_features['sim_lag'][event_pos][lag_ix]
        return features_dict

    @staticmethod
    def _update_match_dict(event_feature_dict: Dict, all_events_feature_dict: Dict = None) -> Dict:
        if not all_events_feature_dict:
            all_events_feature_dict = {k: list() for k in event_feature_dict.keys()}

        for k, v in event_feature_dict.items():
            all_events_feature_dict[k].append(v)
        return all_events_feature_dict

    def create_features(self, match_dict: Dict, league_season_teams: Optional[str]) -> Dict:
        """
        Create features for every event. It returns a Dict, where each key is a feature containing a list with all
        of the values for each event.
        :param match_dict:
        :param league_season_teams:
        :return:
        """
        if len(match_dict['events']) == 0:
            warnings.warn('There are no events')
            return dict()

        if league_season_teams:
            self.processor.league_season_teams = league_season_teams

        match_level_features = self._match_level_features(match_dict['events'])

        all_events_feature_dict = dict()
        for event_pos, event in enumerate(match_dict['events']):
            proc_event = match_level_features['processed_events'][event_pos]
            event_feature_dict = self._event_level_features(event, proc_event,
                                                            match_level_features['players_importance'])
            # Event position
            event_feature_dict['position'] = (event_pos+1)/len(match_dict['events'])
            # TFIDF Sum
            event_feature_dict['tfidf_sum'] = float(match_level_features['tfidf_sum'][event_pos])
            # Lagged similarity
            event_feature_dict = self._add_lags_to_event(event_feature_dict, match_level_features, event_pos)
            # Event ix (needed for join with targets)
            event_feature_dict['event_ix'] = event_pos

            all_events_feature_dict = self._update_match_dict(event_feature_dict, all_events_feature_dict)

        return all_events_feature_dict

    def run_match(self, match_dict: Dict, league_season_teams: Optional[str] = None) -> pd.DataFrame:
        features = self.create_features(match_dict, league_season_teams)
        return pd.DataFrame(features)

    def get_features(self):
        if not os.path.exists(self.file_path):
            print('{} does not exists'.format(self.file_path))
            print('Executing features')
            self.run_all_matches()
        else:
            print('Reading features from {}'.format(self.file_path))
        return self.read()
