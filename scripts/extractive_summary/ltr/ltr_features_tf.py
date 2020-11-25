# Scripts
from scripts.extractive_summary.ltr.learn_to_rank import LearnToRank
from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.conf import TEAMS, LTR_PATH
from scripts.utils.helpers import hashlib_hash


# DS
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix

# Other
from typing import Dict, Optional, List
from tqdm import tqdm
import os
import pickle


class LTRFeaturesTF(LearnToRank):
    LTR_TYPE = 'features'

    def __init__(self, mode: str, count_vec_kwargs: Dict, drop_teams: bool = False,
                 lemma: bool = False, processor: Optional[ArticleTextProcessor] = None):
        self.processor = processor if processor else ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma)
        super().__init__(processor=self.processor)

        if mode not in ['tfidf', 'tf']:
            raise ValueError("Mode must be one of [tfidf, tf]")
        print(f'Setting mode to {mode}')
        self.mode = mode
        self.count_vec_kwargs = count_vec_kwargs
        self.drop_teams = processor.drop_teams if processor else drop_teams
        self.lemma = processor.lemma if processor else lemma

    def config(self) -> Dict:
        return {
            'mode': self.mode,
            'drop_teams': self.drop_teams,
            'lemma': self.lemma,
            'count_vec_kwargs': self.count_vec_kwargs
        }

    def config_events(self) -> Dict:
        return {
            'drop_teams': self.drop_teams,
            'lemma': self.lemma
        }

    def events_id(self) -> str:
        """
        Computes a hash using a config dictionary
        :return:
        """
        return hashlib_hash(sorted(self.config_events().items()))[:10]

    @property
    def events_path(self) -> str:
        """We store the processed events in a different path, so that we don't need to execute it every time
        we run a new experiment"""
        return '{}/{}/events/{}'.format(LTR_PATH, self.ltr_type, self.events_id())

    @property
    def events_file(self) -> str:
        return f'{self.events_path}/processed_events.pickle'

    @property
    def ltr_type(self) -> str:
        return self.LTR_TYPE

    @property
    def file_path(self) -> str:
        return '{}/{}.pickle'.format(self.path, self.LTR_TYPE)

    def read_processed_events(self) -> List[str]:
        return pickle.load(open(self.events_file, 'rb'))

    def save_processed_events(self, events: List[str]):
        pickle.dump(events, open(self.events_path, 'wb'))

    def save_features(self, x: csr_matrix):
        print(f'Saving to {self.file_path}')
        pickle.dump(x, open(self.file_path, 'wb'))

    def run_match(self, match_dict: Dict, league_season_teams: Optional[str] = None):
        """We won't be doing a process per match"""
        return

    def _choose_pipeline(self) -> Pipeline:
        if self.mode == 'tfidf':
            return Pipeline([('count', CountVectorizer(**self.count_vec_kwargs)),
                             ('tfidf', TfidfTransformer())])
        else:
            return Pipeline([('count', CountVectorizer(**self.count_vec_kwargs))])

    def _all_events_list(self) -> List[str]:
        """
        Returns a list of processed events suitable for a CountVectorizer. It first checks if it's already been
        generated (it takes a looong time to finish)
        :return:
        """
        if os.path.exists(self.events_file):
            return self.read_processed_events()

        os.makedirs(self.events_path)

        all_files = self.processor.load_json()
        processed_events_list = list()
        for season_file, season_values in tqdm(all_files.items()):
            print(season_file)
            self.processor.league_season_teams = TEAMS[season_file.split('.')[0]]
            for match_url, match_dict in season_values.items():
                print(match_url)
                proc_events = [' '.join(self.processor.process_match_text(event))
                               for event in match_dict['events']]
                processed_events_list.extend(proc_events)
        self.save_processed_events(processed_events_list)
        return processed_events_list

    @staticmethod
    def _tf_to_pandas(x: np.array, pipe: Pipeline):
        count_stage = pipe['count']
        return pd.DataFrame(x.todense(), columns=count_stage.get_feature_names())

    def run_all_matches(self):
        self._write_config()
        processed_events_list = self._all_events_list()
        print('Finished processing events')
        pipe = self._choose_pipeline()
        print(f'Training {self.mode}')
        x = pipe.fit_transform(processed_events_list)
        self.save_features(x)

    def get_features(self) -> csr_matrix:
        print(f'Reading features from {self.file_path}')
        return pickle.load(open(self.file_path, 'rb'))

