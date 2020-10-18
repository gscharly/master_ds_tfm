from scripts.experiments.experiment import Experiment
from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.conf import TEAMS, LTR_PATH
import scripts.utils.ml_utils as ml_utils

from abc import abstractmethod
from typing import Dict, Optional
import os
import pandas as pd
import pickle
from tqdm import tqdm


class LearnToRank(Experiment):
    """
    Class that provides common methods for the following classes:
    - LTRFeatures
    - LTRTargets
    - LTRFeaturesTargets
    """
    SEED = 10

    def __init__(self, processor: ArticleTextProcessor):
        super().__init__()
        self.processor = processor

    def config(self) -> Dict:
        pass

    def experiment_id(self) -> str:
        experiment_hash = super().experiment_id()
        return experiment_hash

    @property
    def path(self) -> str:
        return '{}/{}/{}'.format(LTR_PATH, self.ltr_type, self.experiment_id())

    @property
    @abstractmethod
    def file_path(self) -> str:
        pass

    @property
    @abstractmethod
    def ltr_type(self) -> str:
        pass

    @property
    def config_path(self) -> str:
        return '{}/config.pickle'.format(self.path)

    @property
    def train_path(self) -> str:
        return '{}/{}/{}/train.csv'.format(LTR_PATH, self.ltr_type, self.experiment_id())

    @property
    def val_path(self) -> str:
        return '{}/{}/{}/validation.csv'.format(LTR_PATH, self.ltr_type, self.experiment_id())

    @property
    def test_path(self) -> str:
        return '{}/{}/{}/test.csv'.format(LTR_PATH, self.ltr_type, self.experiment_id())

    def _match_exists(self, match_url: str) -> bool:
        if os.path.exists(self.file_path):
            pd_all = pd.read_csv(self.file_path)
            return match_url in pd_all['url'].unique()
        else:
            return False

    @staticmethod
    def _write_match(pd_match: pd.DataFrame, path: str):
        if os.path.exists(path):
            pd_all = pd.read_csv(path)
            pd_all = pd.concat([pd_all, pd_match])
            pd_all.to_csv(path, index=False)
        else:
            pd_match.to_csv(path, index=False)

    @staticmethod
    def _non_processed_dict(all_files: Dict, path: str) -> Dict:
        pd_all = pd.read_csv(path) if os.path.exists(path) else None
        processed_files = 0
        seasons_to_del = list()
        urls_to_del = list()
        if pd_all is not None:
            processed_url_list = pd_all['url'].unique()
            # Update deletion lists
            for season_file, season_values in all_files.items():
                if all(match_url in processed_url_list for match_url in season_values.keys()):
                    processed_files += len(season_values.keys())
                    # del all_files[season_file]
                    seasons_to_del.append(season_file)
                else:
                    for match_url, match_dict in season_values.items():
                        if match_url in processed_url_list:
                            # del all_files[season_file][match_url]
                            urls_to_del.append((season_file, match_url))
                            processed_files += 1
            # Delete here (it can't be done during loop)
            for season_file in seasons_to_del:
                del all_files[season_file]
            for season_file, match_url in urls_to_del:
                del all_files[season_file][match_url]
        print('{} matches have already been processed'.format(processed_files))
        return all_files

    def _write_config(self):
        self._create_directory_if_not_exists()
        if not os.path.exists(self.config_path):
            print('Writing config in {}'.format(self.config_path))
            with open(self.config_path, 'wb') as fp:
                pickle.dump(self.config(), fp)

    def read(self) -> pd.DataFrame:
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path)
        else:
            raise ValueError("{} does not exists".format(self.file_path))

    def get_config(self) -> Dict:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'rb') as fp:
                return pickle.load(fp)
        else:
            raise ValueError("{} does not exists".format(self.config_path))

    @abstractmethod
    def run_match(self, match_dict: Dict, league_season_teams: Optional[str] = None) -> pd.DataFrame:
        pass

    def run_all_matches(self):
        """
        Computes and saves a dataset containing features, targets or both information for all of the articles.
        :return:
        """
        self._create_directory_if_not_exists()
        all_files = self.processor.load_json()
        all_files_proc = self._non_processed_dict(all_files, self.file_path)
        print('Updated all_files')
        print('Results path in {}'.format(self.file_path))
        self._write_config()
        for season_file, season_values in tqdm(all_files_proc.items()):
            print(season_file)
            self.processor.league_season_teams = TEAMS[season_file.split('.')[0]]
            for match_url, match_dict in season_values.items():
                print(match_url)
                match_df = self.run_match(match_dict)
                match_df['url'] = match_url
                match_df['json_file'] = season_file
                self._write_match(match_df, self.file_path)

    def train_val_test_split(self, train_perc: float, val_perc: float):
        pd_df = self.read()
        if not (os.path.exists(self.train_path) and os.path.exists(self.val_path) and os.path.exists(self.test_path)):
            train, val, test = ml_utils.train_validate_test_split(pd_df, train_percent=train_perc,
                                                                  validate_percent=val_perc,
                                                                  seed=self.SEED)
            if not os.path.exists(self.train_path):
                train.to_csv(self.train_path, index=False)

            if not os.path.exists(self.val_path):
                val.to_csv(self.val_path, index=False)

            if not os.path.exists(self.test_path):
                test.to_csv(self.test_path, index=False)
        else:
            print('Train, val and test data are already written')

    def read_train(self):
        print("Reading", self.train_path)
        return pd.read_csv(self.train_path)

    def read_validation(self):
        print("Reading", self.val_path)
        return pd.read_csv(self.val_path)

    def read_test(self):
        print("Reading", self.test_path)
        return pd.read_csv(self.test_path)
