from scripts.text.basic_text_processor import BasicTextProcessor
import scripts.conf as conf

import os
from os import listdir
from os.path import isfile, join
import pickle
from typing import Dict, Tuple
import warnings

import pandas as pd
from rouge import Rouge


class SummaryEvaluation:
    AVAILABLE_METRICS = ['rouge']

    def __init__(self, metric: str):
        assert metric in self.AVAILABLE_METRICS, 'Available metrics: {}'.format(self.AVAILABLE_METRICS)
        print('Setting target metric to', metric)
        self.metric = metric
        self.text_proc = BasicTextProcessor()

    def _preprocess_text(self, text: str) -> str:
        """
        - Remove stopwords
        - Remove punctuation
        - Remove numbers
        - Lemmatize
        :param text:
        :return:
        """
        doc = self.text_proc.nlp(text)
        sentence_list = list()
        for sentence in doc.sents:
            tokens = [token.lemma_.lower() for token in sentence if self.text_proc.token_filter_stopword(token) and
                      self.text_proc.token_filter(token) and not self.text_proc.has_numbers(token)]
            sentence_list.append(' '.join(tokens))
        return '. '.join(sentence_list)

    # TODO incluir opcion para que se haga la media solo para unas determinadas noticias
    @staticmethod
    def _avg_rouge(scores_dict: Dict) -> Dict:
        avg_scores_dict = dict()
        for _, league_dict in scores_dict.items():
            for _, scores in league_dict.items():
                if not avg_scores_dict:
                    avg_scores_dict = {k: {'f': list(), 'p': list(), 'r': list()} for k in scores[0].keys()}
                for rouge_mode, metrics in scores[0].items():
                    for metric_name, metric_value in metrics.items():
                        avg_scores_dict[rouge_mode][metric_name].append(metric_value)
        scores_dict = avg_scores_dict.copy()
        for rouge_mode, metrics in avg_scores_dict.items():
            for metric_name, metrics_list in metrics.items():
                scores_dict[rouge_mode][metric_name] = sum(metrics_list) / len(metrics_list)
        return scores_dict

    @staticmethod
    def rouge(candidate: str, reference: str, rouge_mode: str = None) -> Dict:
        if not rouge_mode:
            rouge = Rouge()
        else:
            rouge = Rouge(metrics=[rouge_mode])
        score = rouge.get_scores(candidate, reference)
        return score

    def evaluate(self, pd_df_summaries: pd.DataFrame, path: str, preprocess_text: bool = False,
                 summary_key: str = 'summary_events') -> Tuple[Dict, Dict]:
        cond1 = os.path.exists(path + '.pickle') and os.path.exists(path + '_avg.pickle') and not preprocess_text
        cond2 = (os.path.exists(path + '_processed.pickle') and os.path.exists(path + '_processed_avg.pickle')
                 and preprocess_text)
        if cond1:
            print('Metrics already exist')
            with open(path + '.pickle', 'rb') as fp:
                scores_dict = pickle.load(fp)
            with open(path + '_avg.pickle', 'rb') as fp:
                avg_scores_dict = pickle.load(fp)
            return scores_dict, avg_scores_dict
        elif cond2:
            print('Metrics with processed text already exist')
            with open(path + '_processed.pickle', 'rb') as fp:
                scores_dict = pickle.load(fp)
            with open(path + '_processed_avg', 'rb') as fp:
                avg_scores_dict = pickle.load(fp)
            return scores_dict, avg_scores_dict
        else:
            pd_df_summaries_na = pd_df_summaries[~pd_df_summaries[summary_key].isna()]
            n_nas = len(pd_df_summaries) - len(pd_df_summaries_na)
            if n_nas:
                print("There are {} articles with an empty summary".format(n_nas))
            print("Performing evaluation for {} articles".format(len(pd_df_summaries_na)))
            scores_dict = dict()
            for _, row in pd_df_summaries_na.iterrows():
                candidate = row[summary_key]
                reference = row['article']
                if preprocess_text:
                    candidate = self._preprocess_text(candidate)
                    reference = self._preprocess_text(reference)
                if not candidate or not reference:
                    warnings.warn('Could not perform score: empty summary')
                    continue
                else:
                    scores = self.rouge(candidate, reference)
                    if row['json_file'] in scores_dict.keys():
                        scores_dict[row['json_file']][row['url']] = scores
                    else:
                        scores_dict[row['json_file']] = {row['url']: scores}

            print('Writing to', path + '.pickle')
            with open(path + '.pickle', 'wb') as fp:
                pickle.dump(scores_dict, fp)

            avg_scores_dict = self._avg_rouge(scores_dict)
            print('Writing avg to', path + '_avg.pickle')
            with open(path + '_avg.pickle', 'wb') as fp:
                pickle.dump(avg_scores_dict, fp)
            return scores_dict, avg_scores_dict

    def evaluate_all_summaries(self, preprocess_text: bool = False):
        """
        Performs evaluation for every summary file
        :param preprocess_text:
        :return:
        """
        only_files = [f for f in listdir(conf.SUMMARY_PATH) if isfile(join(conf.SUMMARY_PATH, f)) and 'map' not in f]
        print("Evaluating the following summaries:", only_files)
        pd_matches = pd.read_csv(conf.ARTICLES_PATH)
        join_cols = ['json_file', 'url']
        match_cols = join_cols + ['article', 'events']
        sum_cols = join_cols + ['summary_events']
        for f in only_files:
            print('Evaluating', f)
            pd_summaries = pd.read_csv(join(conf.SUMMARY_PATH, f))
            pd_matches_sum = pd_matches[match_cols].merge(pd_summaries[sum_cols], on=join_cols, how='inner')
            results_path = '{}/summaries/{}/{}'.format(conf.METRICS_PATH, self.metric, f.split('.')[0])
            if preprocess_text:
                results_path += '_processed'
            _ = self.evaluate(pd_matches_sum, results_path, preprocess_text=preprocess_text)

    def bound_metrics(self, preprocess_text: bool = False):
        """
        Calculates the selected metric using directly the events as a summary. An extractive summary will never be able
        to score better recall than this result.
        :param preprocess_text:
        :return:
        """
        pd_matches = pd.read_csv(conf.ARTICLES_PATH)
        results_path = '{}/summaries/{}/upper_bound'.format(conf.METRICS_PATH, self.metric)
        _ = self.evaluate(pd_matches, results_path, preprocess_text=preprocess_text, summary_key='events')

    @staticmethod
    def _exp_metrics_avg(scores_dict: Dict) -> pd.DataFrame:
        pd_dict = {
            'metric': list(),
            'metric_type': list(),
            'value': list()
        }
        for metric, metric_dict in scores_dict.items():
            for metric_type, metric_value in metric_dict.items():
                pd_dict['metric'].append(metric)
                pd_dict['metric_type'].append(metric_type)
                pd_dict['value'].append(metric_value)
        return pd.DataFrame(pd_dict)

    @staticmethod
    def _exp_metrics(scores_dict: Dict) -> pd.DataFrame:
        pd_dict = {
            'metric': list(),
            'metric_type': list(),
            'value': list(),
            'json_file': list(),
            'url': list()
        }

        for league_file, league_dict in scores_dict.items():
            for url, match_scores_list in league_dict.items():
                match_scores_dict = match_scores_list[0]
                for metric, metric_dict in match_scores_dict.items():
                    for metric_type, metric_value in metric_dict.items():
                        pd_dict['metric'].append(metric)
                        pd_dict['metric_type'].append(metric_type)
                        pd_dict['value'].append(metric_value)
                        pd_dict['json_file'].append(league_file)
                        pd_dict['url'].append(url)
        return pd.DataFrame(pd_dict)

    def experiment_metrics_pandas(self, path: str, avg: bool) -> pd.DataFrame:
        with open(path, 'rb') as fp:
            scores_dict = pickle.load(fp)
        if avg:
            return self._exp_metrics_avg(scores_dict)
        else:
            return self._exp_metrics(scores_dict)

    def output_avg_metrics(self) -> pd.DataFrame:
        """
        Returns a dictionary with the average scores for every experiment
        :return:
        """
        results_path = '{}/summaries/{}'.format(conf.METRICS_PATH, self.metric)
        only_files = [f for f in listdir(results_path) if isfile(join(results_path, f)) and 'avg' in f]
        all_scores_df = pd.DataFrame()
        for f in only_files:
            file_score_df = self.experiment_metrics_pandas(f, avg=True)
            file_score_df['experiment'] = f.split('.')[0]
            all_scores_df = pd.concat([all_scores_df, file_score_df])
        return all_scores_df



