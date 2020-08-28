from typing import Dict, List, Optional, Tuple
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import numpy as np

from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.extractive_summary.key_events import KeyEvents


class LearnToRank:
    AVAILABLE_METRICS = ['rouge', 'cosine_tfidf']
    ROUGE_PARAMS = ['rouge_mode', 'rouge_metric']

    def __init__(self, target_metric: str = 'rouge'):
        assert target_metric in self.AVAILABLE_METRICS, 'Available metrics: {}'.format(self.AVAILABLE_METRICS)
        print('Setting target metric to', target_metric)

        self.target_metric = target_metric
        self.text_proc = BasicTextProcessor()
        self.key_events = KeyEvents()

    def _rouge(self, match_dict: Dict, verbose=False, rouge_mode='rouge-l', rouge_metric='f') -> List[Dict]:
        """
        For a given list of events and article, this function calculates the ROUGE metric between each event and each
        article sentence. For each event, the sentence with the maximum ROUGE is selected.
        :param match_dict:
        :param rouge_mode:
        :param rouge_metric:
        :param verbose:
        :return:
        """
        proc_events, proc_article_sents = self._process_events_article(match_dict)
        rouge = Rouge(metrics=[rouge_mode])
        event_article_list = list()
        for event_ix, event in enumerate(proc_events):
            event_ref_scores = [rouge.get_scores(event, ref_sent) for ref_sent in proc_article_sents]
            event_ref_f_scores = [scores[0][rouge_mode][rouge_metric] for scores in event_ref_scores]
            sentence_ix = event_ref_f_scores.index(max(event_ref_f_scores))
            event_article_list.append(
                {'event_ix': event_ix, 'sentence_ix': sentence_ix, 'score': max(event_ref_f_scores)}
            )
            if verbose:
                print('Event:', event)
                print('Nearest article sentence:', proc_article_sents[sentence_ix])
                print()
        return event_article_list

    def _cosine_distance(self, match_dict: Dict, verbose=False, **count_vec_kwargs):
        proc_events, proc_article_sents = self._process_events_article(match_dict)
        # Train tfidf with article sentences
        pipe = Pipeline([('count', CountVectorizer(**count_vec_kwargs)),
                         ('tfid', TfidfTransformer())])
        X = pipe.fit_transform(proc_article_sents)
        # X = pipe.transform()
        # vectorizer = CountVectorizer(**count_vec_kwargs)
        # X = vectorizer.fit_transform(proc_article_sents).toarray()
        # tfidfconverter = TfidfTransformer()
        # X = tfidfconverter.fit_transform(X).toarray()
        # Events
        # X_events = vectorizer.transform(proc_events)
        # X_events = tfidfconverter.transform(X_events).toarray()
        X_events = pipe.transform(proc_events)
        # Distances
        distances = cosine_similarity(X_events, X)
        sentences_ixs = distances.argmax(axis=1)
        event_article_list = [{'event_ix': event_ix,
                               'sentence_ix': sentences_ixs[event_ix],
                               'score': np.amax(distances[event_ix])}
                              for event_ix in range(len(proc_events))]
        if verbose:
            article_sentences = self.text_proc.get_sentences(match_dict['article'])
            article_sentences_text = [str(sent).replace('\n', '') for sent in article_sentences]
            for event_ix in range(len(proc_events)):
                print('Event:', match_dict['events'][event_ix])
                print('Nearest article sentence:', article_sentences_text[sentences_ixs[event_ix]])
                print()

        return event_article_list

    def _process_events_article(self, match_dict: Dict) -> Tuple[List[str], List[str]]:
        """
        Process events and articles depending on the desired metric.
        - rouge: extract article sentences
        - cosine_tfidf: preprocess both events and article sentences
        :param match_dict:
        :return:
        """
        article_sentences = self.text_proc.get_sentences(match_dict['article'])
        if self.target_metric == 'rouge':
            article_sentences_text = [str(sent).replace('\n', '') for sent in article_sentences]
            return match_dict['events'], article_sentences_text
        elif self.target_metric == 'cosine_tfidf':
            proc_events = [' '.join(self.key_events.process_match_text(event)) for event in match_dict['events']]
            proc_article_sents = self.key_events.process_match_article(match_dict['article'])
            return proc_events, proc_article_sents

    def create_match_targets(self, match_dict: Dict, verbose: bool, league_season_team: Optional[str] = None,
                             **metrics_params):
        """
        Calculates the target for a match. Specific metric params can be passed.
        :param match_dict:
        :param verbose:
        :param league_season_team:
        :param metrics_params:
        :return:
        """

        if self.target_metric == 'rouge':
            assert all(k in self.ROUGE_PARAMS for k in metrics_params.keys()),\
                'Rouge params are {}'.format(self.ROUGE_PARAMS)
            event_article_list = self._rouge(match_dict, verbose, **metrics_params)
        elif self.target_metric == 'cosine_tfidf':
            self.key_events.league_season_teams = league_season_team
            event_article_list = self._cosine_distance(match_dict, verbose, **metrics_params)
        else:
            raise ValueError('Metric {} is not available. Try one of {}'.format(self.target_metric,
                                                                                self.AVAILABLE_METRICS))
        return event_article_list

    def print_scores_info(self, match_dict: Dict, event_article_list: List[Dict], sort=True):
        article_sentences = self.text_proc.get_sentences(match_dict['article'])
        article_sentences_text = [str(sent).replace('\n', '') for sent in article_sentences]
        if sort:
            scores = sorted([(el['score'], el['event_ix'], el['sentence_ix']) for el in event_article_list],
                            reverse=True)
        else:
            scores = event_article_list
        for info in scores:
            print('Score:', info[0])
            print('Event:', match_dict['events'][info[1]])
            print('Nearest article sentence:', article_sentences_text[info[2]])
            print()
