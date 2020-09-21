from typing import List, Dict, Tuple, Optional

from rouge import Rouge
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import gensim.downloader as api

from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.text.article_text_processor import ArticleTextProcessor
from scripts.extractive_summary.ltr.learn_to_rank import LearnToRank


class LTRTargets(LearnToRank):
    """
    Class that contains metrics and distances used to build targets for a Learning to Rank algorithm.
    """
    AVAILABLE_METRICS = ['rouge', 'cosine_tfidf', 'wmd']
    ROUGE_PARAMS = ['rouge_mode', 'rouge_metric']
    LTR_TYPE = 'targets'

    def __init__(self, metric: str, metric_params: Dict, drop_teams: bool = False, lemma: bool = False,
                 processor: Optional[ArticleTextProcessor] = None):
        assert metric in self.AVAILABLE_METRICS, 'Available metrics: {}'.format(self.AVAILABLE_METRICS)
        print('Setting target metric to', metric)
        self.processor = processor if processor else ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma)
        super().__init__(processor=self.processor)
        self.metric = metric
        self.metric_params = metric_params
        self.text_proc = BasicTextProcessor()
        self.drop_teams = drop_teams
        self.lemma = lemma

    def config(self) -> Dict:
        return {
            'target_metric': self.metric,
            'drop_teams': self.drop_teams,
            'lemma': self.lemma,
            'metric_params': self.metric_params
        }

    @property
    def ltr_type(self) -> str:
        return self.LTR_TYPE

    @property
    def file_path(self) -> str:
        return '{}/{}.csv'.format(self.path, self.LTR_TYPE)
    
    def _process_events_article(self, match_dict: Dict) -> Tuple[List[str], List[str]]:
        """
        Process events and articles text.
        :param match_dict:
        :return:
        """
        proc_events = [' '.join(self.processor.process_match_text(event))
                       for event in match_dict['events']]
        proc_article_sents = self.processor.process_match_article(match_dict['article'])
        # Some events and articles are empty after processing
        # proc_events_fil = list(filter(lambda event: event != '', proc_events))
        # proc_article_sents_fil = list(filter(lambda sent: sent != '', proc_article_sents))
        return proc_events, proc_article_sents

    def rouge(self, match_dict: Dict, verbose=False, rouge_mode='rouge-l', rouge_metric='f') -> List[Dict]:
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
            # Some events and articles are empty after processing
            event_ref_scores = [rouge.get_scores(event, ref_sent) if event and ref_sent else list()
                                for ref_sent in proc_article_sents]
            event_ref_f_scores = [scores[0][rouge_mode][rouge_metric] if scores else 0 for scores in event_ref_scores]
            sentence_ix = event_ref_f_scores.index(max(event_ref_f_scores))
            event_article_list.append(
                {'event_ix': event_ix, 'sentence_ix': sentence_ix, 'score': max(event_ref_f_scores)}
            )
            if verbose:
                print('Event:', event)
                print('Nearest article sentence:', proc_article_sents[sentence_ix])
                print()
        return event_article_list

    def cosine_distance(self, match_dict: Dict, verbose: bool = False, **count_vec_kwargs):
        proc_events, proc_article_sents = self._process_events_article(match_dict)
        # Train tfidf with article sentences
        pipe = Pipeline([('count', CountVectorizer(**count_vec_kwargs)),
                         ('tfid', TfidfTransformer())])
        x = pipe.fit_transform(proc_article_sents)
        x_events = pipe.transform(proc_events)
        # Distances
        distances = cosine_similarity(x_events, x)
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
                print('Processed event:', proc_events[event_ix])
                print('Processed article sentence:', proc_article_sents[sentences_ixs[event_ix]])
                print()

        return event_article_list

    def wmd(self, match_dict: Dict, verbose: bool = False, norm: bool = True):
        """
        This functions uses WMD to calculate distances between events and article sentences.
        :param match_dict:
        :param verbose:
        :param norm: whether to normalize the word2vec vectors
        :return:
        """
        proc_events, proc_article_sents = self._process_events_article(match_dict)
        # Download word2vec using gensim
        model = api.load('word2vec-google-news-300')
        if norm:
            model.init_sims(replace=True)
        event_article_list = list()
        for event_ix, event in enumerate(proc_events):
            event_ref_scores = [model.wmdistance(event, ref_sent) for ref_sent in proc_article_sents]
            # Minimum distance
            sentence_ix = event_ref_scores.index(min(event_ref_scores))
            event_article_list.append(
                {'event_ix': event_ix, 'sentence_ix': sentence_ix, 'score': min(event_ref_scores)}
            )
            if verbose:
                print('Event:', event)
                print('Nearest article sentence:', proc_article_sents[sentence_ix])
                print()
        return event_article_list

    def create_match_targets(self, match_dict: Dict, verbose: bool, league_season_teams: Optional[str] = None):
        """
        Calculates the target for a match. Specific metric params can be passed.
        :param match_dict:
        :param verbose:
        :param league_season_teams:
        :return:
        """
        if league_season_teams:
            self.processor.league_season_teams = league_season_teams

        if self.metric == 'rouge':
            assert all(k in self.ROUGE_PARAMS for k in self.metric_params.keys()),\
                'Rouge params are {}'.format(self.ROUGE_PARAMS)
            event_article_list = self.rouge(match_dict, verbose, **self.metric_params)
        elif self.metric == 'cosine_tfidf':
            event_article_list = self.cosine_distance(match_dict, verbose, **self.metric_params)
        elif self.metric == 'wmd':
            event_article_list = self.wmd(match_dict, verbose, **self.metric_params)
        else:
            raise ValueError('Metric {} is not available. Try one of {}'.format(self.metric,
                                                                                self.AVAILABLE_METRICS))
        return event_article_list

    def print_scores_info(self, match_dict: Dict, event_article_list: List[Dict], reverse=True):
        article_sentences = self.text_proc.get_sentences(match_dict['article'])
        article_sentences_text = [str(sent).replace('\n', '') for sent in article_sentences]
        scores = sorted([(el['score'], el['event_ix'], el['sentence_ix']) for el in event_article_list],
                        reverse=reverse)
        for info in scores:
            print('Score:', info[0])
            print('Event:', match_dict['events'][info[1]])
            print('Nearest article sentence:', article_sentences_text[info[2]])
            print()

    def run_match(self, match_dict: Dict, league_season_teams: Optional[str] = None):
        event_article_list = self.create_match_targets(match_dict, verbose=False,
                                                       league_season_teams=league_season_teams)
        pd_targets = pd.DataFrame(event_article_list)
        return pd_targets
