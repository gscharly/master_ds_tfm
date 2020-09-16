from typing import List, Dict, Tuple

from rouge import Rouge
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import numpy as np
import gensim.downloader as api

from scripts.text.basic_text_processor import BasicTextProcessor
from scripts.text.article_text_processor import ArticleTextProcessor


class TargetMetrics:
    """
    Class that contains metrics and distances used to build targets for a Learning to Rank algorithm.
    """
    AVAILABLE_METRICS = ['rouge', 'cosine_tfidf', 'wmd']
    ROUGE_PARAMS = ['rouge_mode', 'rouge_metric']

    def __init__(self, metric: str, drop_teams: bool = False, lemma: bool = False):
        assert metric in self.AVAILABLE_METRICS, 'Available metrics: {}'.format(self.AVAILABLE_METRICS)
        print('Setting target metric to', metric)
        self.metric = metric
        self.text_proc = BasicTextProcessor()
        self.processor = ArticleTextProcessor(drop_teams=drop_teams, lemma=lemma)

    def _process_events_article(self, match_dict: Dict) -> Tuple[List[str], List[str]]:
        """
        Process events and articles text.
        :param match_dict:
        :return:
        """
        proc_events = [' '.join(self.processor.process_match_text(event))
                       for event in match_dict['events']]
        proc_article_sents = self.processor.process_match_article(match_dict['article'])
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
