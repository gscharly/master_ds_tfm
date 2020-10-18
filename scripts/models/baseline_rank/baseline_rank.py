# Scripts imports
from scripts.models.row_number_rank import RowNumberRank

# DS imports
import pandas as pd


class BaselineRank(RowNumberRank):
    """
    This class uses the output of a system that assigns scores to events to order them. It is a simple baseline_rank
    that will be used as baseline: it orders the events and chooses the first N.
    """
    RANK_TYPE = 'baseline_rank'

    def __init__(self, ltr, n: int):
        super().__init__(ltr=ltr, n=n)

    @property
    def rank_type(self) -> str:
        return self.RANK_TYPE

    def get_scores_df(self) -> pd.DataFrame:
        df = self.load_data()
        df_cols = df[['url', 'event_ix', 'score']].copy()
        return df_cols
