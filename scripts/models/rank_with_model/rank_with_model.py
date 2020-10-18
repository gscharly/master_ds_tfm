# Scripts imports
from scripts.models.row_number_rank import RowNumberRank
from scripts.models.train_experiment import TrainExperiment

# DS imports
import pandas as pd


class RankModel(RowNumberRank):
    RANK_TYPE = 'rank_with_model'

    def __init__(self, ltr, n: int):
        assert isinstance(ltr, TrainExperiment), "ltr object must be of class TrainExperiment"
        super().__init__(ltr=ltr, n=n)

    @property
    def rank_type(self) -> str:
        return self.RANK_TYPE

    def get_scores_df(self) -> pd.DataFrame:
        """
        Returns a df containing the following columns:
        - url
        - event_ix
        - predicted score using the provided model
        :return:
        """
        # Load model
        model = self.ltr.read_model()
        # Load data
        df = self.load_data()
        X = self.ltr.preprocess_data(df)
        X = X.drop(self.ltr.target_col, axis=1)
        # Score
        y_pred = model.predict(X)
        # Create df with url, event_ix, score
        df_cols = df[['url', 'event_ix']].copy()
        df_cols['score'] = y_pred
        return df_cols
