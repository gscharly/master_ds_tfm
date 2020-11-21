# Scripts imports
from scripts.models.row_number_rank import RowNumberRank
from scripts.models.train_experiment import TrainExperiment
from scripts.models.train_all_experiment import TrainAllExperiment
from scripts.models.train_nn_experiment import TrainNNExperiment

# DS imports
import pandas as pd


class RankModel(RowNumberRank):
    RANK_TYPE = 'rank_with_model'

    def __init__(self, ltr, n: int, is_nn: bool = False):
        assert isinstance(ltr, (TrainExperiment, TrainAllExperiment, TrainNNExperiment)),\
            "ltr object must be of class TrainExperiment or TrainAllExperiment"
        super().__init__(ltr=ltr, n=n)
        self.is_nn = is_nn

    @property
    def rank_type(self) -> str:
        return self.RANK_TYPE

    def _reduce_dim(self, x):
        dim_reduction_stage = self.ltr.read_dim_reduction_trained()
        print('Reducing x dimension')
        print(f'Shape before: {x.shape}')
        x = dim_reduction_stage.transform(x)
        print(f'Shape after: {x.shape}')
        return x

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
        features_target = self.load_data()
        if isinstance(features_target, tuple):
            x = features_target[0]
            if self.is_nn:
                if self.ltr.dim_reduction:
                    x = self._reduce_dim(x)
                else:
                    # This is mandatory to predict using NN. This shouldn't alter the order of the rows (checked in notebook)
                    # It only orders the indices for each row
                    x.sort_indices()
            df_cols = features_target[1][['url', 'event_ix']].copy()
        else:
            x = self.ltr.preprocess_data(features_target)
            x = x.drop(self.ltr.target_col, axis=1)
            df_cols = features_target[['url', 'event_ix']].copy()
        # Score
        y_pred = model.predict(x)
        df_cols['score'] = y_pred
        return df_cols
