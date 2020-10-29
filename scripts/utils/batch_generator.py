import math
from scipy.sparse.csr import csr_matrix
from tensorflow import keras

import numpy as np


class BatchGenerator(keras.utils.Sequence):
    def __init__(self, xs, ys, batch_size=64):
        """
        Dataset generator for training epochs.

        Params
        ------
        xs: np.array
            Features.
        ys: np.array
            Targets.
        batch_size: int
            Batch size.
        """
        self.xs = xs
        self.ys = ys
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.xs.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.xs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.ys[idx * self.batch_size:(idx + 1) * self.batch_size]

        if isinstance(self.xs, csr_matrix):
            batch_x = np.array(batch_x.todense())

        return batch_x, batch_y