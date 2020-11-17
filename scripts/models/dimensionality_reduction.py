from sklearn.decomposition import TruncatedSVD

from typing import Dict


class DimensionalityReduction:
    """
    Class that provides methods to reduce datasets dimensions before training a model.
    """
    DIM_REDUCTION_AVAILABLE_METHODS = ['truncated_svd']

    def __init__(self, dim_reduction: str, dim_reduction_params: Dict):
        if dim_reduction not in self.DIM_REDUCTION_AVAILABLE_METHODS:
            raise ValueError("dim_reduction must be one of {}".format(self.DIM_REDUCTION_AVAILABLE_METHODS))
        print(f'Using {dim_reduction} for dimensionality reduction')
        self.dim_reduction = dim_reduction
        self.dim_reduction_params = dim_reduction_params

    def dim_reduction_pipe(self):
        if self.dim_reduction == 'truncated_svd':
            return self.truncated_svd(**self.dim_reduction_params)
        else:
            raise ValueError('{} is not available. Try one of {}'.format(self.dim_reduction,
                                                                         self.DIM_REDUCTION_AVAILABLE_METHODS))

    @staticmethod
    def truncated_svd(n_components: int, random_state: int = 10):
        """
        Linear dimensionality reduction. This method does NOT center the data (unlike PCA). This is the only
        available method in sklearn to work with sparse matrices directly.
        :return:
        """
        return TruncatedSVD(n_components=n_components, random_state=random_state)
