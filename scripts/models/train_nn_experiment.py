# Scripts imports
from scripts.experiments.experiment import Experiment
from scripts.conf import MODELS_PATH
from scripts.utils.batch_generator import BatchGenerator

# DS imports
import numpy as np
from sklearn.pipeline import Pipeline
from scipy.sparse.csr import csr_matrix
from tensorflow import keras
from tensorflow.keras import Model

# Other imports
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union
import pickle
import os


class TrainNNExperiment(Experiment):
    """
    This abstraction is meant to be used when features are too big to use a df (for example with a
    tf with big vocabulary) and a neural net is used

    This training experiment will directly load and train a preprocessed sparse and compressed numpy matrix.
    """

    def __init__(self, model_params: Dict, epochs: int, batch_size: int, opt_metric: Optional[str] = None, cv: int = 0,
                 shuffle: bool = True, workers: int = 6, use_multiprocessing: bool = False, max_queue_size: int = 10):
        super().__init__()
        self.model_params = model_params
        # CV settings
        self.cv = cv
        self.opt_metric = opt_metric
        # NN
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Threads
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

    def config(self) -> Dict:
        pass

    def experiment_id(self) -> str:
        experiment_hash = super().experiment_id()
        return experiment_hash

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @property
    def path(self) -> str:
        return '{}/{}/{}'.format(MODELS_PATH, self.model_type, self.experiment_id())

    @property
    def model_path(self) -> str:
        return f'{self.path}/keras_model.h5'

    def read_model(self) -> Union[Pipeline, keras.Model]:
        return keras.models.load_model(self.model_path)

    def read_keras_model(self) -> Pipeline:
        # Load the pipeline first:
        pipeline = pickle.load(open(self.model_path, 'rb'))
        # Then, load the Keras model:
        pipeline.named_steps['estimator'].model = keras.models.load_model('keras_model.h5')
        return pipeline

    @property
    def model_info_path(self) -> str:
        return '{}/history.pickle'.format(self.path)

    def read_model_info(self) -> Dict:
        return pickle.load(open(self.model_info_path, 'rb'))

    @abstractmethod
    def pipeline(self) -> Union[Pipeline, Model]:
        """Define the model's pipeline"""
        pass

    @abstractmethod
    def model_out(self, model):
        """Returns model additional outputs on training data"""
        pass

    @abstractmethod
    def train_data(self) -> Tuple[csr_matrix, np.array]:
        """Returns x_train and y_train"""
        pass

    @abstractmethod
    def validation_data(self) -> Tuple[csr_matrix, np.array]:
        """Returns x_val and y_val"""
        pass

    def _persist_model(self, best_model, model_info: Dict):
        pass

    def _persist_keras_pipeline(self, model: Union[Pipeline, Model], history):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if isinstance(model, Pipeline):
            # Save the Keras model first:
            model.named_steps['estimator'].model.save(self.model_path)

            # This hack allows us to save the sklearn pipeline:
            model.named_steps['estimator'].model = None

            # Finally, save the pipeline:
            pickle.dump(model, open(self.model_path, 'wb'))
        else:
            model.save(self.model_path)
        pickle.dump(history.history, open(self.model_info_path, 'wb'))

    def _model_info(self, pipeline: Pipeline) -> Dict:
        pass

    def train(self):
        """
        Method that trains a model
        :return:
        """
        self._write_config()
        X_train, y_train = self.train_data()
        X_val, y_val = self.validation_data()
        # Load model
        pipeline = self.pipeline()
        # Train model
        print('Training model...')
        # We will train the model by mini batches, converting the sparse matrix into dense little by little using
        # a generator
        train_generator = BatchGenerator(X_train, y_train, batch_size=self.batch_size)
        val_generator = BatchGenerator(X_val, y_val, batch_size=self.batch_size)
        history = pipeline.fit(x=train_generator,
                               validation_data=val_generator,
                               epochs=self.epochs,
                               validation_steps=len(val_generator),
                               steps_per_epoch=len(train_generator),
                               shuffle=self.shuffle,
                               workers=self.workers,
                               use_multiprocessing=self.use_multiprocessing,
                               max_queue_size=self.max_queue_size,
                               callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
                               verbose=1)
        # Persist
        self._persist_keras_pipeline(pipeline, history)
