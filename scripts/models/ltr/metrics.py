# Scripts imports
from scripts.models.metrics_experiment import MetricsExperiment
from scripts.models.ltr.train import LTRTrain

# DS imports
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

# Other scripts
import os
import pickle
from typing import Dict


class LTRMetrics(MetricsExperiment):

    def __init__(self, train_exp: LTRTrain):
        super().__init__()
        self.train_exp = train_exp
        self.METRICS_PATH = {data_type: f'{self.train_exp.path}/{data_type}_metrics.pickle'
                             for data_type in self.DATA_TYPES}

    def persist_metrics(self, metrics: Dict, data_type: str):
        path_2_write = self.METRICS_PATH[data_type]
        print('Writing metrics to', path_2_write)
        pickle.dump(metrics, open(path_2_write, 'wb'))

    def metrics(self, df: pd.DataFrame) -> Dict:
        model = self.train_exp.read_model()
        X = self.train_exp.preprocess_data(df)
        y_true = X[self.train_exp.target_col]
        X = X.drop(self.train_exp.target_col, axis=1)
        y_pred = model.predict(X)
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
            'error': y_true - y_pred
        }

    def metrics_train(self):
        df = self.train_exp.train_data()
        metrics = self.metrics(df)
        self.persist_metrics(metrics, data_type='train')

    def metrics_val(self):
        df = self.train_exp.ltr.read_validation()
        metrics = self.metrics(df)
        self.persist_metrics(metrics, data_type='validation')

    def metrics_test(self):
        df = self.train_exp.ltr.read_test()
        metrics = self.metrics(df)
        self.persist_metrics(metrics, data_type='test')

    def get_metrics(self, data_type: str) -> Dict:
        path_2_read = self.METRICS_PATH[data_type]
        print('Reading metrics from', path_2_read)
        if os.path.exists(path_2_read):
            return pickle.load(open(path_2_read, 'rb'))
        else:
            raise ValueError(f"{path_2_read} does not exist; please execute metrics")

    def show_metrics(self, data_type: str):
        metrics = self.get_metrics(data_type)
        metrics_list = ['mse', 'mae', 'r2']
        # Print metrics
        for m in metrics_list:
            print(m, ':', metrics[m])
        # Plot distributions
        to_plot = ['y_true', 'y_pred']

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        for t_plot in to_plot:
            axs[0].hist(metrics[t_plot], bins=20, label=t_plot)
        axs[0].legend()

        for t_plot in to_plot:
            sns.kdeplot(metrics[t_plot], ax=axs[1], label=t_plot)
        axs[1].legend()
        plt.plot()

        sns.displot(metrics['error'], bins=20)
        plt.title('error')
        plt.show()





