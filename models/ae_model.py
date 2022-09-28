from pyod.models.auto_encoder import AutoEncoder

from models.ae_lstm import LSTMAutoEncoder
from models.ae_tgcn import TGCNAutoEncoder
from static import *
from utils import *


class CustomAutoEncoder(object):
    """
    An aggregate class encapsulating various types of autoencoders.
    """

    def __init__(self, kind, params={'epochs': 100,
                                     'dropout_rate': 0.2,
                                     'l2_regularizer': 0.1,
                                     'timesteps': 3,
                                     'layers': 2,
                                     'outer_dim': 128,
                                     'sensor_nodes': [],
                                     'weighted': False,
                                     'augmented': False,
                                     'use_in_tgcn': False,
                                     'use_out_tgcn': False,
                                     'gru': False}, load=False):
        """
        Initialize with given params
        :param kind:
        :param params:
        :param load:
        """
        super().__init__()
        self.kind = kind
        self.params = params
        print("Parameters:", params)
        if kind == "std":
            self.ae_model = AutoEncoder(epochs=params['epochs'],
                                        dropout_rate=params['dropout_rate'],
                                        l2_regularizer=params['l2_regularizer'])
        elif kind == "lstm":
            self.ae_model = LSTMAutoEncoder(epochs=params['epochs'],
                                            dropout_rate=params['dropout_rate'],
                                            l2_regularizer=params['l2_regularizer'],
                                            timesteps=params['timesteps'])
        elif kind == "tgcn":
            self.ae_model = TGCNAutoEncoder(epochs=params['epochs'],
                                            dropout_rate=params['dropout_rate'],
                                            l2_regularizer=params['l2_regularizer'],
                                            timesteps=params['timesteps'],
                                            layers=params['layers'],
                                            outer_dim=params['outer_dim'],
                                            sensor_nodes=params['sensor_nodes'],
                                            weighted=params['weighted'],
                                            augmented=params['augmented'],
                                            use_in_tgcn=params['use_in_tgcn'],
                                            use_out_tgcn=params['use_out_tgcn'],
                                            gru=params['gru'])
        if load:
            weights_file = os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "weights-improvement-ae.hdf5")
            internals_file = os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "internals.pkl")
            if os.path.exists(weights_file) and os.path.exists(internals_file):
                internals = load_from_pickle(internals_file)
                self.ae_model.n_samples_, self.ae_model.n_features_, self.ae_model.scaler_ = internals
                self.ae_model.model_ = self.ae_model._build_model()
                self.ae_model.model_.load_weights(weights_file)
                self.ae_model.is_fitted = True

    def fit(self, X):
        """
        Fit the chosen AE model to the input data.
        :param X:
        :return:
        """
        X_train = X
        self.ae_model.fit(X_train)

        train_anom_choices, X_train_rec = None, None
        if self.kind == "std":
            X_train_rec = self.ae_model.scaler_.inverse_transform(
                self.ae_model.model_.predict(self.ae_model.scaler_.transform(X_train)))
        else:
            # train_anom_choices = self.ae_model.custom_predict(X_train)
            X_a = self.ae_model.augment(X_train)
            X_t, y_t = self.ae_model.temporize(X_a)
            X_train_rec = self.ae_model.flatten(
                self.ae_model.scale(self.ae_model.model_.predict(self.ae_model.scale(X_t)), inverse=True))
        return train_anom_choices, X_train_rec

    def predict(self, X):
        """
        Predict the anomaly scores for the given input data.
        :param X:
        :return:
        """
        X_test = X

        test_anom_choices, X_test_rec = None, None
        if self.kind == "std":
            X_test_rec = self.ae_model.scaler_.inverse_transform(
                self.ae_model.model_.predict(self.ae_model.scaler_.transform(X_test)))
        else:
            # test_anom_choices = self.ae_model.custom_predict(X_test)
            X_a = self.ae_model.augment(X_test)
            X_t, y_t = self.ae_model.temporize(X_a)
            X_test_rec = self.ae_model.flatten(
                self.ae_model.scale(self.ae_model.model_.predict(self.ae_model.scale(X_t)), inverse=True))
        return test_anom_choices, X_test_rec