import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils import pairwise_distances_no_broadcast
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import ModelCheckpoint

from static import *
from utils import *


class LSTMAutoEncoder(AutoEncoder):
    """
    An autoencoder model using lstm layers.
    """

    def __init__(self, hidden_neurons=[],
                 hidden_activation='relu', output_activation='sigmoid',
                 loss='mse', optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1, timesteps=3):
        super(LSTMAutoEncoder, self).__init__(hidden_neurons,
                                              hidden_activation, output_activation,
                                              loss, optimizer,
                                              epochs, batch_size, dropout_rate,
                                              l2_regularizer, validation_size, preprocessing,
                                              verbose, random_state, contamination)
        self.timesteps = timesteps
        if self.preprocessing:
            self.scaler_ = StandardScaler()
        self.augmented = False

    def _build_model(self):
        pass

    def augment(self, X):
        return X

    def fit(self, X, y=None):
        """
        Fit the the autoencoder to the given data.
        :param X:
        :param y:
        :return:
        """
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        X_a = self.augment(X)

        X_t, y_t = self.temporize(X_a)

        if self.preprocessing:
            X_norm = self.scale(X_t, fit=True)
        else:
            X_norm = np.copy(X_t)

        # Shuffle the data for validation as Keras do not shuffling for
        np.random.shuffle(X_norm)

        # Calculate the dimension of the encoding layer & compression rate
        self.encoding_dim_ = np.median(self.hidden_neurons)
        self.compression_rate_ = self.n_features_ // self.encoding_dim_

        # Build AE model & fit with X
        self.model_ = self._build_model()
        filepath = os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "weights-improvement-ae.hdf5")
        # Save internals
        dump_to_pickle((self.n_samples_, self.n_features_, self.scaler_),
                       os.path.join(DATA_ROOT, RESOURCES, "checkpoints", "internals.pkl"))
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.history_ = self.model_.fit(X_norm, X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        callbacks=callbacks_list,
                                        verbose=self.verbose).history

        self.custom_predict(X_t)
        return self

    def custom_predict(self, X):
        """
        Predict the anomalousness of the given data.
        :param X:
        :return:
        """
        X_t = X.copy()
        if len(X.shape) < 3:
            X_t = self.augment(X_t)
            X_t, y_t = self.temporize(X_t)

        if self.preprocessing:
            X_norm = self.scale(X_t)
        else:
            X_norm = np.copy(X_t)

        pred_scores = self.flatten(self.model_.predict(X_norm))
        self.decision_scores_ = pairwise_distances_no_broadcast(self.flatten(X_norm), pred_scores)
        self._process_decision_scores()
        return self.labels_

    def temporize(self, X):
        """
        Temporize the input.
        This means the input will be windowed using the given number of time steps.
        The input is assumed to have dimensionality of 2/3.
        The output will use a dimensionality of 3/4
        :param X:
        :return:
        """
        y = np.arange(0, X.shape[0])
        output_X = []
        output_y = []
        for i in range(len(X) - (self.timesteps - 1)):
            t = []
            for j in range(self.timesteps):
                # Gather the past records into the lookback period
                t.append(X[[(i + j)], :])
            output_X.append(t)
            output_y.append(y[i + self.timesteps - 1])
        return np.squeeze(np.array(output_X)), np.array(output_y)

    def flatten(self, X):
        """
        Flatten the input to get the original dimensionality (before temporization).
        :param X:
        :return:
        """
        flattened_X = np.empty((X.shape[0] + self.timesteps - 1, X.shape[2]))  # sample x features array.
        # get the first n-1 elements of the first window
        for i in range(self.timesteps - 1):
            flattened_X[i] = X[0, i, :]
        for i in range(0, X.shape[0]):
            # sample the last element in every window
            flattened_X[i + self.timesteps - 1] = X[i, (X.shape[1] - 1), :]
        if self.augmented:
            flattened_X = flattened_X[:, ::4]
        return flattened_X

    def scale(self, X, scaler=None, fit=False, inverse=False):
        """
        Scale the temporized input.
        :param X:
        :param scaler:
        :param fit:
        :param inverse:
        :return:
        """
        if scaler is None:
            scaler = self.scaler_
        X_scaled = X.copy()
        if fit:
            for i in range(X.shape[0]):
                scaler.partial_fit(X_scaled[i, :, :])

        for i in range(X.shape[0]):
            if not inverse:
                X_scaled[i, :, :] = scaler.transform(X_scaled[i, :, :])
            else:
                X_scaled[i, :, :] = scaler.inverse_transform(X_scaled[i, :, :])

        return X_scaled
