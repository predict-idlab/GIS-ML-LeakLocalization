import math

import tensorflow as tf
from stellargraph.layer.gcn_lstm import FixedAdjacencyGraphConvolution
from tensorflow.keras import initializers, constraints, regularizers
from tensorflow.keras.layers import Input, Dropout, LSTM, GRU, Dense, Permute, Reshape, TimeDistributed
from tensorflow.keras.regularizers import l2


class GCN_LSTM:
    """
    GCN_LSTM is a univariate timeseries forecasting method. The architecture  comprises of a stack of N1 Graph Convolutional layers followed by N2 LSTM layers, a Dropout layer, and  a Dense layer.
    This main components of GNN architecture is inspired by: T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction (https://arxiv.org/abs/1811.05320).
    The implementation of the above paper is based on one graph convolution layer stacked with a GRU layer.

    The StellarGraph implementation is built as a stack of the following set of layers:

    1. User specified no. of Graph Convolutional layers
    2. User specified no. of LSTM layers
    3. 1 Dense layer
    4. 1 Dropout layer.

    The last two layers consistently showed better performance and regularization experimentally.

    .. seealso::

       Example using GCN_LSTM: `spatio-temporal time-series prediction <https://stellargraph.readthedocs.io/en/stable/demos/time-series/gcn-lstm-time-series.html>`__.

       Appropriate data generator: :class:`.SlidingFeaturesNodeGenerator`.

       Related model: :class:`.GCN` for graphs without time-series node features.

    Args:
       seq_len: No. of LSTM cells
       adj: unweighted/weighted adjacency matrix of [no.of nodes by no. of nodes dimension
       gc_layer_sizes (list of int): Output sizes of Graph Convolution  layers in the stack.
       lstm_layer_sizes (list of int): Output sizes of LSTM layers in the stack.
       generator (SlidingFeaturesNodeGenerator): A generator instance.
       bias (bool): If True, a bias vector is learnt for each layer in the GCN model.
       dropout (float): Dropout rate applied to input features of each GCN layer.
       gc_activations (list of str or func): Activations applied to each layer's output; defaults to ``['relu', ..., 'relu']``.
       lstm_activations (list of str or func): Activations applied to each layer's output; defaults to ``['tanh', ..., 'tanh']``.
       kernel_initializer (str or func, optional): The initialiser to use for the weights of each layer.
       kernel_regularizer (str or func, optional): The regulariser to use for the weights of each layer.
       kernel_constraint (str or func, optional): The constraint to use for the weights of each layer.
       bias_initializer (str or func, optional): The initialiser to use for the bias of each layer.
       bias_regularizer (str or func, optional): The regulariser to use for the bias of each layer.
       bias_constraint (str or func, optional): The constraint to use for the bias of each layer.
     """

    def __init__(
            self,
            seq_len,
            adj,
            gc_layer_size,
            lstm_layer_size,
            variates,
            gc_activation=None,
            lstm_activation=None,
            bias=True,
            dropout=0.5,
            kernel_initializer=None,
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer=None,
            bias_regularizer=None,
            bias_constraint=None,
            gru=False,
            return_sequences=False,
            inverted=False,
    ):
        super(GCN_LSTM, self).__init__()

        self.lstm_layer_size = lstm_layer_size
        self.gc_layer_size = gc_layer_size
        self.bias = bias
        self.dropout = dropout
        self.adj = adj
        self.n_nodes = adj.shape[0]
        self.n_features = seq_len
        self.seq_len = seq_len
        self.multivariate_input = variates is not None and variates > 1
        self.variates = variates
        self.outputs = self.seq_len * self.variates

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.return_sequences = return_sequences

        # Activation function for each gcn layer
        if gc_activation is None:
            gc_activation = "relu"
        self.gc_activation = gc_activation

        # Activation function for each lstm layer
        if lstm_activation is None:
            lstm_activation = "tanh"
        self.lstm_activation = lstm_activation

        self._gc_layers = [
            FixedAdjacencyGraphConvolution(
                units=gc_layer_size * variates,
                A=self.adj,
                activation=gc_activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint,
            ),
            FixedAdjacencyGraphConvolution(
                units=int(math.sqrt(gc_layer_size)) * variates,
                A=self.adj,
                activation=gc_activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint,
            )
        ]

        if inverted:
            self._gc_layers = self._gc_layers[::-1]
            self.spatial_dist = TimeDistributed(Dense(self.seq_len))
            self.temp_dist = TimeDistributed(Dense(self.n_nodes))

        if not gru:
            self._lstm_layers = [
                LSTM(self.lstm_layer_size,
                     # input_shape=(self.seq_len, self.n_nodes),
                     activation=self.lstm_activation, activity_regularizer=l2(0.1),
                     return_sequences=self.return_sequences)
            ]
        else:
            self._lstm_layers = [
                GRU(self.lstm_layer_size,
                    # input_shape=(self.seq_len, self.n_nodes),
                    activation=self.lstm_activation, activity_regularizer=l2(0.1),
                    return_sequences=self.return_sequences)
            ]

        self.inverted = inverted

    def __call__(self, x):
        if not self.inverted:
            return self.forward(x)
        else:
            return self.backward(x)

    def forward(self, x):
        """
        Forward call (encoder).
        :param x:
        :return:
        """
        x_in, out_indices = x

        if len(x_in.shape) > 3:
            x_in = Permute((2, 1, 3))(x_in)
        else:
            x_in = Permute((2, 1))(x_in)

        h_layer = x_in
        if not self.multivariate_input:
            # normalize to always have a final variate dimension, with V = 1 if it doesn't exist
            # shape = B x N x T x 1
            h_layer = tf.expand_dims(h_layer, axis=-1)

        # flatten variates into sequences, for convolution
        # shape B x N x (TV)
        h_layer = Reshape((self.n_nodes, self.seq_len * self.variates))(h_layer)

        for layer in self._gc_layers:
            h_layer = layer(h_layer)

        # return the layer to its natural multivariate tensor form
        # shape B x N x T' x V (where T' is the sequence length of the last GC)
        h_layer = Reshape((self.n_nodes, -1, self.variates))(h_layer)

        # put time dimension first for LSTM layers
        # shape B x T' x N x V
        h_layer = Permute((2, 1, 3))(h_layer)

        # flatten the variates across all nodes, shape B x T' x (N V)
        h_layer = Reshape((-1, self.n_nodes * self.variates))(h_layer)

        for layer in self._lstm_layers:
            h_layer = layer(h_layer)

        h_layer = Dropout(self.dropout)(h_layer)

        return h_layer

    def backward(self, x):
        """
        Backward call (decoder).
        :param x:
        :return:
        """
        x_in, out_indices = x

        # shape B x T x N'
        h_layer = x_in

        for layer in self._lstm_layers:
            h_layer = layer(h_layer)

        h_layer = Dropout(self.dropout)(h_layer)
        h_layer = self.temp_dist(h_layer)

        # shape B x N x T'
        h_layer = Permute((2, 1))(h_layer)

        for layer in self._gc_layers:
            h_layer = layer(h_layer)

        h_layer = Dropout(self.dropout)(h_layer)
        h_layer = self.spatial_dist(h_layer)
        h_layer = Permute((2, 1))(h_layer)

        return h_layer

    def in_out_tensors(self, shape=None):
        """
        Builds a GCN model for node  feature prediction

        Returns:
            tuple: ``(x_inp, x_out)``, where ``x_inp`` is a list of Keras/TensorFlow
                input tensors for the GCN model and ``x_out`` is a tensor of the GCN model output.
        """
        # Inputs for features
        if shape is None:
            if self.multivariate_input:
                shape = (None, self.n_features, self.n_nodes, self.variates)
            else:
                shape = (None, self.n_features, self.n_nodes)

        x_t = Input(batch_shape=shape)

        # Indices to gather for model output
        out_indices_t = Input(batch_shape=(None, self.n_nodes), dtype="int32")

        x_inp = [x_t, out_indices_t]
        x_out = self(x_inp)

        return x_inp[0], x_out
