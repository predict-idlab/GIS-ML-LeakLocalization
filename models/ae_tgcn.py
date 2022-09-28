from collections import defaultdict

import numpy as np
import pandas as pd
import stellargraph as sg
from networkx.algorithms.shortest_paths import all_shortest_paths
from tensorflow.keras.layers import Dense, LSTM, GRU, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2

from models.ae_lstm import LSTMAutoEncoder
from models.gcn_lstm import GCN_LSTM
from static import *
from utils import *


class TGCNAutoEncoder(LSTMAutoEncoder):
    def __init__(self, hidden_neurons=[],
                 hidden_activation='relu', output_activation='sigmoid',
                 loss='mse', optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1,
                 layers=2, outer_dim=128,
                 timesteps=4, sensor_nodes=[],
                 weighted=0, augmented=False, use_in_tgcn=False, use_out_tgcn=False, gru=False):
        super(TGCNAutoEncoder, self).__init__(hidden_neurons,
                                              hidden_activation, output_activation,
                                              loss, optimizer,
                                              epochs, batch_size, dropout_rate,
                                              l2_regularizer, validation_size, preprocessing,
                                              verbose, random_state, contamination, timesteps)
        topology = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "nx_topology_ud.pkl"))
        node2pos = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "translation.pkl"))
        pos2node = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "reverse_translation.pkl"))
        sensor_ids = [node2pos[n] for i, n in enumerate(sensor_nodes)]

        nx_topology = topology

        source = []
        target = []
        weight = []

        self.neighbours = defaultdict(list)
        self.distances = dict()

        visited = set()
        for node1 in sensor_ids:
            for node2 in sensor_ids:
                if node1 != node2 and (node1, node2) not in visited:
                    try:
                        all_paths = list(all_shortest_paths(nx_topology, node1, node2))
                        all_paths.extend(list(all_shortest_paths(nx_topology, node2, node1)))
                        all_paths = [p for p in all_paths if
                                     not any([id in p for id in sensor_ids if id != node1 and id != node2])]
                        all_path_lengths = [len(p) for p in all_paths]
                        path = all_paths[np.argmin(all_path_lengths)]
                        if len(all_paths) > 0:
                            source.append(sensor_ids.index(node1))
                            target.append(sensor_ids.index(node2))
                            if weighted == 1:
                                weight.append(1 / len(path))
                            elif weighted == 2:
                                distinct_nodes = set()
                                for p in all_paths:
                                    distinct_nodes.update(p)
                                print("distinct nodes between", node1, "and", node2, ":", len(distinct_nodes))
                                weight.append((1 / len(distinct_nodes)))
                            else:
                                weight.append(1)
                            self.neighbours[sensor_ids.index(node1)].append(sensor_ids.index(node2))
                            self.neighbours[sensor_ids.index(node2)].append(sensor_ids.index(node1))
                            self.distances[(sensor_ids.index(node1), sensor_ids.index(node2))] = len(path)
                            self.distances[(sensor_ids.index(node2), sensor_ids.index(node1))] = len(path)
                    except:
                        # print("no path found between", node1, "and", node2)
                        pass
                visited.add((node1, node2))
                visited.add((node2, node1))

        adj_edges = pd.DataFrame({
            "source": source,
            "target": target,
            "weight": weight
        })

        print("adjacency edges:", adj_edges)

        coordinates = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "coordinates.pkl"))

        # print(coordinates)
        # print(sensor_ids)

        adj_node_features = pd.DataFrame(
            {"x": [coordinates[pos2node[n]][0] for n in sensor_ids],
             "y": [coordinates[pos2node[n]][1] for n in sensor_ids]},
            index=list(range(len(sensor_ids)))
        )

        # no need to normalize beforehand or add self-loops
        # the node generator performs symmetric normalization for us
        # and automatically adds self-connections on the diagonal
        self.adj = sg.mapper.FullBatchNodeGenerator(sg.StellarGraph(nodes=adj_node_features, edges=adj_edges),
                                                    weighted=(True if weighted > 0 else False),
                                                    method="gcn").Aadj.todense()

        self.num_layers = layers
        self.outer_dim = outer_dim

        self.augmented = augmented
        self.use_in_tgcn = use_in_tgcn
        self.use_out_tgcn = use_out_tgcn
        self.gru = gru

    def augment(self, X):
        """
        Augment the input by adding additional variates to each sensor dimension.
        Additional variates:
        - weighted node degree
        - average of neighbours
        - std dev of neighbours

        Input has dim 2 (B x F), output has dim 2 (B X FV)

        :param X:
        :return:
        """
        X_a = np.expand_dims(X, axis=-1)
        if self.augmented:
            extra_variates = np.zeros((X_a.shape[0], X_a.shape[1], 3))
            for i, _ in enumerate(extra_variates):
                for j, _ in enumerate(extra_variates[i]):
                    nb_avg = np.average([X_a[i, nb] for nb in self.neighbours[j]])
                    nb_std_dev = np.std([X_a[i, nb] for nb in self.neighbours[j]])
                    ndegree = sum([self.distances[(j, nb)] for nb in self.neighbours[j]])
                    extra_variates[i, j, 0] = nb_avg
                    extra_variates[i, j, 1] = nb_std_dev
                    extra_variates[i, j, 2] = ndegree
            X_a = np.concatenate((X_a, extra_variates), axis=-1)
            X_a = np.reshape(X_a, newshape=(X_a.shape[0], -1))
        return X_a

    def _build_model(self):
        return self._build_model_lstm(use_in_tgcn=self.use_in_tgcn, use_out_tgcn=self.use_out_tgcn)

    def _build_model_lstm(self, use_in_tgcn=False, use_out_tgcn=False):
        tgcn_autoencoder = Sequential()

        lstm_dim = self.outer_dim
        gcn_dim = self.timesteps * self.timesteps
        time_dim = self.timesteps

        in_tgcn = GCN_LSTM(self.timesteps, self.adj,
                           gc_layer_size=gcn_dim, lstm_layer_size=lstm_dim,
                           variates=1,
                           gru=self.gru,
                           return_sequences=True,
                           inverted=False)
        out_tgcn = GCN_LSTM(self.timesteps, self.adj,
                            gc_layer_size=gcn_dim, lstm_layer_size=lstm_dim,
                            variates=1,
                            gru=self.gru,
                            return_sequences=True,
                            inverted=True)

        # return sequences of last encoder layer must be False
        # to ensure ndim of output is 2 instead of 3
        # this is necessary for the repeat vector

        input_shape = (self.timesteps, self.n_features_)
        if self.augmented and not (use_in_tgcn or use_out_tgcn):
            input_shape = (self.timesteps, self.n_features_ * 4)

        layers = []
        dim = lstm_dim
        if not self.gru:
            # Encoder
            layers.append(LSTM(lstm_dim, activation='relu',
                               activity_regularizer=l2(self.l2_regularizer),
                               input_shape=input_shape,
                               return_sequences=True))
            for i in range(1, self.num_layers - 1):
                dim //= 2
                layers.append(LSTM(dim, activation='relu',
                                   activity_regularizer=l2(self.l2_regularizer),
                                   return_sequences=True))
            layers.append(LSTM(dim // 2, activation='relu',
                               activity_regularizer=l2(self.l2_regularizer),
                               return_sequences=False))

            # Decoder
            layers.append(LSTM(dim // 2, activation='relu',
                               activity_regularizer=l2(self.l2_regularizer),
                               return_sequences=True))
            for i in range(1, self.num_layers - 1):
                dim *= 2
                layers.append(LSTM(dim, activation='relu',
                                   activity_regularizer=l2(self.l2_regularizer),
                                   return_sequences=True))
            layers.append(LSTM(lstm_dim, activation='relu',
                               activity_regularizer=l2(self.l2_regularizer),
                               return_sequences=True))
        else:
            # Encoder
            layers.append(GRU(lstm_dim, activation='relu',
                              activity_regularizer=l2(self.l2_regularizer),
                              input_shape=input_shape,
                              return_sequences=True))
            for i in range(1, self.num_layers - 1):
                dim //= 2
                layers.append(GRU(dim, activation='relu',
                                  activity_regularizer=l2(self.l2_regularizer),
                                  return_sequences=True))
            layers.append(GRU(dim // 2, activation='relu',
                              activity_regularizer=l2(self.l2_regularizer),
                              return_sequences=False))

            # Decoder
            layers.append(GRU(dim // 2, activation='relu',
                              activity_regularizer=l2(self.l2_regularizer),
                              return_sequences=True))
            for i in range(1, self.num_layers - 1):
                dim *= 2
                layers.append(GRU(dim, activation='relu',
                                  activity_regularizer=l2(self.l2_regularizer),
                                  return_sequences=True))
            layers.append(GRU(lstm_dim, activation='relu',
                              activity_regularizer=l2(self.l2_regularizer),
                              return_sequences=True))

        # Encoder
        if use_in_tgcn:
            in_tgcn_inp, in_tgcn_outp = in_tgcn.in_out_tensors()
            in_tgcn_model = Model(inputs=in_tgcn_inp, outputs=in_tgcn_outp)
            tgcn_autoencoder.add(in_tgcn_model)
        else:
            tgcn_autoencoder.add(layers[0])
            tgcn_autoencoder.add(Dropout(self.dropout_rate))

        for i in range(1, self.num_layers):
            tgcn_autoencoder.add(layers[i])
            tgcn_autoencoder.add(Dropout(self.dropout_rate))

        tgcn_autoencoder.add(RepeatVector(time_dim))

        # Decoder
        for i in range(1, self.num_layers):
            tgcn_autoencoder.add(layers[self.num_layers - 1 + i])
            tgcn_autoencoder.add(Dropout(self.dropout_rate))

        if use_out_tgcn:
            out_tgcn_inp, out_tgcn_outp = out_tgcn.in_out_tensors(tgcn_autoencoder.layers[-1].output_shape)
            out_tgcn_model = Model(inputs=out_tgcn_inp, outputs=out_tgcn_outp)
            tgcn_autoencoder.add(out_tgcn_model)
        else:
            tgcn_autoencoder.add(layers[-1])
            tgcn_autoencoder.add(Dropout(self.dropout_rate))
            tgcn_autoencoder.add(TimeDistributed(Dense(input_shape[1])))

        # Compile model
        tgcn_autoencoder.compile(loss=self.loss, optimizer=self.optimizer)
        if self.verbose >= 1:
            print(tgcn_autoencoder.summary())
        return tgcn_autoencoder
