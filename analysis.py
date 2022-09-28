from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from networkx.algorithms.shortest_paths import bidirectional_shortest_path
from networkx import connected_components

from utils import *
from static import *

from plot_network import draw_network

g = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "nx_topology_d.pkl"))
g_alt = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "nx_topology_ud.pkl"))
coords = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "coordinates.pkl"))
translation = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "translation.pkl"))
rtranslation = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "reverse_translation.pkl"))
clusters = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "leak_partitions.pkl"))
lengths = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "pipe_lengths.pkl"))

coords = {translation[node]: coords[node] for node in coords if node in translation}

# choose correct measurement campaign: MC1, MC2, 13_14_july_events
measurements_dir = 'MC1'

mc1_sensors = ['1135866', '1138102', '1151599', '1160964', '1161577', '1171286', '1249368', '1251053', '1256995',
               '1257142',
               '1260304', '1267265', '1377059', '1618407', '1620451', '1628166', '1629341', '1983396', '2018747',
               '2179206',
               '2288344', '2426841', '2427371', '2901330', '2902689', '3085780', '366014', '367398', '402436', '772093',
               '773117', '775700', '776019', '776884', '780007', '923397', '932942', '938205', '938607']

mc2_sensors = ['1130816', '1239222', '1240717', '1249368', '1249408', '1249633', '1250189', '1251053', '1256995',
               '1257142',
               '1257639', '1258609', '1259881', '1260304', '1261324', '1267649', '1269161', '1368299', '1374298',
               '1377059',
               '1422947', '1627949', '1628550', '1628842', '1629354', '1630543', '1631265', '1658112', '1773936',
               '1987738',
               '2018688', '2018705', '2288344', '2933990', '2940654', '402436', '541993']

july_sensors = ['1267265', '2902689', '932942', '1620451', '2179206', '1138102', '402436', '367398', '938903', '776884',
                '2018747', '775700', '2427371', '1135866', '1249368', '1983396', '3085780', '1151599', '1618407',
                '923397', '1160964', '938205']

if "MC1" in measurements_dir:
    sensors = mc1_sensors
elif "MC2" in measurements_dir:
    sensors = mc2_sensors
    g_tmp = g.copy()
    from_node = translation['393474FR']
    to_node = translation['393474TO']
    g_tmp.remove_edge(from_node, to_node)
    areas = ['btown', 'ktown']
    components = dict()
    components_or = dict()
    for i, comp in enumerate(connected_components(g_tmp.to_undirected())):
        components[areas[i]] = comp
        components_or[i] = comp
    g_tmp.remove_nodes_from(components['btown'])
    g = g_tmp
elif "13_14_july_events" in measurements_dir:
    sensors = july_sensors

MC1_leaks = ['775125', '773051', '780176', '936678', '2288126', '2288309', '1164867', '1135282']
MC2_leaks = ['1239007', '1240389', '1657869', '402336', '1260152', '2288323', '401772', '1811296', '2018703']
july_leaks = ['393488', '923252']

n_colors = dict()
n_sizes = dict()
for i, node in enumerate(coords.keys()):
    if rtranslation[node] in sensors:
        n_colors[node] = 'red'
        n_sizes[node] = 10
    else:
        n_colors[node] = 'black'
        n_sizes[node] = 0.1


def spread(data, data_dict, bin_mod=1, spread_mod=1, continuous=True, pipes=True):
    counts, bins = np.histogram(data, bins=10 * bin_mod)
    # create remainder bin if necessary...
    if bins[-1] < 1.0:
        bins = np.append(bins, 1.0)
    while bins[-1] - bins[-2] <= 0.1:
        bins = np.delete(bins, -2)
    buckets = [(bin1, bin2) for bin1, bin2 in zip(bins[:-1], bins[1:])]

    def check_pipe(pipe, cluster):
        check_left, check_right = pipe[0] in cluster, pipe[1] in cluster
        # use "or" instead of "and", so that boundary pipes are also approved
        return check_left or check_right

    spread_dict1 = defaultdict(list)
    spread_dict2 = defaultdict(set)
    for key in data_dict:
        for bucket in buckets:
            if bucket[0] < data_dict[key] <= bucket[1]:
                if not pipes:
                    cls = [cluster for cluster in clusters if key in clusters[cluster]]
                else:
                    cls = [cluster for cluster in clusters if check_pipe(key, clusters[cluster])]
                if len(cls) > 0:
                    spread_dict1[bucket].append(key)
                    spread_dict2[bucket].add(cls[0])
                    break

    spread_factors = defaultdict(int)
    for i, bucket in enumerate(buckets[::-1]):
        spread_factors[bucket] = len([part for part in spread_dict2[bucket]]) * spread_mod
        if i > 0:
            # make sure that lower buckets do receive higher spread factors
            spread_factors[bucket] = max(spread_factors[bucket], spread_factors[buckets[len(buckets) - i]])
    boundaries = dict()
    for i, bucket in enumerate(buckets):
        if i == 0:
            boundaries[bucket] = (spread_factors[bucket],
                                  spread_factors[buckets[i + 1]])
        elif i == len(buckets) - 1:
            boundaries[bucket] = (spread_factors[buckets[i - 1]],
                                  spread_factors[bucket])
        else:
            boundaries[bucket] = ((spread_factors[buckets[i - 1]] + spread_factors[bucket]) / 2,
                                  (spread_factors[bucket] + spread_factors[buckets[i + 1]]) / 2)

    new_data_dict = data_dict.copy()
    for bucket in spread_dict1:
        for key in spread_dict1[bucket]:
            factor = spread_factors[bucket]
            if continuous:
                value = new_data_dict[key]
                # min-max normalization
                scale = (value - bucket[0]) / (bucket[1] - bucket[0])
                factor = (1 - scale) * boundaries[bucket][0] + scale * boundaries[bucket][1]
            new_data_dict[key] /= max(1, factor)

    return new_data_dict


def search_cost(data_dict, leakloc):
    new_dict = data_dict.copy()
    cumul_length = 0
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
               (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
               (0.8, 0.9), (0.9, 1.0)]

    hist = dict()
    while len(new_dict.keys()) > 0:
        max_el = max(new_dict, key=lambda x: new_dict[x])
        cumul_length += lengths[max_el]
        for i, bucket in enumerate(buckets):
            if bucket[0] < new_dict[max_el] <= bucket[1]:
                hist[bucket] = cumul_length
                break
        del new_dict[max_el]
        if max_el[0] == leakloc or max_el[1] == leakloc:
            # complete the histogram
            for i, bucket in enumerate(buckets[::-1]):
                if bucket not in hist:
                    # fill histogram with previous value
                    hist[bucket] = 0.0
                    if i > 0 and buckets[len(buckets) - i] in hist:
                        hist[bucket] = hist[buckets[len(buckets) - i]]

            return cumul_length


def search_cost_bis(data_dict, leakloc):
    max_el = max(data_dict, key=lambda x: data_dict[x])
    path = bidirectional_shortest_path(g_alt, leakloc, max_el[0])
    cumul_length = 0
    for (head, tail) in zip(path, path[1:]):
        if (head, tail) in lengths:
            cumul_length += lengths[(head, tail)]
        elif (tail, head) in lengths:
            cumul_length += lengths[(tail, head)]
    return cumul_length


count = 0.0
avg_lstm_search_cost, avg_tgcn_search_cost = 0.0, 0.0
avg_lstm_search_cost_bis, avg_tgcn_search_cost_bis = 0.0, 0.0
for filename1, filename2 in zip(sorted(os.listdir("analysis/" + measurements_dir + "/lstm")),
                                           sorted(os.listdir("analysis/" + measurements_dir + "/tgcn"))):
    with open(os.path.join("analysis", measurements_dir, "lstm", filename1), 'r') as csvfile1:
        with open(os.path.join("analysis", measurements_dir, "tgcn", filename2), 'r') as csvfile2:
            print("\n\nreading files", filename1, "and", filename2, "...")
            colors = n_colors.copy()
            sizes = n_sizes.copy()
            data1 = pd.read_csv(csvfile1, dtype=str)
            data2 = pd.read_csv(csvfile2, dtype=str)

            reverse_pipes = {(row[data1.columns[0]].split(".")[0],
                              row[data1.columns[0]].split(".")[1]): row[data1.columns[0]]
                             for idx, row in data1.iterrows()}
            pipe_dict1 = {(row[data1.columns[0]].split(".")[0],
                           row[data1.columns[0]].split(".")[1]): float(row[data1.columns[1]])
                          for idx, row in data1.iterrows()}
            pipe_dict2 = {(row[data2.columns[0]].split(".")[0],
                           row[data2.columns[0]].split(".")[1]): float(row[data2.columns[1]])
                          for idx, row in data2.iterrows()}

            common_pipes = set(pipe_dict1.keys()).intersection(set(pipe_dict2.keys()))

            new_pipe_dict1 = dict()
            new_pipe_dict2 = dict()
            for pipe in common_pipes:
                new_pipe_dict1[(translation[pipe[0]], translation[pipe[1]])] = pipe_dict1[pipe]
                new_pipe_dict2[(translation[pipe[0]], translation[pipe[1]])] = pipe_dict2[pipe]
                reverse_pipes[(translation[pipe[0]], translation[pipe[1]])] = reverse_pipes[pipe]

            pipe_dict1 = new_pipe_dict1
            pipe_dict2 = new_pipe_dict2

            leak = filename1.split(".")[0].split("_")[0]
            if "-" in leak:
                leak = leak.split("-")[0]
            colors[translation[leak]] = 'blue'
            sizes[translation[leak]] = 25

            data = [pipe_dict1[key] for key in pipe_dict1]
            pipe_dict1 = spread(data, pipe_dict1, pipes=True)
            pipe_dict2 = spread(data, pipe_dict2, pipes=True)

            draw_network(g, coords, "analysis/" + measurements_dir, "lstm_" + filename1.split(".")[0],
                         colors=colors, sizes=sizes, errors=pipe_dict1,
                         color_nodes=False)
            draw_network(g, coords, "analysis/" + measurements_dir, "tgcn_" + filename1.split(".")[0],
                         colors=colors, sizes=sizes, errors=pipe_dict2,
                         color_nodes=False)

            gis_lstm_search_cost = search_cost(pipe_dict1, translation[leak])
            gis_tgcn_search_cost = search_cost(pipe_dict2, translation[leak])
            gis_lstm_search_cost_bis = search_cost_bis(pipe_dict1, translation[leak])
            gis_tgcn_search_cost_bis = search_cost_bis(pipe_dict2, translation[leak])

            # save histogram with cumulative costs
            bins = range(5)
            plt.hist(['AE-RNN - SC1', 'AE-TGCN - SC1', 'AE-RNN - SC2', 'AE-TGCN - SC2'],
                     bins=bins,
                     weights=[gis_lstm_search_cost, gis_tgcn_search_cost,
                              gis_lstm_search_cost_bis, gis_tgcn_search_cost_bis])

            bin_w = (max(range(len(bins))) - min(range(len(bins)))) / (len(range(len(bins))) - 1)
            ticks = np.arange(min(range(len(bins))) + bin_w / 2, max(range(len(bins))), bin_w)
            plt.xticks(ticks, ['AE-RNN - SC1', 'AE-TGCN - SC1', 'AE-RNN - SC2', 'AE-TGCN - SC2'])
            plt.xlim(bins[0], bins[-1])
            for i, length in enumerate([gis_lstm_search_cost, gis_tgcn_search_cost,
                                        gis_lstm_search_cost_bis, gis_tgcn_search_cost_bis]):
                plt.text(ticks[i] - 0.25, length, "{:.2f}".format(length))
            plt.ylim(0, 250000)
            plt.xlabel('Model type')
            plt.ylabel('Search cost [pipe length in m]')
            print("saving fig", "analysis/" + measurements_dir + "/" + filename1.split(".")[0] + "_overview_search_cost.png")
            plt.savefig("analysis/" + measurements_dir + "/" + filename1.split(".")[0] + "_overview_search_cost.png")
            plt.close()

            avg_lstm_search_cost += gis_lstm_search_cost
            avg_tgcn_search_cost += gis_tgcn_search_cost
            avg_lstm_search_cost_bis += gis_lstm_search_cost_bis
            avg_tgcn_search_cost_bis += gis_tgcn_search_cost_bis
            count += 1.0

avg_lstm_search_cost /= count
avg_tgcn_search_cost /= count
avg_lstm_search_cost_bis /= count
avg_tgcn_search_cost_bis /= count

bins = range(5)
plt.hist(['AE-RNN - SC1', 'AE-TGCN - SC1', 'AE-RNN - SC2', 'AE-TGCN - SC2'],
         bins=bins,
         weights=[avg_lstm_search_cost, avg_tgcn_search_cost, avg_lstm_search_cost_bis, avg_tgcn_search_cost_bis])

bin_w = (max(range(len(bins))) - min(range(len(bins)))) / (len(range(len(bins))) - 1)
ticks = np.arange(min(range(len(bins))) + bin_w / 2, max(range(len(bins))), bin_w)
plt.xticks(ticks, ['AE-RNN - SC1', 'AE-TGCN - SC1', 'AE-RNN - SC2', 'AE-TGCN - SC2'])
plt.xlim(bins[0], bins[-1])
for i, length in enumerate([avg_lstm_search_cost, avg_tgcn_search_cost,
                            avg_lstm_search_cost_bis, avg_tgcn_search_cost_bis]):
    plt.text(ticks[i] - 0.25, length, "{:.2f}".format(length))
plt.ylim(0, 250000)
plt.xlabel('Model type')
plt.ylabel('Search cost [pipe length in m]')
print("saving fig", "analysis/" + measurements_dir + "/summary_search_cost.png")
plt.savefig("analysis/" + measurements_dir + "/summary_search_cost.png")
plt.close()
