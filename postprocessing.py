import csv
import json
from collections import defaultdict

from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.algorithms.shortest_paths.unweighted import bidirectional_shortest_path
from tqdm import tqdm

from static import *
from utils import *

topology = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "nx_topology_ud.pkl"))
topology_alt = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "nx_topology_d.pkl"))
pipes = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "pipe_header.pkl"))
translation = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "translation.pkl"))
reverse_translation = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "reverse_translation.pkl"))
coordinates = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "coordinates.pkl"))


def get_distances(name, sensors=None):
    if not os.path.isfile(os.path.join(DATA_ROOT, RESOURCES, name, "distances.pkl")):
        nx_topology = topology
        all_nodes = nx_topology.nodes
        distances = defaultdict(list)
        if sensors is None:
            sensors = all_nodes
        print("Getting distances for", name, "with sensors", sensors)

        avg_nr_of_hops = average_shortest_path_length(nx_topology)
        for s in tqdm(sensors):
            for n in all_nodes:
                start = s
                if s in translation:
                    start = translation[s]
                nr_of_hops = len(bidirectional_shortest_path(nx_topology, n, start))
                distances[s].append(nr_of_hops)
        dump_to_pickle((avg_nr_of_hops, distances), os.path.join(DATA_ROOT, RESOURCES, name, "distances.pkl"))
        return avg_nr_of_hops, distances


def distribute_with_dist(errors, distances):
    avg_dist, distances = distances[0] / 2.5, distances[1]

    nx_topology = topology
    all_nodes = nx_topology.nodes
    score_per_node = defaultdict(float)
    for error in tqdm(errors):
        sensor, score = error
        print("Distributing error for node", sensor, "with score", score)
        start = translation[sensor]
        for i, node in enumerate(all_nodes):
            nr_of_hops = distances[sensor][i]
            n_score = score * max(0.0, (avg_dist - nr_of_hops) / avg_dist)
            if n_score > score_per_node[node]:
                score_per_node[node] = n_score

    # clean and normalize
    max_score = max(score_per_node.values())
    for n in score_per_node:
        score_per_node[n] *= (1 / max_score)

    return score_per_node


def output(name, leak_id, score_map, to_file=True):
    edge_scores = dict()
    edge_map = {tuple(pipe.split(".")[:2]): pipe for pipe in pipes}
    for i, e in enumerate(topology_alt.edges()):
        re = (reverse_translation[e[0]], reverse_translation[e[1]])
        edge_scores[edge_map[re]] = (score_map[e[0]] + score_map[e[1]]) / 2

    if to_file:
        with open(DATA_ROOT + "/" + RESULTS + "/" + name + "/" + str(leak_id) + '_results.csv', 'w') as csvfile:
            wr = csv.writer(csvfile, delimiter=',')
            wr.writerow(["pipe_id", "score"])
            for e in edge_scores:
                wr.writerow([e, edge_scores[e]])
        with open(DATA_ROOT + "/" + RESULTS + "/" + name + "/" + str(leak_id) + '_results.json', 'w') as jsonfile:
            json.dump(edge_scores, jsonfile)
    return edge_scores
