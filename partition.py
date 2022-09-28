import uuid
from collections import defaultdict

import numpy as np
from ortools.linear_solver import pywraplp
from scipy.sparse.csgraph import connected_components

from utils import *
from static import *


class TreeNode(object):
    def __init__(self, val=None):
        super().__init__()
        self.nid = str(uuid.uuid4())
        self.val = val
        self.children = []

    def add_child(self, val):
        self.children.append(TreeNode(val))
        return self.children[-1]

    def is_leaf(self):
        return len(self.children) == 0


class Tree(object):

    def __init__(self, n):
        super().__init__()
        self.root = n
        self.all_nodes = {self.root.nid: self.root}
        self.leaves = []

    def add_node(self, p, val):
        child = p.add_child(val)
        if p in self.leaves:
            self.leaves.remove(p)
        if child.is_leaf():
            self.leaves.append(child)
        self.all_nodes[child.nid] = child
        return child

    def update_node(self, nid, val):
        node = self.all_nodes[nid]
        node.val = val

    def get_leaves(self):
        return self.leaves


def tree():
    """
    Recursively construct a tree of partitions.
    :return:
    """
    g = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "nx_topology_ud.pkl"))
    pipe_header = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "pipe_header.pkl"))
    rtranslation = load_from_pickle(os.path.join(DATA_ROOT, RESOURCES, "reverse_translation.pkl"))
    for name in pipe_header:
        if name.split(".")[-1] != "1":
            print(name)

    tr = Tree(TreeNode())

    partitions, A, n = partition(g)
    total_nodes = len(g.nodes())

    def one_step_down(partitions, A, n, tr_node):
        new_boundary_edges = list()
        new_edge_map = dict()

        n1 = g.edges()[0]
        n2 = g.edges()[1]
        edge_map = {tuple(pipe.split(".")[:2]): pipe for pipe in pipe_header}
        for i, e in enumerate(zip(n1, n2)):
            node1, node2 = e[0].item(), e[1].item()
            if node1 in partitions[0] and node2 in partitions[1] or \
               node1 in partitions[1] and node2 in partitions[0]:
                new_boundary_edges.append((pipe_header.index(edge_map[(rtranslation[node1], rtranslation[node2])])))
                new_edge_map[new_boundary_edges[-1]] = (node1, node2)

        for edge in new_edge_map:
            n1, n2 = new_edge_map[edge]
            A[n.index(n1), n.index(n2)] = 0
            A[n.index(n2), n.index(n1)] = 0

        n_components, labels = connected_components(A, directed=False)
        # gather vertex indices per component index
        components = defaultdict(list)
        for i, component in enumerate(labels):
            components[component].append(n[i])
        all_component_nodes = []

        # sanity check
        print("performing sanity check...")
        for c in components:
            all_component_nodes.extend(components[c])
        for node in n:
            if node not in all_component_nodes:
                print(node)
            assert node in all_component_nodes
        print("done...")

        # update current node in the tree
        tr.update_node(tr_node.nid, (components, new_boundary_edges, new_edge_map))
        for component in components:
            (p1, p2), new_A, nodes = partition(g, components[component])
            if len(nodes) > max(total_nodes // 10, 10):
                # add new empty child node
                child = tr.add_node(tr_node, None)
                # get values for child node
                one_step_down((p1, p2), new_A, nodes, child)

    one_step_down(partitions, A, n, tr.root)
    return tr


def partition(g, leaky_part=None):
    """
    Partition a given (sub-)network.
    :param g:
    :param leaky_part:
    :return:
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Create the incidence matrix. N x M matrix with N number of nodes and M number of edges.
    # Each column contains a -1 and a 1. -1 corresponds to the source of edge, 1 to the destination
    if leaky_part is not None:
        nodes = [n.item() for n in g.nodes() if n.item() in leaky_part]
        edges = [(e[0].item(), e[1].item()) for e in zip(g.edges()[0], g.edges()[1])
                 if e[0].item() in leaky_part and e[1].item() in leaky_part]
    else:
        nodes = [n.item() for n in g.nodes()]
        edges = [(e[0].item(), e[1].item()) for e in zip(g.edges()[0], g.edges()[1])]
    J = np.zeros((len(nodes), len(edges)))
    for i, edge in enumerate(edges):
        J[nodes.index(edge[0]), i] = -1
        J[nodes.index(edge[1]), i] = 1

    # We have 3 arrays of boolean decision variables: x (n-dimensional) and t1 + t2 (m-dimensional)
    x = [solver.BoolVar('x_{}'.format(i)) for i in range(len(nodes))]
    t1 = [solver.BoolVar('t1_{}'.format(i)) for i in range(len(edges))]
    t2 = [solver.BoolVar('t2_{}'.format(i)) for i in range(len(edges))]

    # Add the constraints
    for i in range(len(edges)):
        solver.Add(t1[i] - t2[i] == sum([J[j, i] * x[j] for j in range(len(nodes))]))
        solver.Add(t1[i] + t2[i] <= 1)
    solver.Add(sum([x[i] for i in range(len(nodes))]) == len(nodes) // 2)

    # Specify our objectives
    solver.Minimize(sum(t1) + sum(t2))
    # solver.Minimize(-2 * sum(x))
    status = solver.Solve()
    print("status", status)

    p1, p2 = set(), set()
    for i, n in enumerate(nodes):
        if x[i].solution_value() > 0:
            p1.add(n)
        else:
            p2.add(n)

    A = np.zeros((len(nodes), len(nodes)))
    for i, edge in enumerate(edges):
        A[nodes.index(edge[0]), nodes.index(edge[1])] = 1
        A[nodes.index(edge[1]), nodes.index(edge[0])] = 1

    # check if everything makes sense
    combined_p = p1.union(p2)
    for n in nodes:
        if n not in combined_p:
            print(n)
        assert n in combined_p
    for n in leaky_part:
        if n not in combined_p:
            print(n)
        assert n in combined_p

    return (p1, p2), A, nodes


def get_partitions():
    """
    Get lowest-level partitions.
    :return:
    """
    tr = tree()

    # amass partitions
    partitions = defaultdict(list)
    for i, leaf in enumerate(tr.get_leaves()):
        for component in leaf.val[0]:
            partitions[i].extend(leaf.val[0][component])

    return partitions
