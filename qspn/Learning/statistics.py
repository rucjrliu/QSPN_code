from collections import Counter
import numpy as np
from Structure.nodes import get_nodes_by_type, print_spn_structure, Sum, Product, Factorize, Leaf, get_number_of_edges, get_depth, Node, bfs, QSum
import logging
from Structure.leaves.fspn_leaves.Multi_Histograms import Multi_histogram
import sys
import pdb

logger = logging.getLogger(__name__)

def print_spn_structure(root):
    print('{}-rooted'.format(root))
    edges = print_spn_structure(root, Node)
    for i in edges:
        print(i)
    print()

def get_structure_stats_dict(node):
    nodes = get_nodes_by_type(node, Node)
    num_nodes = len(nodes)

    node_types = dict(Counter([type(n) for n in nodes]))

    edges = get_number_of_edges(node)
    layers = get_depth(node)

    params = 0
    for n in nodes:
        if isinstance(n, Sum):
            params += len(n.children)
        if isinstance(n, Leaf):
            params += len(n.parameters)

    result = {"nodes": num_nodes, "params": params, "edges": edges, "layers": layers, "count_per_type": node_types}
    return result


def get_structure_stats(node):
    num_nodes = len(get_nodes_by_type(node, Node))
    sum_nodes = get_nodes_by_type(node, Sum)
    qsum_nodes = get_nodes_by_type(node, QSum)
    n_qsum_nodes = len(qsum_nodes)
    n_sum_nodes = len(sum_nodes)
    n_prod_nodes = len(get_nodes_by_type(node, Product))
    n_fact_nodes = len(get_nodes_by_type(node, Factorize))
    leaf_nodes = get_nodes_by_type(node, Leaf)
    n_leaf_nodes = len(leaf_nodes)
    multi_leaf_nodes = get_nodes_by_type(node, Multi_histogram)
    n_multi_leaf_nodes = len(multi_leaf_nodes)
    edges = get_number_of_edges(node)
    layers = get_depth(node)
    params = 0
    for n in sum_nodes:
        params += len(n.children)
    
    for n in qsum_nodes:
        print(n.id)
        for i in n.cluster_centers:
            print(i)
        #print(n.cluster_centers[0])
        #print(n.cluster_centers[1])
        print()

    # all_nodes = sum_nodes+get_nodes_by_type(node, Product)+get_nodes_by_type(node, Factorize)+leaf_nodes
    # print("all nodes: ", len(all_nodes))
    # all_params = 0
    # node_size = 0
    # for n in all_nodes:
    #     all_params += len(n.parameters)

    # print("all_params: ", all_params)
    # print("node size", node_size, sys.getsizeof(node))

    l_params = 0
    l_2 = 0
    l_3 = 0
    for n in multi_leaf_nodes:
        if len(n.breaks) == 2:
            l_2 += 1
        elif len(n.breaks) > 2:
            l_3 += 1
        l_params += np.size(n.pdf) + np.size(n.cdf)
    print(l_params, l_2, l_3)
    #pdb.set_trace()
    return """---Structure Statistics---
# nodes               %s
    # sum nodes       %s
    # qsplit nodes    %s
    # factorize nodes %s
    # prod nodes      %s
    # leaf nodes      %s
    # multileaf nodes %s
# params              %s
# edges               %s
# layers              %s""" % (
        num_nodes,
        n_sum_nodes,
        n_qsum_nodes,
        n_fact_nodes,
        n_prod_nodes,
        n_leaf_nodes,
        n_multi_leaf_nodes,
        params,
        edges,
        layers,
    )

def get_range_states(node):
    def print_range(n):
        if isinstance(n, Leaf):
            print(n.range)

    bfs(node, print_range)
    return None

def get_scope_states(node):
    def print_scope(n):
        if isinstance(n, Leaf):
            print(n.scope)

    bfs(node, print_scope)
    return None
