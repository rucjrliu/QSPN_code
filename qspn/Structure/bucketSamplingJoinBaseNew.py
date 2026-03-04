from copy import copy
import bisect
import numpy as np
import pandas as pd
import heapq
import math

class BucketSample:
    def __init__(self, data: np.ndarray, sampleratio=0.1):
        sampled_rows = np.random.choice(data.shape[0], size=max(1, int(sampleratio * data.shape[0])), replace=False)
        self.data = data[sampled_rows]
        self.n = self.data.shape[0]
        self.ratio = data.shape[0] / self.n
        print('Constructed From data_{} to BucketSample_({},n={},ratio={})'.format(data.shape, self.data.shape, self.n, self.ratio))
    
    def probability(self, filtering_predicates: tuple, query_scope: set):
        #selected_data = self.data[(self.data >= filtering_predicates[0][0]) & (self.data <= filtering_predicates[1][0])]
        selected_data = self.data
        for i in query_scope:
            selected_data = selected_data[(selected_data[: , i] >= filtering_predicates[0][0][i]) & (selected_data[: , i] <= filtering_predicates[1][0][i])]
        return selected_data.shape[0] * self.ratio / self.n

class Partition:
    def __init__(self, n, ndv, speval_n, speval_ndv, qspn=None, speval_qspn=None):
        self.n = n
        self.ndv = ndv
        self.speval_n = speval_n
        self.speval_ndv = speval_ndv
        self.qspn = qspn
        self.speval_qspn = speval_qspn
    
    def _partition_info(self):
        return 'Partition( n={} , ndv={} , speval_n={} , speval_ndv={} , qspn={} , speval_qspn={})'.format(self.n, self.ndv, self.speval_n, self.speval_ndv, self.qspn, self.speval_qspn)

    def _print(self):
        print(self._partition_info())

class Table_Models_Partitions:
    def __init__(self, name: str, partitions_loc_no: list, partitions_allNone: np.ndarray):
        self.name = name
        #self.partitions_loc_no = [{None: ()} if i is None else {} for i in join_keys_colname]
        self.partitions_loc_no = partitions_loc_no
        self.partitions = partitions_allNone
    #partitions_loc_no: list(dict) #[{6:(0,), 8: (1,)}, {1: (0,), 2: (1,), 4: (2,), 6: (3,)}, {None: ()}, {2: (0,), 3: (1,), 5: (2,)}]
    #partitionss: ndarray #ndarray with type=Partition, index is no (not loc)
    #qspns: ndarray #ndarray with type=QSPN, index is no (not loc)
    def _print(self, partitions_info=False):
        print('************************\n{}:'.format(self.name))
        print(self.partitions_loc_no)
        print('{} partitions'.format(self.partitions.shape))
        if partitions_info:
            partitions_print = np.full(self.partitions.shape, None)
            for ith, i in np.ndenumerate(self.partitions):
                partitions_print[ith] = None if i is None else i._partition_info()
            print(partitions_print)
            # qspns_print = np.full(self.partitions.shape, None)
            # for ith, i in np.ndenumerate(self.qspns):
            #     qspns_print[ith] = None if i is None else get_structure_stats(i)
        print('************************')

class CEtree_Node:
    def __init__(self, table_models_partitions: dict):
        self.edges = None
        self.fa = None
        self.children = None

        self.table_models_partitions = table_models_partitions
        self.selected_loc_no = None
        self.calculated_partitions = None

    def set_edges(self, edges: list):
        self.edges = edges
    
    def set_fa_children(self, fa: tuple, children: list):
        self.fa = fa
        self.children = children
        self.edges = None
    
    def _print(self):
        print('************************\n{}:'.format(self.table_models_partitions.name))
        if self.edges is None:
            if self.fa[1] == -1:
                print('fa:{}'.format(self.fa))
            else:
                print('fa:{}'.format((self.fa[0].table_models_partitions.name, self.fa[1])))
            print('children:{}'.format([(i[0].table_models_partitions.name, i[1]) for i in self.children]))
            print('selected_loc_no:{}'.format(self.selected_loc_no))
            if self.calculated_partitions is not None:
                partitions_print = np.full(self.calculated_partitions.shape, None)
                for ith, i in np.ndenumerate(self.calculated_partitions):
                    partitions_print[ith] = None if i is None else i._partition_info()
                print(partitions_print)
        else:
            print('edges:{}'.format([(i[0].table_models_partitions.name, i[1]) for i in self.edges]))
        print('************************')

def loc2no(nos: list, loc: tuple):
    no = ()
    for i, j in enumerate(loc):
        no += nos[i][j]
    return no

def enum_all_loc_no(partitions_loc_no: list, partitions: np.ndarray, ith, loc, no):
    if ith >= len(partitions_loc_no):
        if partitions[no] is not None:
            yield loc, no
    else:
        for k, v in partitions_loc_no[ith].items():
            yield from enum_all_loc_no(partitions_loc_no, partitions, ith+1, loc+(k,), no+v)

def intersection_selected_loc_no_forward(node: CEtree_Node):
    th_joinkeys_intersected_loc = [set(i.keys()) for i in node.selected_loc_no]
    # for ith, i in enumerate(node.selected_loc_no):
    #     print(node.table_models_partitions.name, ith, i)
    for i in node.children:
        edge_node = i[0]
        edge_th_joinkey = i[1]
        # print(edge_node.table_models_partitions.name, edge_th_joinkey, edge_node.table_models_partitions.partitions_loc_no[edge_th_joinkey])
        th_joinkeys_intersected_loc[edge_th_joinkey] = th_joinkeys_intersected_loc[edge_th_joinkey] & edge_node.table_models_partitions.partitions_loc_no[edge_th_joinkey].keys()
    
    for ith, i in enumerate(th_joinkeys_intersected_loc):
        # print('debug', node.selected_loc_no[ith], node.table_models_partitions.partitions_loc_no[ith])
        node.selected_loc_no[ith] = {j: node.table_models_partitions.partitions_loc_no[ith][j] for j in i}
        # print('debug', node.selected_loc_no[ith], node.table_models_partitions.partitions_loc_no[ith])
        # input()
    
    for i in node.children:
        edge_node = i[0]
        edge_th_joinkey = i[1]
        assert edge_node.selected_loc_no is None, 'selected_loc_no not None before visited'
        edge_node.selected_loc_no = [{k: edge_node.table_models_partitions.partitions_loc_no[jth][k] for k in j} if edge_th_joinkey == jth else edge_node.table_models_partitions.partitions_loc_no[jth] for jth, j in enumerate(th_joinkeys_intersected_loc)]

def intersection_selected_loc_no_back(node: CEtree_Node):
    th_joinkeys_intersected_loc = [set(i.keys()) for i in node.selected_loc_no]
    # for ith, i in enumerate(node.selected_loc_no):
    #     print(node.table_models_partitions.name, ith, i, node.table_models_partitions.partitions_loc_no)
    for i in node.children:
        edge_node = i[0]
        edge_th_joinkey = i[1]
        # print(edge_node.table_models_partitions.name, edge_th_joinkey, edge_node.selected_loc_no[edge_th_joinkey], edge_node.table_models_partitions.partitions_loc_no)
        th_joinkeys_intersected_loc[edge_th_joinkey] = th_joinkeys_intersected_loc[edge_th_joinkey] & edge_node.selected_loc_no[edge_th_joinkey].keys()
    
    for ith, i in enumerate(th_joinkeys_intersected_loc):
        # print('debug', node.selected_loc_no[ith], node.table_models_partitions.partitions_loc_no[ith])
        node.selected_loc_no[ith] = {j: node.table_models_partitions.partitions_loc_no[ith][j] for j in i}
        # print('debug', node.selected_loc_no[ith], node.table_models_partitions.partitions_loc_no[ith])
        # input()

# def enum_all_loc_no(partitions_loc_no: list):
#     partitions_loc_no_items = [d.items() for d in partitions_loc_no]
#     return ((tuple(i[0] for i in locnos if i[0] is not None), tuple(itt.chain(*[i[1] for i in locnos]))) for locnos in itt.product(*partitions_loc_no_items))

class MultiQSPN:
    def __init__(self, table_columns: dict, table_joinkeys: dict, joinkey_partition_width_limit: int, ranges: dict, table_th_joinkeys: dict, speval_joinkey=None, speval_joinkey_partition_width=None):
        self.table_columns = table_columns
        self.table_joinkeys = {t: {j: jth for jth, j in enumerate(joinkeys)} for t, joinkeys in table_joinkeys.items()}
        self.table_filtering_columns = {i: {} for i in table_columns}
        for t, cs in table_columns.items():
            t_join_columns = set(table_th_joinkeys[t])
            t_filtering_columns_list = []
            for c in cs:
                if c not in t_join_columns:
                    t_filtering_columns_list.append(c)
            self.table_filtering_columns[t] = {c: i for i, c in enumerate(t_filtering_columns_list)}
        # self.table_domain = {}
        # self.table_cardinality = {}
        self.table_models_partitions = {}
        
        self.speval_joinkey = speval_joinkey
        if speval_joinkey is None:
            self.speval_joinkey_partition_width = None
            self.speval_table_models_partitions = None
        else:
            self.speval_joinkey_partition_width = speval_joinkey_partition_width
            self.speval_table_models_partitions = {}
        
        self.JOINKEY_PARTITION_WIDTH_LIMIT = joinkey_partition_width_limit
        self.global_joinkeys_range = ranges #list(tuple)
    
    def get_joinkey_loc_projected(self, join_key_th, key):
        join_key_th_range_len = self.global_joinkeys_range[join_key_th][1] - self.global_joinkeys_range[join_key_th][0] + 1
        #partition_n = math.ceil(join_key_th_range_len / self.JOINKEY_PARTITION_WIDTH_LIMIT)
        loc_projected = (key - self.global_joinkeys_range[join_key_th][0]) // join_key_th_range_len
        #assert loc_projected < partition_n
        return loc_projected
    
    def calc_partition_width(self, joinkey_th, loc_projected):
        width_limit = self.JOINKEY_PARTITION_WIDTH_LIMIT
        mini = width_limit * loc_projected + self.global_joinkeys_range[joinkey_th][0]
        maxi = min(self.global_joinkeys_range[joinkey_th][1], mini + width_limit - 1)
        return maxi - mini + 1

    def _print(self, info_level=0):
        print('###')
        print()
        print('table_columns:', self.table_columns)
        print()
        print('table_joinkeys:', self.table_joinkeys)
        print()
        print('table_filtering_columns:', self.table_filtering_columns)
        print()
        print('JOINKEY_PARTITION_WIDTH_LIMIT =', self.JOINKEY_PARTITION_WIDTH_LIMIT)
        print()
        print('global_joinkeys_range', self.global_joinkeys_range)
        print()
        print('table_models_partitions', self.table_models_partitions)
        print()
        print('speval_joinkey', self.speval_joinkey)
        print('speval_joinkey_partition_width', self.speval_joinkey_partition_width)
        print()
        if info_level > 0:
            for i in self.table_models_partitions.values():
                i._print(info_level > 1)
        print('###')
