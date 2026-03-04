from copy import copy
import bisect
import numpy as np
import pandas as pd
import heapq
import math

from Structure.model import FSPN
from Learning.statistics import get_structure_stats

# FJBuckets_K = 13
# #FJBuckets_K = 113
# #FJBuckets_K = 137
# #FJBuckets_K = 151
# #FJBuckets_K = 211
# #FJBuckets_K = 787
# #FJBuckets_K = 877
# #FJBuckets_K = 977
# #FJBuckets_K = 911
# #FJBuckets_K = 1123
# #FJBuckets_K = 1057
# #FJBuckets_K = 1277
# #FJBuckets_K = 1103
# #FJBuckets_K = 2111
# #FJBuckets_K = 10007
# #FJBuckets_K = 1000007
# def set_FJBuckets_K(val):
#     global FJBuckets_K
#     FJBuckets_K = val

# def hash_FJBuckets(x, mini, maxi):
#     global FJBuckets_K
#     #print(FJBuckets_K)
#     #print('hash_FJBuckets mini={},maxi={}', mini, maxi)
#     #exit(-1)
#     if x == float('-inf'):
#         return -1
#     elif x == float('inf'):
#         return FJBuckets_K + 1
#     else:
#         return int((x - mini) / (maxi - mini + 1) * FJBuckets_K)

class Partition:
    def __init__(self, n: float, ndv: list):
        self.n = n
        self.ndv = ndv
    
    def _partition_info(self):
        return 'Partition( n={} , ndv={} )'.format(self.n, self.ndv)

    def _print(self):
        print(self.__partition_info())

def create_partition(data: pd.DataFrame, joinkeys: list):
    n = data.shape[0]
    ndv = [None if i is None else data[i].nunique() for i in joinkeys]
    return Partition(n, ndv)

class Table_Models_Partitions:
    def __init__(self, name: str, partitions_loc_no: list, partitions_allNone: np.ndarray, qspns_allNone: np.ndarray):
        self.name = name
        #self.partitions_loc_no = [{None: ()} if i is None else {} for i in join_keys_colname]
        self.partitions_loc_no = partitions_loc_no
        self.partitions = partitions_allNone
        self.qspns = qspns_allNone
    #partitions_loc_no: list(dict) #[{6:(0,), 8: (1,)}, {1: (0,), 2: (1,), 4: (2,), 6: (3,)}, {None: ()}, {2: (0,), 3: (1,), 5: (2,)}]
    #partitionss: ndarray #ndarray with type=Partition, index is no (not loc)
    #qspns: ndarray #ndarray with type=QSPN, index is no (not loc)
    def _print(self, partitions_info=False):
        print('************************\n{}:'.format(self.name))
        print(self.partitions_loc_no)
        print('{} partitions, {} qspns'.format(self.partitions.shape, self.qspns.shape))
        if partitions_info:
            partitions_print = np.full(self.partitions.shape, None)
            for ith, i in np.ndenumerate(self.partitions):
                partitions_print[ith] = None if i is None else i._partition_info()
            print(partitions_print)
            # qspns_print = np.full(self.partitions.shape, None)
            # for ith, i in np.ndenumerate(self.qspns):
            #     qspns_print[ith] = None if i is None else get_structure_stats(i)
            print(self.qspns)
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

# def enum_all_loc_no_speval_no(all_selected_loc: list, partitions_loc_no: list, speval_partitions_loc_no: list, partitions: np.ndarray, speval_partitions: np.ndarray, ith, loc, no, speval_no):
#     if ith >= len(all_selected_loc):
#         yield_no = None if no is None else partitions[no]
#         yield_speval_no = None if speval_no is None else speval_partitions[speval_no]
#         if yield_no is not None or yield_speval_no is not None:
#             yield loc, yield_no, yield_speval_no
#     else:
#         partitions_loc_no_ith = partitions_loc_no[ith]
#         speval_partition_loc_no_ith = None if speval_partitions_loc_no is None else speval_partitions_loc_no[ith]
#         for k in all_selected_loc[ith]:
#             v = partitions_loc_no_ith[k] if k in partitions_loc_no_ith else None
#             speval_v = speval_partition_loc_no_ith[k] if speval_partition_loc_no_ith is not None and k in speval_partition_loc_no_ith else None
#             nex_no = None if v is None else no+v
#             nex_speval_no = None if speval_v is None else speval_no+speval_v
#             yield from enum_all_loc_no_speval_no(all_selected_loc, partitions_loc_no, speval_partitions_loc_no, partitions, speval_partitions, ith+1, loc+(k,), nex_no, nex_speval_no)

def intersection_selected_loc_no_forward(node: CEtree_Node):
    th_joinkeys_intersected_loc = [set(i.keys()) for i in node.selected_loc_no]
    for ith, i in enumerate(node.selected_loc_no):
        print(node.table_models_partitions.name, ith, i)
    for i in node.children:
        edge_node = i[0]
        edge_th_joinkey = i[1]
        print(edge_node.table_models_partitions.name, edge_th_joinkey, edge_node.table_models_partitions.partitions_loc_no[edge_th_joinkey])
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
    for ith, i in enumerate(node.selected_loc_no):
        print(node.table_models_partitions.name, ith, i, node.table_models_partitions.partitions_loc_no)
    for i in node.children:
        edge_node = i[0]
        edge_th_joinkey = i[1]
        print(edge_node.table_models_partitions.name, edge_th_joinkey, edge_node.selected_loc_no[edge_th_joinkey], edge_node.table_models_partitions.partitions_loc_no)
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
    
    # def set_JOINKEY_PARTITION_WIDTH_LIMIT(self, val):
    #     assert self.JOINKEY_PARTITION_WIDTH_LIMIT is None
    #     self.JOINKEY_PARTITION_WIDTH_LIMIT = val
    
    # def set_global_joinkeys_range(self, ranges):
    #     assert self.global_joinkeys_range is None
    #     self.global_joinkeys_range = copy(ranges)
    
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
        if info_level > 0:
            for i in self.table_models_partitions.values():
                i._print(info_level > 1)
        print()
        print('speval_joinkey={}'.format(self.speval_joinkey))
        print('speval_joinkey_partition_width', self.speval_joinkey_partition_width)
        print('speval_table_models_partitions', self.speval_table_models_partitions)
        if info_level > 0:
            for i in self.speval_table_models_partitions.values():
                i._print(info_level > 1)
        print('###')
    
    # def calc_tables_domain(self, table_names: list, datas: list):
    #     for table_name, data in zip(table_names, datas):
    #         data_min = data.min(axis=0, skipna=True)
    #         data_max = data.max(axis=0, skipna=True)
    #         self.table_domain[table_name] = [(data_min[i], data_max[i]) for i in range(data.shape[1])]
    #         self.table_cardinality[table_name] = data.shape[0]
    
    # def set_table_columns(self, table_name: str, columns: list, data):
    #     assert data.shape[1] == len(columns)
    #     self.table_columns[table_name] = {c: i for i, c in enumerate(columns)}
    
    # def set_table_column_domain(self, table_name: str, column_name: str, mini=None, maxi=None):
    #     if mini is not None and maxi is not None:
    #         scope_col = self.table_columns[table_name][column_name]
    #         self.table_domain[table_name][scope_col] = (mini, maxi)
    
    # def set_table_qspn_model(self, table: str, qspn):
    #     self.table_qspn_model[table] = qspn
    
    # def get_table_column_scope(self, table: str, columns: list):
    #     scope = [self.table_columns[table][i] for i in columns]
    #     return scope

def loadmodel_MultiQSPN_table_models_partitions_qspns(mqspn: MultiQSPN):
    #mqspn._print(info_level=2)
    #print()
    table_has_speval_joinkey = {}
    for i in mqspn.table_models_partitions.values():
        if None in i.partitions_loc_no[mqspn.speval_joinkey]:
            table_has_speval_joinkey[i.name] = False
        else:
            table_has_speval_joinkey[i.name] = True
        for jth, j in np.ndenumerate(i.qspns):
            if j is not None:
                model = FSPN()
                model.model = j
                model.store_factorize_as_dict()
                i.qspns[jth] = model
    # print(table_has_speval_joinkey)
    # exit(-1)
    for i in mqspn.speval_table_models_partitions.values():
        if table_has_speval_joinkey[i.name]:
            for jth, j in np.ndenumerate(i.qspns):
                if j is not None:
                    model = FSPN()
                    model.model = j
                    model.store_factorize_as_dict()
                    i.qspns[jth] = model
    #mqspn._print(info_level=2)
    return mqspn