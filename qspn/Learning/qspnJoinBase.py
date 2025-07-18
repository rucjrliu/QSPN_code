from copy import deepcopy
import bisect
import numpy as np
import pandas
import heapq
import math

from Learning.structureLearning import calculate_RDC
from Learning.qspnJoinDynFactor import set_dynfactor_function, calc_dynfactor_ratio

ALGOS = ['factorjoin', 'mul1', 'mul1_nomcv', 'mul1+mul2', 'mixed_fj_mul1', 'ada_fj', 'ada_fj_mul1+mul2', 'fj_ada_factor', 'fj_dyn_factor']
#ALGO = ALGOS[0]
ALGO = ALGOS[7]
SHOW_VE = False

def mqspn_sum_prune_by_datadomain(node, childi, query):
    cover = True
    for i, (l, r) in enumerate(zip(node.node_error[1]['data_min'][childi], node.node_error[1]['data_max'][childi])):
        if query[0][0, node.scope[i]] > r or query[1][0, node.scope[i]] < l:
            return np.array([0.0])
    return None

def multi_table_RDC(A, B, A_join_col, B_join_col, threshold=0.3):
    corr = False
    #gen matrix[cols of tA, cols of tB]
    #for i in tA for j in tB if (i != A_join_col or j != B_join_col) and join mat[i][j] > threshold corr=True
    #no need to implemented now
    corr = True
    return corr

def join_table_origin(A_tbname, A, A_join_col, B_tbname, B, B_join_col):
    assert A is not None and B is not None
    A_join_col = '{}.{}'.format(A_tbname, A_join_col)
    B_join_col = '{}.{}'.format(B_tbname, B_join_col)
    A.columns = ['{}.{}'.format(A_tbname, col) for col in A.columns]
    B.columns = ['{}.{}'.format(B_tbname, col) for col in B.columns]
    A['__index_A__'] = A.index
    B['__index_B__'] = B.index
    AB = pandas.merge(A, B, left_on=A_join_col, right_on=B_join_col, how='outer')
    AB['__join_key__'] = AB[A_join_col].combine_first(AB[B_join_col])
    count_index_A = AB['__index_A__'].value_counts()
    count_index_B = AB['__index_B__'].value_counts()
    AB['__index_A__'] = AB['__index_A__'].fillna(-1).astype(int)
    AB['__index_B__'] = AB['__index_B__'].fillna(-1).astype(int)
    AB['__dsf_A__'] = AB['__index_A__'].apply(lambda x: calculate_downscale_factor(x, count_index_A))
    AB['__dsf_B__'] = AB['__index_B__'].apply(lambda x: calculate_downscale_factor(x, count_index_B))
    downscale_factor_A = AB['__dsf_A__'].values
    downscale_factor_B = AB['__dsf_B__'].values
    AB.drop(columns=['__index_A__', '__index_B__', A_join_col, B_join_col, '__dsf_A__', '__dsf_B__'], inplace=True)
    return AB, downscale_factor_A, downscale_factor_B
def join_table(A_tbname, A, A_join_col, B_tbname, B, B_join_col):
    assert A is not None and B is not None
    A_join_col = '{}.{}'.format(A_tbname, A_join_col)
    B_join_col = '{}.{}'.format(B_tbname, B_join_col)
    A.columns = ['{}.{}'.format(A_tbname, col) for col in A.columns]
    B.columns = ['{}.{}'.format(B_tbname, col) for col in B.columns]
    A['__index_A__'] = A.index
    A = A.set_index(A_join_col)
    B['__index_B__'] = B.index
    B = B.set_index(B_join_col)
    #print(A)
    #print(B)
    #exit(-1)
    #AB = pandas.merge(A, B, how='outer', left_index=True, right_index=True)
    #AB = pandas.concat([A, B], axis=1, join='outer')
    AB = A.join(B, how='outer')
    AB.reset_index(inplace=True)
    AB.rename(columns={'index': '__join_key__'}, inplace=True)
    #print(AB)
    #exit(-1)
    #AB['__join_key__'] = AB[A_join_col].combine_first(AB[B_join_col])
    count_index_A = AB['__index_A__'].value_counts()
    count_index_B = AB['__index_B__'].value_counts()
    AB['__index_A__'] = AB['__index_A__'].fillna(-1).astype(int)
    AB['__index_B__'] = AB['__index_B__'].fillna(-1).astype(int)
    AB['__dsf_A__'] = AB['__index_A__'].apply(lambda x: calculate_downscale_factor(x, count_index_A))
    AB['__dsf_B__'] = AB['__index_B__'].apply(lambda x: calculate_downscale_factor(x, count_index_B))
    downscale_factor_A = AB['__dsf_A__'].values
    downscale_factor_B = AB['__dsf_B__'].values
    #AB.drop(columns=['__index_A__', '__index_B__', A_join_col, B_join_col, '__dsf_A__', '__dsf_B__'], inplace=True)
    AB.drop(columns=['__index_A__', '__index_B__', '__dsf_A__', '__dsf_B__'], inplace=True)
    return AB, downscale_factor_A, downscale_factor_B
def get_join_table(A_tbname, A, A_join_col, B_tbname, B, B_join_col):
    #return join_table(A_tbname, A, A_join_col, B_tbname, B, B_join_col)
    return join_table(A_tbname, A.copy(), A_join_col, B_tbname, B.copy(), B_join_col)
def try_join_table():
    dA = pandas.DataFrame({'keyA': [2, 3, 3, 4], 'x': [1, 2, 3, 4]})
    dB = pandas.DataFrame({'keyB': [2, 2, 2, 3, 3, 5], 'y': [10, 20, 30, 40, 50, 60]})
    ddA = dA.copy()
    ddB = dB.copy()
    print(join_table_origin('A', dA, 'keyA', 'B', dB, 'keyB'))
    AB, dsfA, dsfB = join_table('A', ddA, 'keyA', 'B', ddB, 'keyB')
    print(AB)
    print(AB.values)
    #print(np.isnan(AB.values[:,2]))
    print(dsfA)
    print(dsfB)
    exit(-1)

FJBuckets_K = 13
#FJBuckets_K = 113
#FJBuckets_K = 137
#FJBuckets_K = 151
#FJBuckets_K = 211
#FJBuckets_K = 787
#FJBuckets_K = 877
#FJBuckets_K = 977
#FJBuckets_K = 911
#FJBuckets_K = 1123
#FJBuckets_K = 1057
#FJBuckets_K = 1277
#FJBuckets_K = 1103
#FJBuckets_K = 2111
#FJBuckets_K = 10007
#FJBuckets_K = 1000007
def set_FJBuckets_K(val):
    global FJBuckets_K
    FJBuckets_K = val

def hash_FJBuckets(x, mini, maxi):
    global FJBuckets_K
    #print(FJBuckets_K)
    #print('hash_FJBuckets mini={},maxi={}', mini, maxi)
    #exit(-1)
    if x == float('-inf'):
        return -1
    elif x == float('inf'):
        return FJBuckets_K + 1
    else:
        return int((x - mini) / (maxi - mini + 1) * FJBuckets_K)

class MultiQSPN:
    def __init__(self):
        self.table_columns = {}
        self.table_domain = {}
        self.table_cardinality = {}
        self.table_rdc_adjacency_matrix = {}
        self.table_qspn_model = {}
        self.bigtable_columns = None #dict
        self.bigtable_domain = None #dict
        self.bigtable_cardinality = None #dict
        self.bigtable_qspn_model = None #dict
        self.bigtable_join_info = None #dict, eg: {'A': ([(A,B), (A,C)], joincolA), 'B': ([(A,B)], joincolB), 'C': ([(A,C)], joincolC)}
        #self.bigtable_join_info = None #dict: {(tbAname, tbBname): (joincolA, joincolB), ...}
    
    def init_bigtable(self):
        self.bigtable_columns = {}
        self.bigtable_domain = {}
        self.bigtable_cardinality = {}
        self.bigtable_qspn_model = {}
        self.bigtable_join_info = {}
    
    # def init_bigtable_join_info(self, table_join_cols: list):
    #     self.bigtable_join_info = {t: ([], jc) for t, jc in table_join_cols}
    def set_bigtable_join_info(self, table_name_A: str, join_colA: str, table_name_B: str, join_colB: str):
        assert table_name_A != table_name_B
        if table_name_A < table_name_B:
            tA, tB = table_name_A, table_name_B
            cA, cB = join_colA, join_colB
        else:
            tA, tB = table_name_B, table_name_A
            cA, cB = join_colB, join_colA
        if tA not in self.bigtable_join_info:
            self.bigtable_join_info[tA] = ([(tA, tB)], cA)
        else:
            assert self.bigtable_join_info[tA][1] == cA and (tA, tB) not in self.bigtable_join_info[tA][0]
            self.bigtable_join_info[tA][0].append((tA, tB))
        if tB not in self.bigtable_join_info:
            self.bigtable_join_info[tB] = ([(tA, tB)], cB)
        else:
            assert self.bigtable_join_info[tB][1] == cB and (tA, tB) not in self.bigtable_join_info[tB][0]
            self.bigtable_join_info[tB][0].append((tA, tB))
        return (tA, tB)

    def calc_tables_domain(self, table_names: list, datas: list):
        for table_name, data in zip(table_names, datas):
            data_min = data.min(axis=0, skipna=True)
            data_max = data.max(axis=0, skipna=True)
            self.table_domain[table_name] = [(data_min[i], data_max[i]) for i in range(data.shape[1])]
            self.table_cardinality[table_name] = data.shape[0]
        # #print(self.table_domain)
        # minmin = None
        # maxmax = None
        # for i in self.table_domain.values():
        #     for j in i:
        #         if minmin is None or j[0] < minmin:
        #             minmin = j[0]
        #         if maxmax is None or j[1] > maxmax:
        #             maxmax = j[1]
        # for i in self.table_domain:
        #     for j in range(len(self.table_domain[i])):
        #         self.table_domain[i][j] = (minmin, maxmax)
    def calc_bigtables_domain(self, bigtable_names: list, datas: list):
        assert len(datas) == len(self.bigtable_columns)
        for table_name, data in zip(bigtable_names, datas):
            assert table_name in self.bigtable_columns
            data_min = data.min(axis=0, skipna=True)
            data_max = data.max(axis=0, skipna=True)
            self.bigtable_domain[table_name] = [(data_min[i], data_max[i]) for i in range(data.shape[1])]
            self.bigtable_cardinality[table_name] = data.shape[0]
    def set_table_columns(self, table_name: str, columns: list, data):
        assert data.shape[1] == len(columns)
        self.table_columns[table_name] = {c: i for i, c in enumerate(columns)}
    def set_bigtable_columns(self, bigtable_name: tuple, columns: list, data):
        assert data.shape[1] == len(columns)
        self.bigtable_columns[bigtable_name] = {c: i for i, c in enumerate(columns)}
    def set_table_column_domain(self, table_name: str, column_name: str, mini=None, maxi=None):
        if mini is not None and maxi is not None:
            scope_col = self.table_columns[table_name][column_name]
            self.table_domain[table_name][scope_col] = (mini, maxi)
    def set_bigtable_column_domain(self, bigtable_name: tuple, column_name: str, mini=None, maxi=None):
        if mini is not None and maxi is not None:
            scope_col = self.bigtable_columns[bigtable_name][column_name]
            self.bigtable_domain[bigtable_name][scope_col] = (mini, maxi)
    def calc_table_RDC(self, table_name: str, data, ds_context, sample_size):
        scope = [i for i in range(data.shape[1])]
        condition = []
        rdc_adjacency_matrix, scope_loc, condition_loc = calculate_RDC(data, ds_context, scope, condition, sample_size)
        # print(table_name)
        # print(rdc_adjacency_matrix)
        # print(scope_loc)
        # print(condition_loc)
        self.table_rdc_adjacency_matrix[table_name] = rdc_adjacency_matrix
    def set_table_qspn_model(self, table: str, qspn):
        self.table_qspn_model[table] = qspn
    def set_bigtable_qspn_model(self, bigtable: tuple, qspn):
        self.bigtable_qspn_model[bigtable] = qspn
    def get_table_column_scope(self, table: str, columns: list):
        scope = [self.table_columns[table][i] for i in columns]
        return scope
    def get_bigtable_column_scope(self, bigtable: tuple, columns: list):
        scope = [self.bigtable_columns[bigtable][i] for i in columns]
        return scope

def ufs(belong: dict, x):
    if belong[x] != x:
        belong[x] = ufs(belong, belong[x])
    return belong[x]

def TablesColumns_ufs(join_pairs: list):
    belong = {}
    for i in join_pairs:
        for j in i:
            if j not in belong:
                belong[j] = j
    for i in join_pairs:
        ufs(belong, i[0])
        ufs(belong, i[1])
        belong[belong[i[0]]] = belong[i[1]]
    for i in belong:
        ufs(belong, i)
    join_parameters_nodes = {}
    table_join_columns = {}
    for i in belong:
        if belong[i] not in join_parameters_nodes:
            join_parameters_nodes[belong[i]] = [i]
        else:
            join_parameters_nodes[belong[i]].append(i)
    return join_parameters_nodes

def get_join_TablesColumns_groups(mqspn: MultiQSPN, join_pairs: list):
    join_parameters_nodes = TablesColumns_ufs(join_pairs)
    #print(join_parameters_nodes)
    join_parameters_nodes_th = {}
    for i in join_parameters_nodes:
        join_parameters_nodes_th[i] = len(join_parameters_nodes_th)
    #print(join_parameters_nodes_th)
    RDC_list_mat = [[[] for j in join_parameters_nodes_th] for i in join_parameters_nodes_th]
    for i, ith in join_parameters_nodes_th.items():
        for j, jth in join_parameters_nodes_th.items():
            if i == j:
                RDC_list_mat[ith][jth].append((0, 1))
            else:
                for itc in join_parameters_nodes[i]:
                    for jtc in join_parameters_nodes[j]:
                        it, ic = itc.split('.')
                        jt, jc = jtc.split('.')
                        if it == jt:
                            ic = mqspn.table_columns[it][ic]
                            jc = mqspn.table_columns[jt][jc]
                            RDC_list_mat[ith][jth].append((mqspn.table_rdc_adjacency_matrix[ic][jc], mqspn.table_cardinality[it]))
    #print(RDC_list_mat)
    for i, ith in join_parameters_nodes_th.items():
        for j, jth in join_parameters_nodes_th.items():
            if len(RDC_list_mat[ith][jth]) == 0:
                RDC_list_mat[ith][jth] = 0
            else:
                values = [k[0] for k in RDC_list_mat[ith][jth]]
                weights = [k[1] for k in RDC_list_mat[ith][jth]]
                RDC_list_mat[ith][jth] = np.average(values, weights=weights)
    #print(RDC_list_mat)
    joined_parameters = set()
    group = []
    join_parameters_edges = [(i, j, RDC_list_mat[ith][jth]) for j, jth in join_parameters_nodes_th.items() for i, ith in join_parameters_nodes_th.items()]
    join_parameters_edges = sorted(join_parameters_edges, key=lambda z: z[2], reverse=True)
    for i in join_parameters_edges:
        if len(joined_parameters) == len(join_parameters_nodes):
            break
        if i[0] != i[1]:
            if i[0] not in joined_parameters and i[1] not in joined_parameters:
                group.append([i[0], i[1]])
                joined_parameters.add(i[0])
                joined_parameters.add(i[1])
        else:
            group.append([i[0]])
            joined_parameters.add(i[0])
    return group, join_parameters_nodes

class Bucket:
    def __init__(self):
        self.mcv = None
        self.mcv_freq = None
        self.domain = None
        self.n = 0
        self.downscale_factor = None #list
        self.downscale_factor_2col = None #value
    def calc_n_mcv(self, data):
        self.n = len(data)
        hist = {}
        for i in data:
            # if data.shape[1] == 1:
            #     ki = (i,)
            # else:
            ki = tuple(i)
            if ki not in hist:
                hist[ki] = 1
            else:
                hist[ki] += 1
        self.mcv = None
        self.mcv_freq = None
        for i in hist:
            if self.mcv is None or self.mcv_freq < hist[i]:
                self.mcv = i
                self.mcv_freq = hist[i]
        self.domain = len(hist)
    def set_n_mcv_domain(self, n, mcv, mcv_freq, domain):
        self.n = n
        self.mcv = mcv
        self.mcv_freq = mcv_freq
        self.domain = domain
    def calc_downscale_factor(self, dsf_col: list):
        #when building
        #join table A and B, calc downscale_factor col
        #dsf_col is divided individually (only sum) when multiQSPN building
        #on leaf nodes, calc_downscale_factor of each bucket by downscale_factor col (mean)
        self.downscale_factor = [float(np.mean(i)) for i in dsf_col]
        self.downscale_factor_2col = None
        for ith, i in enumerate(dsf_col):
            if ith == 0:
                self.downscale_factor_2col = i.copy()
            else:
                self.downscale_factor_2col *= i
        self.downscale_factor_2col[self.downscale_factor_2col > 0] = 1
        #for i in dsf_col:
        #    print(i.shape, i)
        #print(self.downscale_factor_2col.shape, self.downscale_factor_2col)
        self.downscale_factor_2col = float(np.mean(self.downscale_factor_2col))
        #print(self.downscale_factor_2col)
        #exit(-1)

def set_attr(x):
    #y = -3.861 * math.log(x)
    y = -4.359 * math.log(x)
    #y_s = y + 45.348
    y_s = y + 48.37
    y_s = max(y_s, 1)
    #y_s = 39
    #print(y_s)
    return (None, None, None, y_s)

class FJBuckets:
    def __init__(self):
        #print(FJBuckets_K)
        self.buckets = []
        self.buckets_keys = []
        self.scope = None
        self.downscale_factor_th = None #dict
    def calc_from_data(self, data_slice, scope: list, data_slice_fjbuckets_idx, this_table_domain: list):
        #assert data_slice is not None and data_slice.shape[1] == 1
        assert data_slice is not None
        self.buckets = []
        self.buckets_keys = []
        key_bucket = {}
        self.scope = tuple(scope)
        for dsi in data_slice:
            #print(dsi, type(dsi), dsi.shape, scope)
            i = dsi[data_slice_fjbuckets_idx]
            assert i.shape[0] == len(scope)
            #print(i)
            # if data_slice.shape[1] == 1:
            #     hash_i = (hash_FJBuckets(i, this_table_domain[scope[0]][0], this_table_domain[scope[0]][1]), )
            # else:
            hash_i = tuple([hash_FJBuckets(j, this_table_domain[sc][0], this_table_domain[sc][1]) for j, sc in zip(i, scope)])
            if hash_i not in key_bucket:
                key_bucket[hash_i] = [i]
            else:
                key_bucket[hash_i].append(i)
        #key_bucket = list(key_bucket.items())
        #key_bucket = sorted(list(key_bucket.items()), key=lambda t: t[0])
        key_bucket = sorted(list(key_bucket.items()), key=lambda t: t[0])
        for i, li in key_bucket:
            bi = Bucket()
            #print(i, np.array(li).shape)
            #exit(-1)
            bi.calc_n_mcv(np.array(li))
            self.buckets_keys.append(i)
            self.buckets.append(bi)
    def calc_from_data2(self, data_slice, scope: list, data_slice_fjbuckets_idx, this_table_domain: list, joined_tables_name: list, local_joined_downscale_factor_cols: list):
        #assert data_slice is not None and data_slice.shape[1] == 1
        assert data_slice is not None
        self.buckets = []
        self.buckets_keys = []
        key_bucket = {}
        key_bucket_dsfs = {}
        self.scope = tuple(scope)
        self.downscale_factor_th = {}
        for i_name, i in zip(joined_tables_name, local_joined_downscale_factor_cols):
            assert len(data_slice) == len(i)
            #key_bucket_dsfs.append({})
            self.downscale_factor_th[i_name] = len(self.downscale_factor_th)
        for dsi_th, dsi in enumerate(data_slice):
            #print(dsi, type(dsi), dsi.shape, scope)
            i = dsi[data_slice_fjbuckets_idx]
            assert i.shape[0] == len(scope)
            #print(i)
            # if data_slice.shape[1] == 1:
            #     hash_i = (hash_FJBuckets(i, this_table_domain[scope[0]][0], this_table_domain[scope[0]][1]), )
            # else:
            hash_i = tuple([hash_FJBuckets(j, this_table_domain[sc][0], this_table_domain[sc][1]) for j, sc in zip(i, scope)])
            if hash_i not in key_bucket:
                key_bucket[hash_i] = [i]
                key_bucket_dsfs[hash_i] = []
                for j in local_joined_downscale_factor_cols:
                    key_bucket_dsfs[hash_i].append([j[dsi_th]])
            else:
                key_bucket[hash_i].append(i)
                for jth, j in enumerate(local_joined_downscale_factor_cols):
                    key_bucket_dsfs[hash_i][jth].append(j[dsi_th])
        #key_bucket = list(key_bucket.items())
        key_bucket = sorted(list(key_bucket.items()), key=lambda t: t[0])
        for i_v in key_bucket_dsfs.values():
            for jth, j in enumerate(i_v):
                i_v[jth] = np.array(j)
        key_bucket_dsfs = sorted(list(key_bucket_dsfs.items()), key=lambda t: t[0])
        #print(key_bucket_dsfs)
        #print(len(key_bucket))
        #print(len(key_bucket_dsfs))
        #exit(-1)
        for (i, li), (dsfi_k, dsfi_v) in zip(key_bucket, key_bucket_dsfs):
            assert i == dsfi_k
            bi = Bucket()
            #print(i, np.array(li).shape)
            #exit(-1)
            bi.calc_n_mcv(np.array(li))
            assert len(self.downscale_factor_th) == len(dsfi_v)
            bi.calc_downscale_factor(dsfi_v)
            self.buckets_keys.append(i)
            self.buckets.append(bi)
        #exit(-1)
    def _print(self, no_zero=False, show_bs=False):
        print('FJBuckets scope =', self.scope)
        if show_bs:
            for bkeys, bs in zip(self.buckets_keys, self.buckets):
                if not no_zero or bs.n > 0:
                    print('bucket hash =', bkeys)
                    print('\tmcv={}\n\tmcv_freq={}\n\tdomain={}\n\tn={}'.format(bs.mcv, bs.mcv_freq, bs.domain, bs.n))
                    if bs.downscale_factor is not None:
                        print('\tdownscale_factor={}'.format(bs.downscale_factor))
        print('{} - {} buckets on scope:{}'.format(len(self.buckets_keys), len(self.buckets), self.scope))
        if self.downscale_factor_th is not None:
            print('joined bigtable: ', self.downscale_factor_th)

def product_merge_FJBuckets2(l_a, b, c: float):
    #print('l_a:', l_a)
    assert b is None
    if l_a is None:
        return None
    for a in l_a:
        for i in range(len(a.buckets)):
            #a.buckets[i].n = round(a.buckets[i].n * c)
            a.buckets[i].n = a.buckets[i].n * c
            #a.buckets[i].mcv_freq = max(1, round(a.buckets[i].mcv_freq * c))
            a.buckets[i].mcv_freq = a.buckets[i].mcv_freq * c
            if a.buckets[i].mcv_freq > a.buckets[i].n:
                a.buckets[i].mcv_freq = a.buckets[i].n
            #a.buckets[i].domain *= c
            #a.buckets[i].domain = min(a.buckets[i].domain, a.buckets[i].n)
    return l_a

def product_merge_FJBuckets(a, b, c: float):
    if a is None:
        assert b is None
        return None
    if b is None:
        for i in range(len(a.buckets)):
            #a.buckets[i].n = round(a.buckets[i].n * c)
            a.buckets[i].n = a.buckets[i].n * c
            #a.buckets[i].mcv_freq = max(1, round(a.buckets[i].mcv_freq * c))
            a.buckets[i].mcv_freq = a.buckets[i].mcv_freq * c
            if a.buckets[i].mcv_freq > a.buckets[i].n:
                a.buckets[i].mcv_freq = a.buckets[i].n
            #a.buckets[i].domain *= c
            #a.buckets[i].domain = min(a.buckets[i].domain, a.buckets[i].n)
        return a
    assert a.scope != b.scope
    ab = FJBuckets()
    ab.scope = a.scope + b.scope
    for i, bucket_i in zip(a.buckets_keys, a.buckets):
        for j, bucket_j in zip(b.buckets_keys, b.buckets):
            bucket_ij = Bucket()
            n = round(bucket_i.n * bucket_j.n * 2 / (bucket_i.n + bucket_j.n) * c)
            mcv = bucket_i.mcv + bucket_j.mcv
            mcv_freq = max(1, round(bucket_i.mcv_freq * bucket_j.mcv_freq * 2 / (bucket_i.mcv_freq + bucket_j.mcv_freq) * c))
            if mcv_freq > n:
                mcv_freq = n
            domain = min(n, bucket_i.domain * bucket_j.domain)
            bucket_ij.set_n_mcv_domain(n, mcv, mcv_freq, domain)
            ab.buckets_keys.append(i + j)
            ab.buckets.append(bucket_ij)
    return ab

def product_merge_FJBuckets_opt(l_a, b, c: float):
    #print('Prod', l_a, b, c)
    assert b is None
    if l_a is None:
        return None
    for a in l_a:
        for i in range(len(a.buckets)):
            #a.buckets[i].n = round(a.buckets[i].n * c)
            a.buckets[i].n = a.buckets[i].n * c
            #a.buckets[i].mcv_freq = max(1, round(a.buckets[i].mcv_freq * c))
            a.buckets[i].mcv_freq = a.buckets[i].mcv_freq * c
            if a.buckets[i].mcv_freq > a.buckets[i].n:
                a.buckets[i].mcv_freq = a.buckets[i].n
            #a.buckets[i].domain *= c
            #a.buckets[i].domain = min(a.buckets[i].domain, a.buckets[i].n)
    return l_a
    # assert a.scope != b.scope
    # ab = FJBuckets()
    # ab.scope = a.scope + b.scope
    # for i, bucket_i in zip(a.buckets_keys, a.buckets):
    #     for j, bucket_j in zip(b.buckets_keys, b.buckets):
    #         bucket_ij = Bucket()
    #         n = round(bucket_i.n * bucket_j.n * 2 / (bucket_i.n + bucket_j.n) * c)
    #         mcv = bucket_i.mcv + bucket_j.mcv
    #         mcv_freq = max(1, round(bucket_i.mcv_freq * bucket_j.mcv_freq * 2 / (bucket_i.mcv_freq + bucket_j.mcv_freq) * c))
    #         if mcv_freq > n:
    #             mcv_freq = n
    #         domain = min(n, bucket_i.domain * bucket_j.domain)
    #         bucket_ij.set_n_mcv_domain(n, mcv, mcv_freq, domain)
    #         ab.buckets_keys.append(i + j)
    #         ab.buckets.append(bucket_ij)
    # return ab


def sum_merge_FJBuckets2(children_l: list):
    #not copy all bucket_j, only gen [buckets, buckets, buckets, ...]
    sum_l = None
    for i in children_l:
        if i is not None:
            if sum_l is None:
                sum_l = i
            else:
                sum_l.extend(i)
    return sum_l

def sum_merge_FJBuckets(children: list):
    sum = None
    for i in children:
        if i is not None:
            if sum is None:
                #sum = FJBuckets()
                #sum.scope = i.scope
                sum = i
            assert sum.scope == i.scope
            for j, bucket_j in zip(i.buckets_keys, i.buckets):
                sum.buckets_keys.append(j)
                sum.buckets.append(bucket_j)
    return sum

def sum_merge_FJBuckets_opt(children_l: list):
    sum_l = None
    for i in children_l:
        if i is not None:
            if sum_l is None:
                sum_l = i
            else:
                sum_l.extend(i)
            #assert sum_l.scope == i.scope
            # for j, bucket_j in zip(i.buckets_keys, i.buckets):
            #     sum.buckets_keys.append(j)
            #     sum.buckets.append(bucket_j)
    return sum_l

def leaf_select_FJBuckets2(join_scope: list, node_fjbuckets, query_select_predicates, this_table_domain: list, joined_tables: set):
    if node_fjbuckets is None:
        return None
    node_fjbuckets_scope_index = []
    for i in join_scope:
        if i in node_fjbuckets.scope:
            node_fjbuckets_scope_index.append(node_fjbuckets.scope.index(i))
    if len(node_fjbuckets_scope_index) == 0:
        #print('Leaf Node: ', node_fjbuckets.scope, len(node_fjbuckets.buckets))
        #print('PBFS get from Leaf Node: ', None)
        return None
    assert len(node_fjbuckets_scope_index) == 1
    #downscale
    downscale_factor = []
    for i_k, i_v in node_fjbuckets.downscale_factor_th.items():
        if i_k in joined_tables:
            downscale_factor.append(i_v)
    if len(downscale_factor) == len(node_fjbuckets.downscale_factor_th):
        downscale_factor = None
    else:
        assert len(downscale_factor) == 1
        downscale_factor = downscale_factor[0]
    ret = FJBuckets()
    ret.scope = tuple(join_scope)
    for i, bi in zip(node_fjbuckets.buckets_keys, node_fjbuckets.buckets):
        is_in_query = True
        if query_select_predicates is not None:
            for j, sc in zip(i, node_fjbuckets.scope):
                l = hash_FJBuckets(query_select_predicates[0][0, sc], this_table_domain[sc][0], this_table_domain[sc][1])
                r = hash_FJBuckets(query_select_predicates[1][0, sc], this_table_domain[sc][0], this_table_domain[sc][1])
                #print('l={}, r={}, bucket[{}]'.format(l, r, j))
                if j < l or j > r:
                    is_in_query = False
                    break
        if is_in_query:
            ret_bi = Bucket()
            n = bi.n
            mcv = tuple([bi.mcv[j] for j in node_fjbuckets_scope_index])
            mcv_freq = bi.mcv_freq
            domain = bi.domain
            if downscale_factor is None:
                dsf_join_v = bi.downscale_factor_2col
                dsf_join_v = 1
                #print(dsf_join_v)
                ret_bi.set_n_mcv_domain(n * dsf_join_v, mcv, mcv_freq * dsf_join_v, domain)
            else:
                dsf_v = bi.downscale_factor[downscale_factor]
                dsf_v = 1
                #print(dsf_v)
                ret_bi.set_n_mcv_domain(n * dsf_v, mcv, mcv_freq * dsf_v, domain)
            ret_i = tuple([i[j] for j in node_fjbuckets_scope_index])
            ret.buckets_keys.append(ret_i)
            ret.buckets.append(ret_bi)
    #print('Leaf Node: ', node_fjbuckets.scope, len(node_fjbuckets.buckets))
    #print('PBFS get from Leaf Node: ', ret.scope, len(ret.buckets))
    #print('[ret]=',[ret])
    return [ret]

def leaf_select_FJBuckets(join_scope: list, node_fjbuckets, query_select_predicates, this_table_domain: list):
    if node_fjbuckets is None:
        return None
    node_fjbuckets_scope_index = []
    for i in join_scope:
        if i in node_fjbuckets.scope:
            node_fjbuckets_scope_index.append(node_fjbuckets.scope.index(i))
    if len(node_fjbuckets_scope_index) == 0:
        #print('Leaf Node: ', node_fjbuckets.scope, len(node_fjbuckets.buckets))
        #print('PBFS get from Leaf Node: ', None)
        return None
    ret = FJBuckets()
    ret.scope = tuple(join_scope)
    for i, bi in zip(node_fjbuckets.buckets_keys, node_fjbuckets.buckets):
        is_in_query = True
        if query_select_predicates is not None:
            for j, sc in zip(i, node_fjbuckets.scope):
                l = hash_FJBuckets(query_select_predicates[0][0, sc], this_table_domain[sc][0], this_table_domain[sc][1])
                r = hash_FJBuckets(query_select_predicates[1][0, sc], this_table_domain[sc][0], this_table_domain[sc][1])
                #print('l={}, r={}, bucket[{}]'.format(l, r, j))
                if j < l or j > r:
                    is_in_query = False
                    break
        if is_in_query:
            ret_bi = Bucket()
            n = bi.n
            mcv = tuple([bi.mcv[j] for j in node_fjbuckets_scope_index])
            mcv_freq = bi.mcv_freq
            domain = bi.domain
            ret_bi.set_n_mcv_domain(n, mcv, mcv_freq, domain)
            ret_i = tuple([i[j] for j in node_fjbuckets_scope_index])
            ret.buckets_keys.append(ret_i)
            ret.buckets.append(ret_bi)
    #print('Leaf Node: ', node_fjbuckets.scope, len(node_fjbuckets.buckets))
    #print('PBFS get from Leaf Node: ', ret.scope, len(ret.buckets))
    return ret

def leaf_select_FJBuckets_opt(join_scope: list, node_fjbuckets, query_select_predicates, this_table_domain: list):
    if node_fjbuckets is None:
        return None
    node_fjbuckets_scope_index = []
    for i in join_scope:
        if i in node_fjbuckets.scope:
            node_fjbuckets_scope_index.append(node_fjbuckets.scope.index(i))
    if len(node_fjbuckets_scope_index) == 0:
        #print('Leaf Node: ', node_fjbuckets.scope, len(node_fjbuckets.buckets))
        #print('PBFS get from Leaf Node: ', None)
        return None
    assert len(node_fjbuckets_scope_index) == 1
    ret = FJBuckets()
    ret.scope = tuple(join_scope)
    for i, bi in zip(node_fjbuckets.buckets_keys, node_fjbuckets.buckets):
        is_in_query = True
        if query_select_predicates is not None:
            for j, sc in zip(i, node_fjbuckets.scope):
                l = hash_FJBuckets(query_select_predicates[0][0, sc], this_table_domain[sc][0], this_table_domain[sc][1])
                r = hash_FJBuckets(query_select_predicates[1][0, sc], this_table_domain[sc][0], this_table_domain[sc][1])
                #print('l={}, r={}, bucket[{}]'.format(l, r, j))
                if j < l or j > r:
                    is_in_query = False
                    break
        if is_in_query:
            ret_bi = Bucket()
            n = bi.n
            mcv = tuple([bi.mcv[j] for j in node_fjbuckets_scope_index])
            mcv_freq = bi.mcv_freq
            domain = bi.domain
            ret_bi.set_n_mcv_domain(n, mcv, mcv_freq, domain)
            ret_i = tuple([i[j] for j in node_fjbuckets_scope_index])
            ret.buckets_keys.append(ret_i)
            ret.buckets.append(ret_bi)
    #print('Leaf Node: ', node_fjbuckets.scope, len(node_fjbuckets.buckets))
    #print('PBFS get from Leaf Node: ', ret.scope, len(ret.buckets))
    return [ret]

def ve2(domain_fjbuckets: FJBuckets, others: list, dynamic_factor_attr=(2.0, 0.0, 1.0, 200)):
    if ALGO == 'fj_dyn_factor':
        set_dynfactor_function(dynamic_factor_attr)
    #join_scope = domain_fjbuckets.scope
    # domain_fjbuckets._print()
    # for i in others:
    #     i._print()
    # #exit(-1)
    # others_buckets_keys_search = []
    # for i in others:
    #     others_buckets_keys_search.append({})
    #     for j, key in enumerate(i.buckets_keys):
    #         if key not in others_buckets_keys_search[-1]:
    #             others_buckets_keys_search[-1][key] = [j]
    #         else:
    #             others_buckets_keys_search[-1][key].append(j)
        # for j in others_buckets_keys_search[-1]:
        #     print(j, others_buckets_keys_search[-1][j])
    #    print()
    #exit(-1)
    # others_buckets_scope_mapping = []
    # for i in others:
    #     others_buckets_scope_mapping.append([])
    #     for j in i.scope:
    #         for k, sc in enumerate(join_scope):
    #             if sc == j:
    #                 others_buckets_scope_mapping[-1].append(k)
    #                 break
    #     others_buckets_scope_mapping[-1] = tuple(others_buckets_scope_mapping[-1])
    ret = 0
    ret_factor = 0
    fjbuckets_match = {}
    #print(len(domain_fjbuckets.buckets_keys), 'bins')
    others_pointers = [0] * len(others)
    #print(domain_fjbuckets, others)
    for i, bi in zip(domain_fjbuckets.buckets_keys, domain_fjbuckets.buckets):
        multiply1 = bi.mcv_freq
        multiply2 = bi.mcv_freq
        bi_other_freq = (bi.n - bi.mcv_freq) / bi.domain if bi.n > bi.mcv_freq else 0
        #print(multiply2)
        #ratio_up = np.int64(bi.mcv_freq)
        #ratio_down = np.int64(bi_other_freq)
        #ratio = ratio_up / ratio_down
        if ALGO == 'fj_dyn_factor':
            ratio = calc_dynfactor_ratio(bi.mcv_freq, bi_other_freq)
            factor = (1 - ratio) * 1.0 + ratio * bi.n / bi.mcv_freq
        elif ALGO == 'fj_ada_factor':
            if bi.mcv_freq >= dynamic_factor_attr[3] * bi_other_freq:
            #if bi.mcv_freq >= 0.9 * bi.n:
                factor = 1.0
            else:
                #ratio = bi.mcv_freq / bi_other_freq
                #ub = dynamic_factor_attr[1]
                #ratio /= ub
                #factor = ratio * 1.0 + (1 - ratio) * bi.n / bi.mcv_freq
                factor = bi.n / bi.mcv_freq
        else:
            factor = bi.n / bi.mcv_freq if bi.mcv_freq > 0 else 1
        bi_match = ['mcv']
        if SHOW_VE:
            print('dom', i, bi.mcv, bi.mcv_freq, bi.n - bi.mcv_freq, bi.domain)
        for jth, j in enumerate(others):
            key = i
            bi_mcv = bi.mcv
            #linear multi join
            k = others_pointers[jth]
            while k < len(j.buckets_keys) and j.buckets_keys[k] < key:
                others_pointers[jth] += 1
                k = others_pointers[jth]
            if k >= len(j.buckets_keys) or j.buckets_keys[k] > key:
                multiply1 *= 0
                multiply2 *= 0
                break
            #print('dom->others', key, bi_mcv)
            #print('others')
            #exit(-1)
            #factor.append([])
            #factorj = []
            k0 = k
            factorj_n = 0
            factorj_mcv_freq = 0
            multiply1_j = 0
            multiply2_j = 0
            while k < len(j.buckets_keys) and j.buckets_keys[k] == key:
                jk_mcv = j.buckets[k].mcv
                #print(jth+1, k, 'key:', (key, j.buckets_keys[k]), 'mcv:', (bi_mcv, jk_mcv))
                jk_mcv_freq = j.buckets[k].mcv_freq
                jk_n = j.buckets[k].n
                jk_dom = j.buckets[k].domain
                jk_other_freq = (jk_n - jk_mcv_freq) / jk_dom if jk_n > jk_mcv_freq else 0
                #print(jk_mcv_freq, jk_other_freq)
                if SHOW_VE:
                    print('\t', j.buckets_keys[k], jk_mcv, jk_mcv_freq, jk_n - jk_mcv_freq, jk_dom)
                if ALGO == 'fj_dyn_factor':
                    ratio = calc_dynfactor_ratio(jk_mcv_freq, jk_other_freq)
                    factorj_mcv_freq += (1 - ratio) * jk_n + ratio * jk_mcv_freq
                    factorj_n += jk_n
                elif ALGO == 'fj_ada_factor':
                    if jk_mcv_freq >= dynamic_factor_attr[3] * jk_other_freq:
                    #if jk_mcv_freq >= 0.9 * jk_n:
                        factorj_mcv_freq += jk_n
                        factorj_n += jk_n
                    else:
                        #ratio = jk_mcv_freq / jk_other_freq
                        #ub = dynamic_factor_attr[1]
                        #ratio /= ub
                        #factorj_mcv_freq += ratio * jk_n + (1 - ratio) * jk_mcv_freq
                        #factorj_n += jk_n
                        factorj_mcv_freq += jk_mcv_freq
                        factorj_n += jk_n
                else:
                    factorj_mcv_freq += jk_mcv_freq
                    factorj_n += jk_n
                # if jk_mcv not in factorj_mcv_freq:
                #     factorj_mcv_freq[jk_mcv] = jk_mcv_freq
                # else:
                #     factorj_mcv_freq[jk_mcv] += jk_mcv_freq
                #exit()
                if bi_mcv == jk_mcv:
                    #print('yes!')
                    #print('\tmulti1*', jk_mcv_freq)
                    bi_match.append('same')
                    multiply1_j += jk_mcv_freq
                    #if jk_other_freq < jk_mcv_freq:
                    multiply2_j += jk_mcv_freq
                    #multiply2_j += jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom)
                    #multiply1 *= jk_mcv_freq
                    #factorj.append(jk_n / max(1, jk_mcv_freq))
                    #multiply2 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                    #print('\tmulti2*', max(1, round((jk_n - jk_mcv_freq) / max(1, (jk_dom - 1)))))
                    #multiply2 *= max(1, round((jk_n - jk_mcv_freq) / max(1, (jk_dom - 1))))
                    #factor[-1].append(1.0)
                else:
                    bi_match.append('NOTsame')
                    #multiply1_j += 0
                    #multiply1 *= 0
                    #multiply1_j += jk_mcv_freq                  
                    # if jk_n > jk_mcv_freq:
                    #     other_value_avg_freq = (jk_n - jk_mcv_freq) / jk_dom
                        #multiply1_j += jk_mcv_freq
                        #print((jk_n - jk_mcv_freq) / jk_dom)
                    #if jk_other_freq < jk_mcv_freq:
                    if ALGO in ['factorjoin', 'mul1', 'mixed_fj_mul1', 'ada_fj', 'ada_fj_mul1+mul2', 'fj_ada_factor', 'fj_dyn_factor']:
                        multiply1_j += jk_mcv_freq
                    elif ALGO in ['mul1_nomcv', 'mul1+mul2']:
                        multiply1_j += jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom)
                    else:
                        assert 'Unknown Methods of VE' == False
                    #multiply2_j += min(jk_mcv_freq, jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom))
                    multiply2_j += jk_other_freq
                    #multiply1_j += jk_mcv_freq
                    #multiply1_j += (jk_n - jk_mcv_freq) / jk_dom
                    #multiply1 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                    #factorj.append(jk_n / max(1, jk_mcv_freq))
                    #multiply2 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                    #print('\tmulti1*', round(jk_n / max(1, jk_dom)))
                    #multiply2 *= round(jk_n / max(1, jk_dom))
                #mcv_freq_ratio = jk_mcv_freq / max(1, jk_n)
                #factor[-1].append(mcv_freq_ratio * 1 + (1-mcv_freq_ratio) * jk_n / max(1, jk_dom))
                    #print('\tmulti2*', round(jk_n / max(1, jk_dom)))
                others_pointers[jth] += 1
                k = others_pointers[jth]
            others_pointers[jth] = k0
            #mean_factorj = float(np.mean(factorj)) if len(factorj) > 0 else 1.0
            #factor = max(factor, mean_factorj)
            multiply1 *= multiply1_j
            multiply2 *= multiply2_j
            if SHOW_VE:
                print('\tmultiply1_j =', multiply1_j, 'multiply2_j =', multiply2_j)
            #print('factorj_n=',factorj_n,'factorj_mcv_freq',factorj_mcv_freq)
            factor = min(factor, factorj_n / factorj_mcv_freq if factorj_mcv_freq > 0 else 1)
        # else:
        #     multiply1 *= 0
        #     multiply2 *= 0
        #     break
            #print('multi1=', multiply1, 'multi2=', multiply2)
            #factor[-1] = float(np.mean(factor[-1])) if len(factor[-1]) > 0 else 0.0
        #exit()
        #print(multiply1, factor)
        #print(multiply1)
        #ret = multiply1 + multiply2
        #ret += multiply1 * factor
        bi_match = '_'.join(bi_match)
        if bi_match not in fjbuckets_match:
            fjbuckets_match[bi_match] = 1
        else:
            fjbuckets_match[bi_match] += 1
        if ALGO in ['mul1', 'mul1_nomcv']:
            if SHOW_VE:
                print('multiply1 =', multiply1)
            ret += multiply1
        elif ALGO == 'factorjoin':
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
            ret += multiply1 * factor
        elif ALGO == 'mul1+mul2':
            if SHOW_VE:
                print('multiply1 =', multiply1, 'multiply2 =', multiply2)
            ret += multiply1 + multiply2
        elif ALGO == 'mixed_fj_mul1':
            ret += multiply1
            ret_factor += multiply1 * factor
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
        elif ALGO == 'ada_fj':
            if multiply1 < factor:
                ret += multiply1 * factor
            else:
                ret += multiply1
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
        elif ALGO == 'ada_fj_mul1+mul2':
            if multiply1 < factor:
                ret += multiply1 * factor
            elif multiply2 < factor:
                ret += multiply2 * factor
            else:
                ret += multiply2
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor, 'multiply2 =', multiply2)
        elif ALGO in ['fj_ada_factor', 'fj_dyn_factor']:
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
            ret += multiply1 * factor
        else:
            assert 'Unknown Methods of VE' == False
        #ret += multiply1
        #ret += multiply1 + multiply2
        #ret += multiply1 * factor
        #print(factor)
        #ret += round(multiply1)
        #ret += round(multiply1 * float(np.prod(factor)))
    if ALGO == 'mixed_fj_mul1':
        if SHOW_VE:
            print('dual_res:', 'ret =', ret, 'ret_factor =', ret_factor)
        ret = ret_factor if ret < 1000 or len(others) <= 1 else ret
    if SHOW_VE:
        for i, ii in fjbuckets_match.items():
            print(i, ii)
        print(ret)
    #exit(-1)
    return ret
    #return max(ret, 1)

def ve(domain_fjbuckets: FJBuckets, others: list, others_buckets_scope_mapping: list, dynamic_factor_attr=(2.0, 0.0, 1.0, 18)):
    if ALGO == 'fj_dyn_factor':
        set_dynfactor_function(dynamic_factor_attr)
    join_scope = domain_fjbuckets.scope
    # domain_fjbuckets._print()
    # for i in others:
    #     i._print()
    #exit(-1)
    others_buckets_keys_search = []
    for i in others:
        others_buckets_keys_search.append({})
        for j, key in enumerate(i.buckets_keys):
            if key not in others_buckets_keys_search[-1]:
                others_buckets_keys_search[-1][key] = [j]
            else:
                others_buckets_keys_search[-1][key].append(j)
        # for j in others_buckets_keys_search[-1]:
        #     print(j, others_buckets_keys_search[-1][j])
    #    print()
    #exit(-1)
    # others_buckets_scope_mapping = []
    # for i in others:
    #     others_buckets_scope_mapping.append([])
    #     for j in i.scope:
    #         for k, sc in enumerate(join_scope):
    #             if sc == j:
    #                 others_buckets_scope_mapping[-1].append(k)
    #                 break
    #     others_buckets_scope_mapping[-1] = tuple(others_buckets_scope_mapping[-1])
    ret = 0
    ret_factor = 0
    fjbuckets_match = {}
    #print(len(domain_fjbuckets.buckets_keys), 'bins')
    for i, bi in zip(domain_fjbuckets.buckets_keys, domain_fjbuckets.buckets):
        multiply1 = bi.mcv_freq
        multiply2 = bi.mcv_freq
        bi_other_freq = (bi.n - bi.mcv_freq) / bi.domain if bi.n > bi.mcv_freq else 0
        #print(multiply2)
        #ratio_up = np.int64(bi.mcv_freq)
        #ratio_down = np.int64(bi_other_freq)
        #ratio = ratio_up / ratio_down
        if ALGO == 'fj_dyn_factor':
            ratio = calc_dynfactor_ratio(bi.mcv_freq, bi_other_freq)
            factor = (1 - ratio) * 1.0 + ratio * bi.n / bi.mcv_freq
        elif ALGO == 'fj_ada_factor':
            if bi.mcv_freq >= dynamic_factor_attr[3] * bi_other_freq:
            #if bi.mcv_freq >= 0.9 * bi.n:
                factor = 1.0
            else:
                #ratio = bi.mcv_freq / bi_other_freq
                #ub = dynamic_factor_attr[1]
                #ratio /= ub
                #factor = ratio * 1.0 + (1 - ratio) * bi.n / bi.mcv_freq
                factor = bi.n / bi.mcv_freq
        else:
            factor = bi.n / bi.mcv_freq
        bi_match = ['mcv']
        if SHOW_VE:
            print('dom', i, bi.mcv, bi.mcv_freq, bi.n - bi.mcv_freq, bi.domain)
        for j, jm, jkeys in zip(others, others_buckets_scope_mapping, others_buckets_keys_search):
            key = tuple([i[k] for k in jm])
            bi_mcv = tuple([bi.mcv[k] for k in jm])
            #print('dom->others', key, bi_mcv)
            #print('others')
            #exit(-1)
            #factor.append([])
            #factorj = []
            factorj_n = 0
            factorj_mcv_freq = 0
            multiply1_j = 0
            multiply2_j = 0
            if key in jkeys:
                for k in jkeys[key]:
                    jk_mcv = j.buckets[k].mcv
                    jk_mcv_freq = j.buckets[k].mcv_freq
                    jk_n = j.buckets[k].n
                    jk_dom = j.buckets[k].domain
                    jk_other_freq = (jk_n - jk_mcv_freq) / jk_dom if jk_n > jk_mcv_freq else 0
                    #print(jk_mcv_freq, jk_other_freq)
                    if SHOW_VE:
                        print('\t', j.buckets_keys[k], jk_mcv, jk_mcv_freq, jk_n - jk_mcv_freq, jk_dom)
                    if ALGO == 'fj_dyn_factor':
                        ratio = calc_dynfactor_ratio(jk_mcv_freq, jk_other_freq)
                        factorj_mcv_freq += (1 - ratio) * jk_n + ratio * jk_mcv_freq
                        factorj_n += jk_n
                    elif ALGO == 'fj_ada_factor':
                        if jk_mcv_freq >= dynamic_factor_attr[3] * jk_other_freq:
                        #if jk_mcv_freq >= 0.9 * jk_n:
                            factorj_mcv_freq += jk_n
                            factorj_n += jk_n
                        else:
                            #ratio = jk_mcv_freq / jk_other_freq
                            #ub = dynamic_factor_attr[1]
                            #ratio /= ub
                            #factorj_mcv_freq += ratio * jk_n + (1 - ratio) * jk_mcv_freq
                            #factorj_n += jk_n
                            factorj_mcv_freq += jk_mcv_freq
                            factorj_n += jk_n
                    else:
                        factorj_mcv_freq += jk_mcv_freq
                        factorj_n += jk_n
                    # if jk_mcv not in factorj_mcv_freq:
                    #     factorj_mcv_freq[jk_mcv] = jk_mcv_freq
                    # else:
                    #     factorj_mcv_freq[jk_mcv] += jk_mcv_freq
                    #exit()
                    if bi_mcv == jk_mcv:
                        #print('yes!')
                        #print('\tmulti1*', jk_mcv_freq)
                        bi_match.append('same')
                        multiply1_j += jk_mcv_freq
                        #if jk_other_freq < jk_mcv_freq:
                        multiply2_j += jk_mcv_freq
                        #multiply2_j += jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom)
                        #multiply1 *= jk_mcv_freq
                        #factorj.append(jk_n / max(1, jk_mcv_freq))
                        #multiply2 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                        #print('\tmulti2*', max(1, round((jk_n - jk_mcv_freq) / max(1, (jk_dom - 1)))))
                        #multiply2 *= max(1, round((jk_n - jk_mcv_freq) / max(1, (jk_dom - 1))))
                        #factor[-1].append(1.0)
                    else:
                        bi_match.append('NOTsame')
                        #multiply1_j += 0
                        #multiply1 *= 0
                        #multiply1_j += jk_mcv_freq                  
                        # if jk_n > jk_mcv_freq:
                        #     other_value_avg_freq = (jk_n - jk_mcv_freq) / jk_dom
                            #multiply1_j += jk_mcv_freq
                            #print((jk_n - jk_mcv_freq) / jk_dom)
                        #if jk_other_freq < jk_mcv_freq:
                        if ALGO in ['factorjoin', 'mul1', 'mixed_fj_mul1', 'ada_fj', 'ada_fj_mul1+mul2', 'fj_ada_factor', 'fj_dyn_factor']:
                            multiply1_j += jk_mcv_freq
                        elif ALGO in ['mul1_nomcv', 'mul1+mul2']:
                            multiply1_j += jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom)
                        else:
                            assert 'Unknown Methods of VE' == False
                        #multiply2_j += min(jk_mcv_freq, jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom))
                        multiply2_j += jk_other_freq
                        #multiply1_j += jk_mcv_freq
                        #multiply1_j += (jk_n - jk_mcv_freq) / jk_dom
                        #multiply1 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                        #factorj.append(jk_n / max(1, jk_mcv_freq))
                        #multiply2 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                        #print('\tmulti1*', round(jk_n / max(1, jk_dom)))
                        #multiply2 *= round(jk_n / max(1, jk_dom))
                    #mcv_freq_ratio = jk_mcv_freq / max(1, jk_n)
                    #factor[-1].append(mcv_freq_ratio * 1 + (1-mcv_freq_ratio) * jk_n / max(1, jk_dom))
                        #print('\tmulti2*', round(jk_n / max(1, jk_dom)))
                #mean_factorj = float(np.mean(factorj)) if len(factorj) > 0 else 1.0
                #factor = max(factor, mean_factorj)
                multiply1 *= multiply1_j
                multiply2 *= multiply2_j
                if SHOW_VE:
                    print('\tmultiply1_j =', multiply1_j, 'multiply2_j =', multiply2_j)
                factor = min(factor, factorj_n / factorj_mcv_freq)
            else:
                multiply1 *= 0
                multiply2 *= 0
                break
            #print('multi1=', multiply1, 'multi2=', multiply2)
            #factor[-1] = float(np.mean(factor[-1])) if len(factor[-1]) > 0 else 0.0
        #exit()
        #print(multiply1, factor)
        #print(multiply1)
        #ret = multiply1 + multiply2
        #ret += multiply1 * factor
        bi_match = '_'.join(bi_match)
        if bi_match not in fjbuckets_match:
            fjbuckets_match[bi_match] = 1
        else:
            fjbuckets_match[bi_match] += 1
        if ALGO in ['mul1', 'mul1_nomcv']:
            if SHOW_VE:
                print('multiply1 =', multiply1)
            ret += multiply1
        elif ALGO == 'factorjoin':
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
            ret += multiply1 * factor
        elif ALGO == 'mul1+mul2':
            if SHOW_VE:
                print('multiply1 =', multiply1, 'multiply2 =', multiply2)
            ret += multiply1 + multiply2
        elif ALGO == 'mixed_fj_mul1':
            ret += multiply1
            ret_factor += multiply1 * factor
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
        elif ALGO == 'ada_fj':
            if multiply1 < factor:
                ret += multiply1 * factor
            else:
                ret += multiply1
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
        elif ALGO == 'ada_fj_mul1+mul2':
            if multiply1 < factor:
                ret += multiply1 * factor
            elif multiply2 < factor:
                ret += multiply2 * factor
            else:
                ret += multiply2
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor, 'multiply2 =', multiply2)
        elif ALGO in ['fj_ada_factor', 'fj_dyn_factor']:
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
            ret += multiply1 * factor
        else:
            assert 'Unknown Methods of VE' == False
        #ret += multiply1
        #ret += multiply1 + multiply2
        #ret += multiply1 * factor
        #print(factor)
        #ret += round(multiply1)
        #ret += round(multiply1 * float(np.prod(factor)))
    if ALGO == 'mixed_fj_mul1':
        if SHOW_VE:
            print('dual_res:', 'ret =', ret, 'ret_factor =', ret_factor)
        ret = ret_factor if ret < 1000 or len(others) <= 1 else ret
    if SHOW_VE:
        for i, ii in fjbuckets_match.items():
            print(i, ii)
        print(ret)
    #exit(-1)
    return ret
    #return max(ret, 1)

def ve_opt(domain_fjbuckets: FJBuckets, others: list, dynamic_factor_attr=(2.0, 0.0, 1.0, 25)):
    global FJBuckets_K
    #print('ve_opt: {}, FJBuckets_K={}'.format(dynamic_factor_attr, FJBuckets_K))
    if ALGO == 'fj_dyn_factor':
        set_dynfactor_function(dynamic_factor_attr)
    #join_scope = domain_fjbuckets.scope
    # domain_fjbuckets._print()
    # for i in others:
    #     i._print()
    # #exit(-1)
    # others_buckets_keys_search = []
    # for i in others:
    #     others_buckets_keys_search.append({})
    #     for j, key in enumerate(i.buckets_keys):
    #         if key not in others_buckets_keys_search[-1]:
    #             others_buckets_keys_search[-1][key] = [j]
    #         else:
    #             others_buckets_keys_search[-1][key].append(j)
        # for j in others_buckets_keys_search[-1]:
        #     print(j, others_buckets_keys_search[-1][j])
    #    print()
    #exit(-1)
    # others_buckets_scope_mapping = []
    # for i in others:
    #     others_buckets_scope_mapping.append([])
    #     for j in i.scope:
    #         for k, sc in enumerate(join_scope):
    #             if sc == j:
    #                 others_buckets_scope_mapping[-1].append(k)
    #                 break
    #     others_buckets_scope_mapping[-1] = tuple(others_buckets_scope_mapping[-1])
    ret = 0
    ret_factor = 0
    fjbuckets_match = {}
    #print(len(domain_fjbuckets.buckets_keys), 'bins')
    others_pointers = [0] * len(others)
    #print(domain_fjbuckets, others)
    for i, bi in zip(domain_fjbuckets.buckets_keys, domain_fjbuckets.buckets):
        multiply1 = bi.mcv_freq
        multiply2 = bi.mcv_freq
        bi_other_freq = (bi.n - bi.mcv_freq) / bi.domain if bi.n > bi.mcv_freq else 0
        #print(multiply2)
        #ratio_up = np.int64(bi.mcv_freq)
        #ratio_down = np.int64(bi_other_freq)
        #ratio = ratio_up / ratio_down
        if ALGO == 'fj_dyn_factor':
            ratio = calc_dynfactor_ratio(bi.mcv_freq, bi_other_freq)
            factor = (1 - ratio) * 1.0 + ratio * bi.n / bi.mcv_freq
        elif ALGO == 'fj_ada_factor':
            if bi.mcv_freq >= dynamic_factor_attr[3] * bi_other_freq:
            #if bi.mcv_freq >= 0.9 * bi.n:
                factor = 1.0
            else:
                #ratio = bi.mcv_freq / bi_other_freq
                #ub = dynamic_factor_attr[1]
                #ratio /= ub
                #factor = ratio * 1.0 + (1 - ratio) * bi.n / bi.mcv_freq
                factor = bi.n / bi.mcv_freq
        else:
            factor = bi.n / bi.mcv_freq if bi.mcv_freq > 0 else 1
        bi_match = ['mcv']
        if SHOW_VE:
            print('dom', i, bi.mcv, bi.mcv_freq, bi.n - bi.mcv_freq, bi.domain)
        for jth, j in enumerate(others):
            key = i
            bi_mcv = bi.mcv
            #linear multi join
            k = others_pointers[jth]
            while k < len(j.buckets_keys) and j.buckets_keys[k] < key:
                others_pointers[jth] += 1
                k = others_pointers[jth]
            if k >= len(j.buckets_keys) or j.buckets_keys[k] > key:
                multiply1 *= 0
                multiply2 *= 0
                break
            #print('dom->others', key, bi_mcv)
            #print('others')
            #exit(-1)
            #factor.append([])
            #factorj = []
            k0 = k
            factorj_n = 0
            factorj_mcv_freq = 0
            multiply1_j = 0
            multiply2_j = 0
            while k < len(j.buckets_keys) and j.buckets_keys[k] == key:
                jk_mcv = j.buckets[k].mcv
                #print(jth+1, k, 'key:', (key, j.buckets_keys[k]), 'mcv:', (bi_mcv, jk_mcv))
                jk_mcv_freq = j.buckets[k].mcv_freq
                jk_n = j.buckets[k].n
                jk_dom = j.buckets[k].domain
                jk_other_freq = (jk_n - jk_mcv_freq) / jk_dom if jk_n > jk_mcv_freq else 0
                #print(jk_mcv_freq, jk_other_freq)
                if SHOW_VE:
                    print('\t', j.buckets_keys[k], jk_mcv, jk_mcv_freq, jk_n - jk_mcv_freq, jk_dom)
                if ALGO == 'fj_dyn_factor':
                    ratio = calc_dynfactor_ratio(jk_mcv_freq, jk_other_freq)
                    factorj_mcv_freq += (1 - ratio) * jk_n + ratio * jk_mcv_freq
                    factorj_n += jk_n
                elif ALGO == 'fj_ada_factor':
                    if jk_mcv_freq >= dynamic_factor_attr[3] * jk_other_freq:
                    #if jk_mcv_freq >= 0.9 * jk_n:
                        factorj_mcv_freq += jk_n
                        factorj_n += jk_n
                    else:
                        #ratio = jk_mcv_freq / jk_other_freq
                        #ub = dynamic_factor_attr[1]
                        #ratio /= ub
                        #factorj_mcv_freq += ratio * jk_n + (1 - ratio) * jk_mcv_freq
                        #factorj_n += jk_n
                        factorj_mcv_freq += jk_mcv_freq
                        factorj_n += jk_n
                else:
                    factorj_mcv_freq += jk_mcv_freq
                    factorj_n += jk_n
                # if jk_mcv not in factorj_mcv_freq:
                #     factorj_mcv_freq[jk_mcv] = jk_mcv_freq
                # else:
                #     factorj_mcv_freq[jk_mcv] += jk_mcv_freq
                #exit()
                if bi_mcv == jk_mcv:
                    #print('yes!')
                    #print('\tmulti1*', jk_mcv_freq)
                    bi_match.append('same')
                    multiply1_j += jk_mcv_freq
                    #if jk_other_freq < jk_mcv_freq:
                    multiply2_j += jk_mcv_freq
                    #multiply2_j += jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom)
                    #multiply1 *= jk_mcv_freq
                    #factorj.append(jk_n / max(1, jk_mcv_freq))
                    #multiply2 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                    #print('\tmulti2*', max(1, round((jk_n - jk_mcv_freq) / max(1, (jk_dom - 1)))))
                    #multiply2 *= max(1, round((jk_n - jk_mcv_freq) / max(1, (jk_dom - 1))))
                    #factor[-1].append(1.0)
                else:
                    bi_match.append('NOTsame')
                    #multiply1_j += 0
                    #multiply1 *= 0
                    #multiply1_j += jk_mcv_freq                  
                    # if jk_n > jk_mcv_freq:
                    #     other_value_avg_freq = (jk_n - jk_mcv_freq) / jk_dom
                        #multiply1_j += jk_mcv_freq
                        #print((jk_n - jk_mcv_freq) / jk_dom)
                    #if jk_other_freq < jk_mcv_freq:
                    if ALGO in ['factorjoin', 'mul1', 'mixed_fj_mul1', 'ada_fj', 'ada_fj_mul1+mul2', 'fj_ada_factor', 'fj_dyn_factor']:
                        multiply1_j += jk_mcv_freq
                    elif ALGO in ['mul1_nomcv', 'mul1+mul2']:
                        multiply1_j += jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom)
                    else:
                        assert 'Unknown Methods of VE' == False
                    #multiply2_j += min(jk_mcv_freq, jk_other_freq * min(bi.domain, jk_dom) / max(bi.domain, jk_dom))
                    multiply2_j += jk_other_freq
                    #multiply1_j += jk_mcv_freq
                    #multiply1_j += (jk_n - jk_mcv_freq) / jk_dom
                    #multiply1 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                    #factorj.append(jk_n / max(1, jk_mcv_freq))
                    #multiply2 *= min(1, (jk_n - jk_mcv_freq) / max(1, jk_dom - 1))
                    #print('\tmulti1*', round(jk_n / max(1, jk_dom)))
                    #multiply2 *= round(jk_n / max(1, jk_dom))
                #mcv_freq_ratio = jk_mcv_freq / max(1, jk_n)
                #factor[-1].append(mcv_freq_ratio * 1 + (1-mcv_freq_ratio) * jk_n / max(1, jk_dom))
                    #print('\tmulti2*', round(jk_n / max(1, jk_dom)))
                others_pointers[jth] += 1
                k = others_pointers[jth]
            others_pointers[jth] = k0
            #mean_factorj = float(np.mean(factorj)) if len(factorj) > 0 else 1.0
            #factor = max(factor, mean_factorj)
            multiply1 *= multiply1_j
            multiply2 *= multiply2_j
            if SHOW_VE:
                print('\tmultiply1_j =', multiply1_j, 'multiply2_j =', multiply2_j)
            #print('factorj_n=',factorj_n,'factorj_mcv_freq',factorj_mcv_freq)
            factor = min(factor, factorj_n / factorj_mcv_freq if factorj_mcv_freq > 0 else 1)
        # else:
        #     multiply1 *= 0
        #     multiply2 *= 0
        #     break
            #print('multi1=', multiply1, 'multi2=', multiply2)
            #factor[-1] = float(np.mean(factor[-1])) if len(factor[-1]) > 0 else 0.0
        #exit()
        #print(multiply1, factor)
        #print(multiply1)
        #ret = multiply1 + multiply2
        #ret += multiply1 * factor
        bi_match = '_'.join(bi_match)
        if bi_match not in fjbuckets_match:
            fjbuckets_match[bi_match] = 1
        else:
            fjbuckets_match[bi_match] += 1
        if ALGO in ['mul1', 'mul1_nomcv']:
            if SHOW_VE:
                print('multiply1 =', multiply1)
            ret += multiply1
        elif ALGO == 'factorjoin':
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
            ret += multiply1 * factor
        elif ALGO == 'mul1+mul2':
            if SHOW_VE:
                print('multiply1 =', multiply1, 'multiply2 =', multiply2)
            ret += multiply1 + multiply2
        elif ALGO == 'mixed_fj_mul1':
            ret += multiply1
            ret_factor += multiply1 * factor
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
        elif ALGO == 'ada_fj':
            if multiply1 < factor:
                ret += multiply1 * factor
            else:
                ret += multiply1
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
        elif ALGO == 'ada_fj_mul1+mul2':
            if multiply1 < factor:
                ret += multiply1 * factor
            elif multiply2 < factor:
                ret += multiply2 * factor
            else:
                ret += multiply2
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor, 'multiply2 =', multiply2)
        elif ALGO in ['fj_ada_factor', 'fj_dyn_factor']:
            if SHOW_VE:
                print('multiply1 =', multiply1, 'factor =', factor)
            ret += multiply1 * factor
        else:
            assert 'Unknown Methods of VE' == False
        #ret += multiply1
        #ret += multiply1 + multiply2
        #ret += multiply1 * factor
        #print(factor)
        #ret += round(multiply1)
        #ret += round(multiply1 * float(np.prod(factor)))
    if ALGO == 'mixed_fj_mul1':
        if SHOW_VE:
            print('dual_res:', 'ret =', ret, 'ret_factor =', ret_factor)
        ret = ret_factor if ret < 1000 or len(others) <= 1 else ret
    if SHOW_VE:
        for i, ii in fjbuckets_match.items():
            print(i, ii)
        print(ret)
    #exit(-1)
    return ret
    #return max(ret, 1)

# max_dfs_columns = None
# min_dfs_card = None
# def dfs_domain_fjbuckets(tables_fjbuckets, join_parameters_cover ,join_parameters_nodes)
#     pass

def final_merge_sort_FJBuckets(fjbuckets_l):
    #merge list_buckets: [fjbuckets, fjbuckets, fjbuckets, ...] -> fjbuckets
    #linear mergesort
    #no build buckets_scope_mapping
    if fjbuckets_l is None:
        return None
    res = FJBuckets()
    #print('final:', fjbuckets_l)
    res.scope = fjbuckets_l[0].scope
    heap = []
    for ith, i in enumerate(fjbuckets_l):
        assert i.scope == res.scope
        if len(i.buckets) > 0:
            heapq.heappush(heap, (i.buckets_keys[0], ith, 0))
    while heap:
        _, ith, b_th = heapq.heappop(heap)
        res.buckets_keys.append(fjbuckets_l[ith].buckets_keys[b_th])
        res.buckets.append(fjbuckets_l[ith].buckets[b_th])
        if b_th < len(fjbuckets_l[ith].buckets) - 1:
            heapq.heappush(heap, (fjbuckets_l[ith].buckets_keys[b_th+1], ith, b_th+1))
    return res

# def try_final_merge_sort_FJBuckets():
#     fjb1 = FJBuckets()
#     fjb2 = FJBuckets()
#     fjb3 = FJBuckets()
#     fjb1.scope = (1,)
#     fjb2.scope = (1,)
#     fjb3.scope = (1,)
#     fjb1.buckets_keys = [2, 2, 4]
#     fjb2.buckets_keys = [1, 2]
#     fjb3.buckets_keys = [3, 5]
#     fjb1.buckets = ['11', '12', '13']
#     fjb2.buckets = ['21', '22']
#     fjb3.buckets = ['31', '32']
#     fjb_l = [fjb1, fjb2, fjb3]
#     fjb = final_merge_sort_FJBuckets(fjb_l)
#     print(fjb.scope)
#     for i_k, i_v in zip(fjb.buckets_keys, fjb.buckets):
#         print(i_k, i_v)

def calc_domain_fjbuckets2(tables_fjbuckets: dict, joined_tables_d: dict):
    score = {i_k: (-len(i_v), len(tables_fjbuckets[i_k].buckets)) for i_k, i_v in joined_tables_d.items()}
    domain = None
    domain_score = None
    for i_k, i_v in score.items():
        if domain_score is None or i_v < domain_score:
            domain = i_k
            domain_score = i_v
    assert domain is not None
    others = []
    for i_k, i_v in tables_fjbuckets.items():
        if i_k != domain:
            others.append(i_v)
    assert len(others) + 1 == len(tables_fjbuckets)
    domain = tables_fjbuckets[domain]
    return domain, others

def calc_domain_fjbuckets(tables_fjbuckets: dict, this_group: list, join_parameters_nodes: dict, mqspn: MultiQSPN):
    #sort by: (1) cover join_parameters number (this_group), (2) size of fjbuckets.buckets
    tfj_th = [i for i in range(len(tables_fjbuckets))]
    tables_name = list(tables_fjbuckets.keys())
    tables_fjbuckets_values = list(tables_fjbuckets.values())
    tfj_card = [len(i.buckets) for i in tables_fjbuckets_values]
    tfj_join_parameters_cover = []
    tfj_join_scope = []
    for i in tables_fjbuckets:
        tfj_join_parameters_cover.append([])
        tfj_join_scope.append({})
        for j in tables_fjbuckets[i].scope:
            for k in this_group:
                for l in join_parameters_nodes[k]:
                    lt, lc = l.split('.')
                    if lt == i:
                        lcs = mqspn.table_columns[lt][lc]
                        if lcs == j:
                            tfj_join_parameters_cover[-1].append(k)
                            tfj_join_scope[-1][k] = j
                            break
    # for i in tfj_th:
    #     print(tables_name[i])
    #     print(tfj_join_parameters_cover[i])
    #     print(tfj_card[i])
    #exit(-1)
    tfj_th = sorted(tfj_th, key=lambda t: (-len(tfj_join_parameters_cover[t]), tfj_card[t]))
    #print('choose domain_fjbuckets tables order:', tfj_th)
    #exit(-1)
    #select tables_fjbuckets in greedy way to cover all join_parameters number (this_group)
    #merge selected fjbuckets like product node
    domain_fjbuckets = None
    domain_tables = set()
    domain_columns = {}
    parameters_need_covering = set(this_group)
    others_th = []
    others = []
    others_buckets_scope_mapping = []
    for i in tfj_th:
        if domain_fjbuckets is None:
            selected_i_scope = []
            for j in tfj_join_parameters_cover[i]:
                selected_i_scope.append(tfj_join_scope[i][j])
                parameters_need_covering.remove(j)
                domain_columns[j] = len(domain_columns)
            #print('domain is', tables_name[i], tfj_card[i], 'buckets')
            domain_fjbuckets = leaf_select_FJBuckets(selected_i_scope, tables_fjbuckets_values[i], None, mqspn.table_domain[tables_name[i]])
        else:
            #get scope and product
            selected_i_scope = []
            selected_parameters = []
            for j in tfj_join_parameters_cover[i]:
                if j in parameters_need_covering:
                    selected_parameters.append(j)
                    selected_i_scope.append(tfj_join_scope[i][j])
            if len(selected_parameters) == 0:
                others_th.append(i)
                continue
            selected_i = leaf_select_FJBuckets(selected_i_scope, tables_fjbuckets_values[i], None, mqspn.table_domain[tables_name[i]])
            domain_fjbuckets = product_merge_FJBuckets(domain_fjbuckets, selected_i, 1.0)
            for j in selected_parameters:
                parameters_need_covering.remove(j)
                domain_columns[j] = len(domain_columns)
    #print(domain_columns)
    #print(others_th)
    #exit(-1)
    assert len(parameters_need_covering) == 0 and len(domain_columns) == len(this_group)
    for i in others_th:
        others.append(tables_fjbuckets_values[i])
        others_buckets_scope_mapping.append([])
        for j in tfj_join_parameters_cover[i]:
            others_buckets_scope_mapping[-1].append(domain_columns[j])
    # print(len(domain_fjbuckets.buckets), domain_fjbuckets.scope)
    # print('others:')
    # for ith, i in enumerate(others):
    #     print(tables_name[others_th[ith]])
    #     print(len(i.buckets), i.scope)
    #     print(others_buckets_scope_mapping[ith])
    #exit(-1)
    return domain_fjbuckets, others, others_buckets_scope_mapping

def calc_domain_fjbuckets_opt(tables_fjbuckets: dict, this_group: list, join_parameters_nodes: dict, mqspn: MultiQSPN):
    #sort by: (1) cover join_parameters number (this_group), (2) size of fjbuckets.buckets
    assert len(this_group) == 1 and len(join_parameters_nodes) == 1
    #print(tables_fjbuckets)
    score = {i_k: len(i_v.buckets) for i_k, i_v in tables_fjbuckets.items()}
    domain = None
    domain_score = None
    for i_k, i_v in score.items():
        if domain_score is None or i_v < domain_score:
            domain = i_k
            domain_score = i_v
    assert domain is not None
    others = []
    for i_k, i_v in tables_fjbuckets.items():
        if i_k != domain:
            others.append(i_v)
    assert len(others) + 1 == len(tables_fjbuckets)
    domain = tables_fjbuckets[domain]
    return domain, others

def calculate_downscale_factor(index, count_series):
    if index == -1:
        return 0
    return 1 / count_series[index]

# def try_join_table():
#     data_a = {'key': [1, 2, 2, 2, 4], 'value_a': ['A1', 'A2', 'A3', 'A4', 'A5']}
#     data_b = {'key': [2, 2, 3, 4, 5], 'value_b': ['B1', 'B2', 'B3', 'B4', 'B5']}
#     A = pandas.DataFrame(data_a)
#     B = pandas.DataFrame(data_b)
#     AB, fA, fB = get_join_table('A', A, 'key', 'B', B, 'key')
#     print(A)
#     print(B)
#     print(AB)
#     print(fA)
#     print(fB)