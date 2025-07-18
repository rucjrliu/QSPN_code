import numpy as np
import time
from copy import deepcopy
from Structure.nodes import Context, Sum, Product, Factorize, Leaf, QSum, qsplit_maxcut_which_child
from Structure.StatisticalTypes import MetaType
from Structure.leaves.fspn_leaves.Merge_leaves import Merge_leaves
from Learning.validity import is_valid
from Inference.inference import prod_likelihood, sum_likelihood, prod_log_likelihood, sum_log_likelihood, Qsum_likelihood, qsum_likelihood
from Learning.qspnJoinBase import mqspn_sum_prune_by_datadomain, MultiQSPN, FJBuckets, product_merge_FJBuckets2, sum_merge_FJBuckets2, leaf_select_FJBuckets2, calc_domain_fjbuckets2, final_merge_sort_FJBuckets, ve2
from Structure.model import FSPN

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

DETAIL_PERF = True

def get_fjbuckets_bfs2(query, join_scope: list, subroot, attr, this_table_domain: list, joined_tables: set):
    #bfs
    #print('FFFKKK')
    q = []
    result = []
    f = -1
    q.append(subroot)
    while len(q) > f + 1:
        node = q[f + 1]
        # if isinstance(node, Leaf):
        #     print('bfs2', node, 'leaf')
        # else:
        #     print('bfs2', node, node.children)
        f += 1
        #print(node)
        if isinstance(node, Leaf):
            result.append(leaf_select_FJBuckets2(join_scope, node.join_buckets, None, this_table_domain, joined_tables))
        elif isinstance(node, Product):
            result.append([])
            for i in node.children:
                for j in i.scope:
                    if j in join_scope:
                        result[-1].append(len(q))
                        q.append(i)
                        break
            #print(result[-1], q)
            assert len(result[-1]) <= 1
        elif isinstance(node, QSum):
            query_join = deepcopy(query)
            for i in join_scope:
                query_join[0][0, i] = -1
                query_join[1][0, i] = -1
            children = qsplit_maxcut_which_child(node, query_join)
            result.append([])
            for i in children:
                result[-1].append(len(q))
                q.append(i)
        elif isinstance(node, Sum):
            result.append([])
            for i, c in enumerate(node.children):
                result[-1].append(len(q))
                q.append(c)
    #calc
    #print('calc', q)
    #print('calc', result)
    assert len(q) == len(result)
    for i in range(len(q)-1, -1, -1):
        if type(result[i]) is list:
            node = q[i]
            if isinstance(node, Leaf):
                continue
            if isinstance(node, Product):
                tmp_children_list = [result[j] for j in result[i]]
                #print(q, i)
                #print(result)
                #print([q[j] for j in result[i]], tmp_children_list)
                while len(tmp_children_list) < 2:
                    tmp_children_list.append(None)
                result[i] = product_merge_FJBuckets2(tmp_children_list[0], tmp_children_list[1], 1.0)
            elif isinstance(node, QSum):
                tmp_children_list = [result[j] for j in result[i]]
                assert len(tmp_children_list) == 1
                result[i] = tmp_children_list[0]
            elif isinstance(node, Sum):
                tmp_children_list = [result[j] for j in result[i]]
                result[i] = sum_merge_FJBuckets2(tmp_children_list)
            else:
                #print('?', node, result[i])
                assert not 'has implemented'
    #print(result[0])
    return result[0]

def gen_ce_tree_pbfs2(query, root, attr, this_table_domain: list, joined_tables: set):
    #print('FFFKKK')
    #exit(-1)
    #bfs with prune
    q = []
    f = -1
    q.append(root)
    result = []
    while len(q) > f + 1:
        f += 1
        node = q[f]
        if isinstance(node, Leaf):
            #print(node.query(query, attr))
            result.append((node.query(query, attr), leaf_select_FJBuckets2(node.scope, node.join_buckets, query, this_table_domain, joined_tables)))
            #print()
            #exit(-1)
        elif isinstance(node, Product):
            query_scope = set()
            for i in range(query[0].shape[1]):
                if query[0][0, i] != float('-inf') or query[1][0, i] != float('inf'):
                    query_scope.add(i)
            result.append([])
            for i in node.children:
                scope_intersect = False
                for j in i.scope:
                    if j in query_scope:
                        result[-1].append(len(q))
                        q.append(i)
                        scope_intersect = True
                        break
                if not scope_intersect:
                    result[-1].append(None)
        elif isinstance(node, QSum):
            children = qsplit_maxcut_which_child(node, query)
            result.append([])
            for i in children:
                result[-1].append(len(q))
                q.append(i)
            assert len(result[-1]) == 1
        elif isinstance(node, Sum):
            result.append([])
            for i, c in enumerate(node.children):
                res_childi = mqspn_sum_prune_by_datadomain(node, i, query)
                if res_childi is None:
                    result[-1].append(len(q))
                    q.append(c)
                else:
                    result[-1].append(None)
        else:
            assert not 'has implemented'
    #print(q)
    #print(result)
    #exit(-1)
    #exit(-1)
    return q, result

def join_probability_execute_ce_tree_pbfs2(query, join_scope: list, this_table_domain: list, q_node: list, q_edge: list, attr, joined_tables: set):
    #calc
    #print(q_node)
    #print(q_edge)
    #print(join_scope)
    #print('q_node:', q_node)
    #print('q_edge:', q_edge)
    assert len(q_node) == len(q_edge)
    result = [None for i in range(len(q_node))]
    for i in range(len(q_node)-1, -1, -1):
        node = q_node[i]
        if isinstance(node, Leaf):
            result[i] = (q_edge[i][0], leaf_select_FJBuckets2(join_scope, q_edge[i][1], None, this_table_domain, joined_tables))
            #q_edge[i][1]._print()
            #print(result[i])
            #exit(-1)
        elif isinstance(node, Product):
            #print(node)
            tmp_children_list = [None if j is None else result[j] for j in q_edge[i]]
            #print(tmp_children_list)
            #print(node.scope)
            #exit(-1)
            assert len(tmp_children_list) == len(node.children)
            for j, child in enumerate(tmp_children_list):
                if child is None:
                    for k in node.children[j].scope:
                        if k in join_scope:
                            tmp_children_list[j] = (1.0, get_fjbuckets_bfs2(query, join_scope, node.children[j], attr, this_table_domain, joined_tables))
                            break
            #print(tmp_children_list)
            # for j in tmp_children_list:
            #     if j is not None and j[1] is not None:
            #         j[1]._print()
            #exit(-1)
            tmp_buckets = []
            prod = np.array([1.0])
            c = 1.0
            for j in tmp_children_list:
                if j is not None:
                    prod *= j[0]
                    if j[1] is not None:
                        tmp_buckets.append(j[1])
                    else:
                        c *= float(j[0][0])
            #print(prod, c, tmp_buckets)
            #exit(-1)
            while len(tmp_buckets) < 2:
                tmp_buckets.append(None)
            assert len(tmp_buckets) == 2
            #print(node, 'Prod', c)
            #print('SSSKKKFFF')
            #print(node, tmp_children_list)
            #print(tmp_buckets)
            #print('FFFKKKSSS')
            result[i] = (prod, product_merge_FJBuckets2(tmp_buckets[0], tmp_buckets[1], c))
            #print(result[i])
            # if result[i][1] is not None:
            #     result[i][1]._print()
            #exit(-1)
        elif isinstance(node, QSum):
            #print(node)
            tmp_children_list = [result[j] for j in q_edge[i]]
            assert len(tmp_children_list) == 1
            result[i] = tmp_children_list[0]
            #print(result[i])
            #exit(-1)
        elif isinstance(node, Sum):
            #print(node)
            tmp_children_list_sel = []
            tmp_children_list_buckets = []
            assert len(q_edge[i]) == len(node.children)
            for j in q_edge[i]:
                if j is None:
                    tmp_children_list_sel.append(np.array([0.0]))
                else:
                    tmp_children_list_sel.append(result[j][0])
                    tmp_children_list_buckets.append(result[j][1])
            #print(tmp_children_list_sel)
            #print(tmp_children_list_buckets)
            result[i] = (sum_likelihood(node, tmp_children_list_sel), sum_merge_FJBuckets2(tmp_children_list_buckets))
            #print(result[i])
            #exit(-1)
        else:
            assert not 'has implemented'
    #print(result)
    #return exit(-1)
    #print('result0', result[0])
    return result[0]

def mqspn_probability2(mqspn: MultiQSPN, query: dict, attr=None):
    #print('start')
    #cover count of each bigtable
    #print(query)
    bigtable_cover = {i: 0 for i in mqspn.bigtable_columns.keys()}
    table_covered = set()
    bigtable_joined_tables = {}
    #join_pairs = []
    for i in query['join']:
        pair = tuple(i.split('='))
        assert len(pair) == 2
        lt = pair[0].split('.')[0]
        rt = pair[1].split('.')[0]
        assert lt != rt
        if lt > rt:
            lt, rt = rt, lt
        table_covered.add(lt)
        table_covered.add(rt)
        bigtable_cover[(lt, rt)] += 1
        #join_pairs.append(pair)
        # for j in pair:
        #     joined_tables_query_select[j.split('.')[0]] = None
    for i in query['select']:
        it = i[0].split('.')[0]
        for j in mqspn.bigtable_join_info[it][0]:
            if bigtable_cover[j] > 0:
                bigtable_cover[j] += 1
    bigtable_cover = sorted(list(bigtable_cover.items()), key=lambda t:t[1], reverse=True)
    for i, i_cover_n in bigtable_cover:
        if i_cover_n == 0:
            break
        joined_tables_i = set()
        lt, rt = i
        if lt in table_covered:
            joined_tables_i.add(lt)
            table_covered.remove(lt)
        if rt in table_covered:
            joined_tables_i.add(rt)
            table_covered.remove(rt)
        bigtable_joined_tables[(lt, rt)] = joined_tables_i
    #print('bigtable_cover:', bigtable_cover)
    #print('bigtable_joined_tables:', bigtable_joined_tables)
    #exit(-1)
    #extract query(select) of joined_tables
    bigtables_query_select = {}
    for i in bigtable_joined_tables.keys():
        col_n = len(mqspn.bigtable_columns[i])
        bigtables_query_select[i] = (np.zeros((1, col_n)), np.zeros((1, col_n)))
        bigtables_query_select[i][0].fill(float('-inf'))
        bigtables_query_select[i][1].fill(float('inf'))
        for j in query['select']:
            jt, jc = j[0].split('.')
            if jt in i:
                if jc == mqspn.bigtable_join_info[jt][1]:
                    jc = mqspn.bigtable_columns[i]['__join_key__']
                else:
                    jc = mqspn.bigtable_columns[i]['{}.{}'.format(jt, jc)]
                jl, jr = float('-inf'), float('inf')
                if j[1] == '=':
                    jl, jr = j[2], j[2]
                elif j[1] == '<=':
                    jr = j[2]
                elif j[1] == '<':
                    jr = j[2] - 1
                elif j[1] == '>=':
                    jl = j[2]
                elif j[1] == '>':
                    jl = j[2] + 1
                else:
                    assert 'CANNOT extract query select op' + j[1] == False
                if jl > bigtables_query_select[i][0][0, jc]:
                    bigtables_query_select[i][0][0, jc] = jl
                if jr < bigtables_query_select[i][1][0, jc]:
                    bigtables_query_select[i][1][0, jc] = jr
    #print('bigtables_query_select:', bigtables_query_select)
    #print()
    #exit(-1)
    # for i in query['select']:
    #     col_n = len(mqspn.table_columns[i])
    #     query_i_select = (np.zeros((1, col_n)), np.zeros((1, col_n)))
    #     query_i_select[0].fill(float('-inf'))
    #     query_i_select[1].fill(float('inf'))
    #     joined_tables_query_select[i] = query_i_select
    # for i in query['select']:
    #     #(table.column, op, value)
    #     ii = i[0].split('.')
    #     assert len(ii) == 2
    #     it = ii[0]
    #     if it not in joined_tables_query_select:
    #         continue
    #     ic = mqspn.table_columns[it][ii[1]]
    #     il, ir = float('-inf'), float('inf')
    #     if i[1] == '=':
    #         il, ir = i[2], i[2]
    #     elif i[1] == '<=':
    #         ir = i[2]
    #     elif i[1] == '<':
    #         ir = i[2] - 1
    #     elif i[1] == '>=':
    #         il = i[2]
    #     elif i[1] == '>':
    #         il = i[2] + 1
    #     else:
    #         assert 'CANNOT extract query select op' + i[1] == False
    #     if il > joined_tables_query_select[it][0][0, ic]:
    #         joined_tables_query_select[it][0][0, ic] = il
    #     if ir < joined_tables_query_select[it][1][0, ic]:
    #         joined_tables_query_select[it][1][0, ic] = ir

    #print(join_pairs)
    #print(joined_tables_query_select)
    #exit(-1)
    #no join
    '''
    if len(query['join']) == 0:
        #print('NO join')
        assert len(joined_tables_query_select) == 0
        single_table_query_select = {}
        for i in query['select']:
            single_table_name = i[0].split('.')[0]
            single_table_query_select[single_table_name] = None
        #print(single_table_query_select)
        assert len(single_table_query_select) == 1 
        for i in single_table_query_select:
            single_table_query_select[i] = np.zeros((len(mqspn.table_columns[i]), 2))
            single_table_query_select[i][:, 0] = float('-inf')
            single_table_query_select[i][:, 1] = float('inf')
        for i in query['select']:
            #(table.column, op, value)
            ii = i[0].split('.')
            assert len(ii) == 2
            it = ii[0]
            ic = mqspn.table_columns[it][ii[1]]
            il, ir = float('-inf'), float('inf')
            if i[1] == '=':
                il, ir = i[2], i[2]
            elif i[1] == '<=':
                ir = i[2]
            elif i[1] == '<':
                ir = i[2] - 1
            elif i[1] == '>=':
                il = i[2]
            elif i[1] == '>':
                il = i[2] + 1
            else:
                assert 'CANNOT extract query select op' + i[1] == False
            if il > single_table_query_select[it][ic, 0]:
                single_table_query_select[it][ic, 0] = il
            if ir < single_table_query_select[it][ic, 1]:
                single_table_query_select[it][ic, 1] = ir
        only_t, only_q = list(single_table_query_select.items())[0]
        #print(only_t, only_q)
        #exit(-1)
        model = FSPN()
        #print(model.scope, model.range)
        #exit(-1)
        model.model = mqspn.table_qspn_model[only_t]
        est = [round(single_ce) for single_ce in model.probability((only_q[:,0].reshape(1,-1), only_q[:,1].reshape(1,-1)),
                                        calculated=dict(), exist_qsum=True, first_time_recur=True) * mqspn.table_cardinality[only_t]]
        #print(int(est[0]))
        #exit(-1)
        return int(est[0])
    '''
    if DETAIL_PERF:
        perf_qspn_prune = perf_counter()
    #gen ce_tree for joined tables
    bigtables_qspn_ce_tree = {}
    #print('gen ce tree...')
    for i in bigtables_query_select:
        #print(i)
        if attr is None:
            query_attr = mqspn.bigtable_qspn_model[i].scope
        else:
            query_attr = attr
        #print(query_attr)
        #exit(-1)
        #print(mqspn.bigtable_qspn_model)
        #print(bigtable_joined_tables)
        bigtables_qspn_ce_tree[i] = gen_ce_tree_pbfs2(bigtables_query_select[i], mqspn.bigtable_qspn_model[i], query_attr, mqspn.bigtable_domain[i], bigtable_joined_tables[i])
        #print(list(zip(bigtables_qspn_ce_tree[i][0], bigtables_qspn_ce_tree[i][1])))
        #print()
        #exit(-1)
    if DETAIL_PERF:
        perf_qspn_prune = (perf_counter() - perf_qspn_prune) * 1000
    #print()
    #print('executing ce tree...')
    #get fjbuckets of bigtables
    if DETAIL_PERF:
        perf_merge_buckets = perf_counter()
    bigtable_fjbuckets = {}
    for i, cetree_i in bigtables_qspn_ce_tree.items():
        #print(i)
        if attr is None:
            query_attr = mqspn.bigtable_qspn_model[i].scope
        else:
            query_attr = attr
        #print('input:', bigtables_query_select[i], [mqspn.bigtable_columns[i]['__join_key__']], mqspn.bigtable_domain[i], cetree_i[0], cetree_i[1], attr, bigtable_joined_tables[i])
        fjbuckets_l = join_probability_execute_ce_tree_pbfs2(bigtables_query_select[i], [mqspn.bigtable_columns[i]['__join_key__']], mqspn.bigtable_domain[i], cetree_i[0], cetree_i[1], attr, bigtable_joined_tables[i])[1]
        #print(fjbuckets_l)
        #for j in fjbuckets_l:
        #    print(j)
            #j._print(show_bs=True)
        bigtable_fjbuckets[i] = final_merge_sort_FJBuckets(fjbuckets_l)
        #print(bigtable_fjbuckets[i])
        #bigtable_fjbuckets[i]._print(show_bs=True)
        #print()
    if DETAIL_PERF:
        perf_merge_buckets = (perf_counter() - perf_merge_buckets) * 1000
    #exit(-1)
    #ve
    if DETAIL_PERF:
        perf_pre_ve = perf_counter()
    domain_fjbuckets, others = calc_domain_fjbuckets2(bigtable_fjbuckets, bigtable_joined_tables)
    if DETAIL_PERF:
        perf_pre_ve = (perf_counter() - perf_pre_ve) * 1000
        perf_ve = perf_counter()
    ret = ve2(domain_fjbuckets, others)
    #print(ret)
    #exit(-1)
    if DETAIL_PERF:
        perf_ve = (perf_counter() - perf_ve) * 1000
    if DETAIL_PERF:
        return int(round(ret)), perf_qspn_prune, perf_merge_buckets, perf_pre_ve, perf_ve
    else:
        return int(round(ret))

