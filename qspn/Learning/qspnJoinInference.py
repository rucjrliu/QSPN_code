import numpy as np
import time
from copy import deepcopy
from Structure.nodes import Context, Sum, Product, Factorize, Leaf, QSum, qsplit_maxcut_which_child
from Structure.StatisticalTypes import MetaType
from Structure.leaves.fspn_leaves.Merge_leaves import Merge_leaves
from Learning.validity import is_valid
from Inference.inference import prod_likelihood, sum_likelihood, prod_log_likelihood, sum_log_likelihood, Qsum_likelihood, qsum_likelihood
#from Learning.qspnJoinBase import mqspn_sum_prune_by_datadomain, MultiQSPN, FJBuckets, product_merge_FJBuckets, sum_merge_FJBuckets, leaf_select_FJBuckets, get_join_TablesColumns_groups, calc_domain_fjbuckets, ve
from Learning.qspnJoinBase import mqspn_sum_prune_by_datadomain, MultiQSPN, set_attr, product_merge_FJBuckets_opt, sum_merge_FJBuckets_opt, leaf_select_FJBuckets_opt, get_join_TablesColumns_groups, calc_domain_fjbuckets_opt, ve_opt, final_merge_sort_FJBuckets
from Structure.model import FSPN

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

DETAIL_PERF = False

def get_fjbuckets_bfs(query, join_scope: list, subroot, attr, this_table_domain: list):
    #bfs
    q = []
    result = []
    f = -1
    q.append(subroot)
    while len(q) > f + 1:
        node = q[f + 1] #debuged
        f += 1
        #print(node)
        if isinstance(node, Leaf):
            result.append(leaf_select_FJBuckets_opt(join_scope, node.join_buckets, None, this_table_domain))
        elif isinstance(node, Product):
            result.append([])
            for i in node.children:
                for j in i.scope:
                    if j in join_scope:
                        result[-1].append(len(q))
                        q.append(i)
                        break
            assert len(result[-1]) <= 2
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
    assert len(q) == len(result)
    for i in range(len(q)-1, -1, -1):
        if type(result[i]) is list:
            node = q[i]
            if isinstance(node, Leaf):
                continue
            if isinstance(node, Product):
                tmp_children_list = [result[j] for j in result[i]]
                while len(tmp_children_list) < 2:
                    tmp_children_list.append(None)
                result[i] = product_merge_FJBuckets_opt(tmp_children_list[0], tmp_children_list[1], 1.0)
            elif isinstance(node, QSum):
                tmp_children_list = [result[j] for j in result[i]]
                assert len(tmp_children_list) == 1
                result[i] = tmp_children_list[0]
            elif isinstance(node, Sum):
                tmp_children_list = [result[j] for j in result[i]]
                result[i] = sum_merge_FJBuckets_opt(tmp_children_list)
            else:
                assert not 'has implemented'
    return result[0]

def gen_ce_tree_pbfs(query, root, attr, this_table_domain: list):
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
            result.append((node.query(query, attr), leaf_select_FJBuckets_opt(node.scope, node.join_buckets, query, this_table_domain)))
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
    return q, result

def join_probability_execute_ce_tree_pbfs(query, join_scope: list, this_table_domain: list, q_node: list, q_edge: list, attr):
    #calc
    #print(q_node)
    #print(q_edge)
    #print(join_scope)
    assert len(q_node) == len(q_edge)
    result = [None for i in range(len(q_node))]
    for i in range(len(q_node)-1, -1, -1):
        node = q_node[i]
        if isinstance(node, Leaf):
            result[i] = (q_edge[i][0], leaf_select_FJBuckets_opt(join_scope, q_edge[i][1], None, this_table_domain))
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
                            tmp_children_list[j] = (1.0, get_fjbuckets_bfs(query, join_scope, node.children[j], attr, this_table_domain))
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
            #print(node, prod, c, tmp_buckets)
            #exit(-1)
            while len(tmp_buckets) < 2:
                tmp_buckets.append(None)
            assert len(tmp_buckets) == 2
            #print(node, 'Prod', c)
            result[i] = (prod, product_merge_FJBuckets_opt(tmp_buckets[0], tmp_buckets[1], c))
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
            result[i] = (sum_likelihood(node, tmp_children_list_sel), sum_merge_FJBuckets_opt(tmp_children_list_buckets))
            #print(result[i])
            #exit(-1)
        else:
            assert not 'has implemented'
    #print(result)
    #return exit(-1)
    return result[0]

def mqspn_probability(mqspn: MultiQSPN, query: dict, attr=(None,)):
    #print(attr)
    #print('start')
    #extract query(join) for join_pairs and joined_tables
    joined_tables_query_select = {}
    join_pairs = []
    for i in query['join']:
        pair = tuple(i.split('='))
        assert len(pair) == 2
        join_pairs.append(pair)
        for j in pair:
            joined_tables_query_select[j.split('.')[0]] = None
    #extract query(select) of joined_tables
    for i in joined_tables_query_select:
        col_n = len(mqspn.table_columns[i])
        query_i_select = (np.zeros((1, col_n)), np.zeros((1, col_n)))
        query_i_select[0].fill(float('-inf'))
        query_i_select[1].fill(float('inf'))
        joined_tables_query_select[i] = query_i_select
    for i in query['select']:
        #(table.column, op, value)
        ii = i[0].split('.')
        assert len(ii) == 2
        it = ii[0]
        if it not in joined_tables_query_select:
            continue
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
        if il > joined_tables_query_select[it][0][0, ic]:
            joined_tables_query_select[it][0][0, ic] = il
        if ir < joined_tables_query_select[it][1][0, ic]:
            joined_tables_query_select[it][1][0, ic] = ir

    #print(join_pairs)
    #print(joined_tables_query_select)
    #exit(-1)
    #no join
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
    
    if DETAIL_PERF:
        perf_qspn_prune = perf_counter()
    #gen ce_tree for joined tables
    joined_tables_qspn_ce_tree = {}
    #print('gen ce tree...')
    for i in joined_tables_query_select:
        #print(i)
        query_attr = mqspn.table_qspn_model[i].scope
        # if attr is None:
        #     query_attr = mqspn.table_qspn_model[i].scope
        # else:
        #     query_attr = attr
        #print(query_attr)
        #exit(-1)
        joined_tables_qspn_ce_tree[i] = gen_ce_tree_pbfs(joined_tables_query_select[i], mqspn.table_qspn_model[i], query_attr, mqspn.table_domain[i])
    if DETAIL_PERF:
        perf_qspn_prune = (perf_counter() - perf_qspn_prune) * 1000
    #get_join_TablesColumns_groups
    #print('gen tree OK')
    #exit(-1)
    group, join_parameters_nodes = get_join_TablesColumns_groups(mqspn, join_pairs)
    #print(group, join_parameters_nodes)
    #exit(-1)
    #ve
    ve_ret = []
    #print('ve and execute ce tree...')
    #print(len(group), 'groups')
    for i in group:
        #execute ce_tree
        group_join_tables_scope = {}
        group_join_tables_fjbuckets = {}
        for j in i:
            for k in join_parameters_nodes[j]:
                kt, kc = k.split('.')
                if kt not in group_join_tables_scope:
                    group_join_tables_scope[kt] = []
                group_join_tables_scope[kt].append(mqspn.table_columns[kt][kc])
        #print(group_join_tables_scope)
        #exit(-1)
        if DETAIL_PERF:
            perf_merge_buckets = perf_counter()
        for j in group_join_tables_scope:
            #print(j)
            exec_ce_tree_j = join_probability_execute_ce_tree_pbfs(joined_tables_query_select[j], group_join_tables_scope[j], mqspn.table_domain[j], joined_tables_qspn_ce_tree[j][0], joined_tables_qspn_ce_tree[j][1], attr=None)
            #print(j, exec_ce_tree_j[0])
            # if j == 'cast_info':
            #      exec_ce_tree_j[1]._print(no_zero=True)
            group_join_tables_fjbuckets[j] =  final_merge_sort_FJBuckets(exec_ce_tree_j[1])
        if DETAIL_PERF:
            perf_merge_buckets = (perf_counter() - perf_merge_buckets) * 1000
        #print('execute tree OK')
        # for j in group_join_tables_scope:
        #     print(j)
        #     group_join_tables_fjbuckets[j]._print()
        #exit(-1)
        if DETAIL_PERF:
            perf_pre_ve = perf_counter()
        group_domain_fjbuckets, group_others = calc_domain_fjbuckets_opt(group_join_tables_fjbuckets, i, join_parameters_nodes, mqspn)
        if DETAIL_PERF:
            perf_pre_ve = (perf_counter() - perf_pre_ve) * 1000
            perf_ve = perf_counter()
        ve_ret.append(ve_opt(group_domain_fjbuckets, group_others, set_attr(attr[0])))
        if DETAIL_PERF:
            perf_ve = (perf_counter() - perf_ve) * 1000
    #print('ve OK')
    #print(int(np.prod(ve_ret)))
    if DETAIL_PERF:
        return int(round(np.prod(ve_ret))), perf_qspn_prune, perf_merge_buckets, perf_pre_ve, perf_ve
    else:
        return int(round(np.prod(ve_ret)))
