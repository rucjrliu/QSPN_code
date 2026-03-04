from Structure.mqspnJoinBase import MultiQSPN, Table_Models_Partitions, Partition, CEtree_Node, loadmodel_MultiQSPN_table_models_partitions_qspns, enum_all_loc_no, intersection_selected_loc_no_forward, intersection_selected_loc_no_back

import numpy as np
import pickle
from copy import copy
import math

def load_multi_QSPN(path: str):
    with open(path, 'rb') as filein:
        mqspn = pickle.load(filein)
    mqspn = loadmodel_MultiQSPN_table_models_partitions_qspns(mqspn)
    return mqspn

def build_CEtree(q: list, table_models_partitions: dict):
    tree = {node: CEtree_Node(table_models_partitions[node]) for node in q[0]}
    for node, edges in q[1].items():
        tree[node].set_edges([(tree[i[0]], i[1]) for i in edges])
    return tree

def find_CEtree_root_opt(tree: dict, joinkeys_num: int):
    opt_score = (0, 0)
    opt_root = None
    edge_types = np.zeros(joinkeys_num, dtype=int)
    edge_types_n = {}
    for node in tree.values():
        partitions_n = node.table_models_partitions.partitions.size
        edge_types.fill(0)
        for j in node.edges:
            edge_types[j[1]] = 1
        score = (np.sum(edge_types), partitions_n)
        # print(node.table_models_partitions.name, score)
        edge_types_n[node.table_models_partitions.name] = score[0]
        if score > opt_score:
            opt_root, opt_score = node, score
    return opt_root, edge_types_n

def CEtree_dfs_init(node: CEtree_Node):
    node.children = []
    for i in node.edges:
        if i != node.fa:
            node.children.append(i)
            i[0].fa = (node, i[1])
            CEtree_dfs_init(i[0])
    node.edges = None

def CEtree_path_compress(node: CEtree_Node, edge_types_n: dict):
    if len(node.children) == 0:
        return [(node, node.fa[1])]

    children_compress = []
    for i in node.children:
        children_compress.extend(CEtree_path_compress(i[0], edge_types_n))
    
    # print(node.table_models_partitions.name, 'Before compress', [(i[0].table_models_partitions.name, i[1]) for i in children_compress])
    if node.fa[0] is not None and edge_types_n[node.table_models_partitions.name] == 1:
        children_compress.append((node, node.fa[1]))
        node.children = []
    else:
        node.children = children_compress
        for i in node.children:
            i[0].fa = (node, i[1])
        children_compress = [(node, node.fa[1])]
    # print('After compress', [(i[0].table_models_partitions.name, i[1]) for i in children_compress])
    # input()
    
    return children_compress

def CEtree_build_opt(q: list, mqspn: MultiQSPN):
    tree = build_CEtree(q, mqspn.table_models_partitions)
    # for node in tree.values():
    #     node._print()
    # exit(-1)
    opt_root, edges_types_n = find_CEtree_root_opt(tree, len(mqspn.global_joinkeys_range))
    # exit(-1)
    opt_root.fa = (None, -1)
    CEtree_dfs_init(opt_root)
    CEtree_dfs_check(opt_root)
    print()
    # exit(-1)
    root_children_compress = CEtree_path_compress(opt_root, edges_types_n)
    # exit(-1)
    assert len(root_children_compress) == 1 and root_children_compress[0][1] == -1
    return opt_root

def CEtree_build(q: list, mqspn: MultiQSPN):
    tree = build_CEtree(q, mqspn.table_models_partitions)
    root = tree.popitem()[1]
    root.fa = (None, -1)
    CEtree_dfs_init(root)
    return root

def partitions_merge_UniIndep(loc_projected: int, partitions_to_merge: list, joinkey_th: int):
    assert len(partitions_to_merge) > 0
    if len(partitions_to_merge) == 1:
        return partitions_to_merge[0]
    
    merged_n = 0
    for i in partitions_to_merge:
        merged_n += i.n
    merged_ndv = None
    # print('test merge{')
    # print([i._partition_info() for i in partitions_to_merge])
    # print(merged_n, merged_ndv)
    # exit(-1)

    return Partition(merged_n, merged_ndv)

def partitions_merge_UniIndep_n_ndv(loc_projected: int, partitions_to_merge: list, joinkey_th: int, mqspn: MultiQSPN):
    assert len(partitions_to_merge) > 0
    if len(partitions_to_merge) == 1:
        return partitions_to_merge[0]
    
    w = mqspn.calc_partition_width(joinkey_th, loc_projected)
    merged_n = 0
    merged_ndv = [None] * len(mqspn.global_joinkeys_range)
    miss_prob = 1.0
    for i in partitions_to_merge:
        merged_n += i.n
        miss_prob *= 1 - i.ndv[joinkey_th] / w
    merged_ndv[joinkey_th] = w * (1 - miss_prob)
    # print('test merge{')
    # print([i._partition_info() for i in partitions_to_merge])
    # print(merged_n, merged_ndv)
    # exit(-1)

    return Partition(merged_n, merged_ndv)

def partitions_merge_UniIndep_n_ndv_speval(loc_projected: int, partitions_to_merge: list, joinkey_th: int, mqspn: MultiQSPN):
    if joinkey_th != mqspn.speval_joinkey:
        return partitions_merge_UniIndep_n_ndv(loc_projected, partitions_to_merge, joinkey_th, mqspn)
    
    assert len(partitions_to_merge) > 0
    if len(partitions_to_merge) == 1:
        return partitions_to_merge[0]
    
    w = mqspn.speval_joinkey_partition_width[loc_projected]
    merged_n = 0
    merged_ndv = [None] * len(mqspn.global_joinkeys_range)
    miss_prob = 1.0
    for i in partitions_to_merge:
        merged_n += i.n
        miss_prob *= 1 - i.ndv[joinkey_th] / w
    merged_ndv[joinkey_th] = w * (1 - miss_prob)
    # print('test merge{')
    # print([i._partition_info() for i in partitions_to_merge])
    # print(merged_n, merged_ndv)
    # exit(-1)

    return Partition(merged_n, merged_ndv)

def partitions_merge_Uni_n_maxndv(loc_projected: int, partitions_to_merge: list, joinkey_th: int, mqspn: MultiQSPN):
    assert len(partitions_to_merge) > 0
    if len(partitions_to_merge) == 1:
        return partitions_to_merge[0]
    
    w = mqspn.calc_partition_width(joinkey_th, loc_projected)
    merged_n = 0
    merged_ndv = [None] * len(mqspn.global_joinkeys_range)
    merged_ndv[joinkey_th] = partitions_to_merge[0].ndv[joinkey_th]
    for i in partitions_to_merge:
        merged_ndv[joinkey_th] = max(merged_ndv[joinkey_th], i.ndv[joinkey_th])
    # print('test merge{')
    # print([i._partition_info() for i in partitions_to_merge])
    # print(merged_n, merged_ndv)
    # exit(-1)

    return Partition(merged_n, merged_ndv)

def partition_join_calc_UniIndep(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node, mqspn: MultiQSPN):
    W = [None if i is None else mqspn.calc_partition_width(ith, i) for ith, i in enumerate(loc)]

    joined_n = center_partition.n * center_partition_singletable_prob
    print(center_partition.n, '*', center_partition_singletable_prob, end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        joined_n *= child.calculated_partitions[child_no].n / W[child_joinkey_th]
        print('*', '{}/{}'.format(child.calculated_partitions[child_no].n, W[child_joinkey_th]), end=' ')
    print('->', end=' ')
    joined_ndv = None
    
    return joined_n, joined_ndv

def partition_join_calc_UniEasyCorr(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node, mqspn: MultiQSPN):
    W = [1 if i is None else mqspn.calc_partition_width(ith, i) for ith, i in enumerate(loc)]
    divW = [0] * len(W)

    joined_n = center_partition.n * center_partition_singletable_prob
    print(center_partition.n, '*', center_partition_singletable_prob, end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        divW[child_joinkey_th] += 1
        joined_n *= child.calculated_partitions[child_no].n
        print('*', '{}'.format(child.calculated_partitions[child_no].n), end=' ')
    pr_divW = sorted([i for i in range(len(W))], key=lambda k: W[k] ** divW[k], reverse=True)
    for ith, i in enumerate(pr_divW):
        print('/', W[i], '**', 1/(ith+1), '**', divW[i], end=' ')
        joined_n /= W[i] ** (divW[i] / (ith+1))
    joined_ndv = None
    print('->', end=' ')
    
    return joined_n, joined_ndv

def partition_join_calc_DoubleUniEasyCorr(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node, mqspn: MultiQSPN):
    W = [1 if i is None else mqspn.calc_partition_width(ith, i) for ith, i in enumerate(loc)]
    divW = [1] * len(W)
    divW_cnt = [2] * len(W)

    joined_n = center_partition.n * center_partition_singletable_prob
    print(center_partition.n, '*', center_partition_singletable_prob, end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        
        divW[child_joinkey_th] *= W[child_joinkey_th] ** (1/divW_cnt[child_joinkey_th])
        divW_cnt[child_joinkey_th] += 1

        joined_n *= child.calculated_partitions[child_no].n
        print('*', '{}'.format(child.calculated_partitions[child_no].n), end=' ')
    pr_divW = sorted([i for i in range(len(W))], key=lambda k: divW[k], reverse=True)
    for ith, i in enumerate(pr_divW):
        print('/', '{}(W={},divW_cnt={})'.format(divW[i], W[i], divW_cnt[i]), '**', 1/(ith+1), end=' ')
        joined_n /= divW[i] ** (1/(ith+1))
    joined_ndv = None
    print('->', end=' ')
    
    return joined_n, joined_ndv

def partition_join_calc_UniEasyCorr_n_ndv(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node, mqspn: MultiQSPN):
    W = [1 if i is None else mqspn.calc_partition_width(ith, i) for ith, i in enumerate(loc)]
    scale = [1] * len(W)

    joined_n = center_partition.n * center_partition_singletable_prob
    joined_ndv = [None if i is None else i * (1 - (1 - center_partition_singletable_prob) ** (center_partition.n / i)) for i in center_partition.ndv]
    print(center_partition._partition_info(), '*', center_partition_singletable_prob, '->', (joined_n, joined_ndv), end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        child_partition = child.calculated_partitions[child_no]

        child_partition_ndv = child_partition.ndv[child_joinkey_th]
        scale[child_joinkey_th] *= child_partition_ndv / W[child_joinkey_th]

        joined_n *= child_partition.n / child_partition_ndv

        print('*', '{}/{}'.format(child.calculated_partitions[child_no].n, child_partition_ndv), end=' ')
    
    for ith, i in enumerate(scale):
        if joined_ndv[ith] is not None:
            joined_ndv[ith] *= i
    
    scale = sorted(scale)
    for ith, i in enumerate(scale):
        print('*', i, '**', 1/(ith+1), end=' ')
        joined_n *= i ** (1/(ith+1))
    print('->', end=' ')
    
    return joined_n, joined_ndv

def partition_join_calc_UniEasyCorr_n_ndv_speval(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node, mqspn: MultiQSPN):
    W = []
    for ith, i in enumerate(loc):
        if i is None:
            W.append(1)
        elif ith == mqspn.speval_joinkey:
            W.append(mqspn.speval_joinkey_partition_width[i])
        else:
            W.append(mqspn.calc_partition_width(ith, i))
    scale = [1] * len(W)

    joined_n = center_partition.n * center_partition_singletable_prob
    joined_ndv = [None if i is None else i * (1 - (1 - center_partition_singletable_prob) ** (center_partition.n / i)) for i in center_partition.ndv]
    print(center_partition._partition_info(), '*', center_partition_singletable_prob, '->', (joined_n, joined_ndv), end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        child_partition = child.calculated_partitions[child_no]

        child_partition_ndv = child_partition.ndv[child_joinkey_th]
        scale[child_joinkey_th] *= child_partition_ndv / W[child_joinkey_th]

        joined_n *= child_partition.n / child_partition_ndv

        print('*', '{}/{}'.format(child.calculated_partitions[child_no].n, child_partition_ndv), end=' ')
    
    for ith, i in enumerate(scale):
        if joined_ndv[ith] is not None:
            joined_ndv[ith] *= i
    
    scale = sorted(scale)
    for ith, i in enumerate(scale):
        print('*', i, '**', 1/(ith+1), end=' ')
        joined_n *= i ** (1/(ith+1))
    print('->', end=' ')
    
    return joined_n, joined_ndv

def partition_join_calc_DoubleUniEasyCorr_n_ndv(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node, mqspn: MultiQSPN):
    joined_n = center_partition.n * center_partition_singletable_prob
    joined_ndv = [None if i is None else i * (1 - (1 - center_partition_singletable_prob) ** (center_partition.n / i)) for i in center_partition.ndv]
    used_joinkey_th = [False] * len(loc)
    for i in node.children:
        used_joinkey_th[i[1]] = True
    W = [mqspn.calc_partition_width(ith, i) if used_joinkey_th[ith] else None for ith, i in enumerate(loc)]
    scale = [None if i is None else i/joined_ndv[ith] for ith, i in enumerate(W)]
    scale_items = [None if i is None else [1/i] for i in scale]
    scale_str = [None if i is None else ['{}/{}'.format(i, joined_ndv[ith])] for ith, i in enumerate(W)]

    print(center_partition._partition_info(), '*', center_partition_singletable_prob, '->', (joined_n, joined_ndv), end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        child_partition = child.calculated_partitions[child_no]

        child_partition_ndv = child_partition.ndv[child_joinkey_th]
        scale_items[child_joinkey_th].append(child_partition_ndv / W[child_joinkey_th])
        #scale_str[child_joinkey_th].append('{}/{}'.format(child_partition_ndv, W[child_joinkey_th]))

        joined_n *= child_partition.n / child_partition_ndv

        print('*', '{}/{}'.format(child.calculated_partitions[child_no].n, child_partition_ndv), end=' ')
    
    for ith, i in enumerate(scale_items):
        if i is not None:
            scale_item_i = sorted(i)
            for jth, j in enumerate(scale_item_i):
                scale[ith] *= j ** (1/(1+jth))
                scale_str[ith].append('{}**{}'.format(j, (1/(1+jth))))
    
    scale_order = sorted([i for i in range(len(scale))], key=lambda k: np.inf if scale[k] is None else scale[k])
    
    for ith, i in enumerate(scale_order):
        if scale[i] is not None:
            joined_n *= scale[i] ** (1/(1+ith))
            joined_ndv[i] *= scale[i]
            print('* ({})**{}'.format(' * '.join(scale_str[i]), (1/(1+ith))), end=' ')
    
    print('->', end=' ')
    
    return joined_n, joined_ndv

def partition_join_calc_UniEasyCorr_n_min_ndv(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node, mqspn: MultiQSPN):
    joined_n = center_partition.n * center_partition_singletable_prob
    joined_ndv = [None if i is None else i * (1 - (1 - center_partition_singletable_prob) ** (center_partition.n / i)) for i in center_partition.ndv]
    used_joinkey_th = [False] * len(loc)
    for i in node.children:
        used_joinkey_th[i[1]] = True
    min_ndv = [joined_ndv[ith] if i else None for ith, i in enumerate(used_joinkey_th)]

    print(center_partition._partition_info(), '*', center_partition_singletable_prob, '->', (joined_n, joined_ndv), end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        child_partition = child.calculated_partitions[child_no]

        child_partition_ndv = child_partition.ndv[child_joinkey_th]
        min_ndv[child_joinkey_th] = min(min_ndv[child_joinkey_th], child_partition_ndv)

        joined_n *= child_partition.n / child_partition_ndv

        print('*', '{}/{}'.format(child.calculated_partitions[child_no].n, child_partition_ndv), end=' ')
    
    scale_order = sorted([i for i in range(len(min_ndv))], key=lambda k: np.inf if min_ndv[k] is None else min_ndv[k] / joined_ndv[k])
    
    for ith, i in enumerate(scale_order):
        if min_ndv[i] is not None:
            joined_n *= (min_ndv[i] / joined_ndv[i]) ** (1/(1+ith))
            print('* ({}/{})**{}'.format(min_ndv[i], joined_ndv[i], (1/(1+ith))), end=' ')
            joined_ndv[i] = min_ndv[i]
    
    print('->', end=' ')
    
    return joined_n, joined_ndv

def partition_join_calc_Uni_n_min_ndvscale(loc: tuple, center_partition: Partition, center_partition_singletable_prob: float, node: CEtree_Node):
    joined_n = center_partition.n * center_partition_singletable_prob
    joined_ndv = [None if i is None else i * (1 - (1 - center_partition_singletable_prob) ** (center_partition.n / i)) for i in center_partition.ndv]
    used_joinkey_th = [False] * len(loc)
    for i in node.children:
        used_joinkey_th[i[1]] = True
    min_scale = 1
    min_scale_updown = None
    min_scale_joinkey_th = None

    print(center_partition._partition_info(), '*', center_partition_singletable_prob, '->', (joined_n, joined_ndv), end=' ')
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        child_partition = child.calculated_partitions[child_no]

        child_partition_ndv = child_partition.ndv[child_joinkey_th]
        scale = child_partition_ndv / joined_ndv[child_joinkey_th]
        if scale < min_scale:
            min_scale = scale
            min_scale_updown = (child_partition_ndv, joined_ndv[child_joinkey_th])
            min_scale_joinkey_th = child_joinkey_th

        joined_n *= child_partition.n / child_partition_ndv

        print('*', '{}/{}'.format(child.calculated_partitions[child_no].n, child_partition_ndv), end=' ')
    
    if min_scale_updown is not None:
        joined_n *= min_scale
        joined_ndv[min_scale_joinkey_th] *= min_scale
        print('* {}/{}'.format(min_scale_updown[0], min_scale_updown[1]), end=' ')
    
    print('->', end=' ')
    
    return joined_n, joined_ndv

def partition_singletable_prob_calc(qspn, filtering_predicates):
    if type(filtering_predicates) is tuple:
        return min(1.0, qspn.probability(filtering_predicates, calculated=dict(), exist_qsum=True, first_time_recur=True, nasupport=True)[0])
    else:
        return filtering_predicates

NONE_LOC_NO = {None: ()}
def multi_QSPN_CEtree_calc_opt_speval(node: CEtree_Node, tables_filtering_predicates: dict, mqspn: MultiQSPN):
    if len(node.children) > 0:
        print('Before intersection')
        intersection_selected_loc_no_forward(node)
        print('After intersection')
        print(node.table_models_partitions.name, node.selected_loc_no)
        for i in node.children:
            print(i[0].table_models_partitions.name, i[0].selected_loc_no)
        print()
        # exit(-1)
        for i in node.children:
            multi_QSPN_CEtree_calc_opt_speval(i[0], tables_filtering_predicates, mqspn)
        print('Before intersection')
        intersection_selected_loc_no_back(node)
        print('After intersection')
        print(node.table_models_partitions.name, node.selected_loc_no)
        print()
        # exit(-1)

    ret_partitions_to_merge = {}
    ret_CE = 0.0

    filtering_predicates = tables_filtering_predicates[node.table_models_partitions.name]
    ret_joinkey_th = node.fa[1]
    print(node.table_models_partitions.name, 'enum{')
    print(filtering_predicates)
    print()
    for loc, no in enum_all_loc_no(node.selected_loc_no, node.table_models_partitions.partitions, 0, (), ()):
        center_partition_singletable_prob = partition_singletable_prob_calc(node.table_models_partitions.qspns[no], filtering_predicates)
        center_partition = node.table_models_partitions.partitions[no]
        print(loc, no, center_partition_singletable_prob)
        if center_partition_singletable_prob == 0:
            continue
        
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_DoubleUniEasyCorr(loc, node.table_models_partitions.partitions[no], center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_DoubleUniEasyCorr_n_ndv(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr_n_min_ndv(loc, node.table_models_partitions.partitions[no], center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_Uni_n_min_ndvscale(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr_n_ndv(loc, node.table_models_partitions.partitions[no], center_partition_singletable_prob, node, mqspn)
        joined_n, joined_ndv = partition_join_calc_UniEasyCorr_n_ndv_speval(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_Uni_n_min_ndvscale(loc, center_partition, center_partition_singletable_prob, node)
        print(joined_n, joined_ndv)
        if ret_joinkey_th == -1:
            ret_CE += joined_n
        else:
            joined_partition = Partition(joined_n, joined_ndv)
            loc_projected = loc[ret_joinkey_th]
            if loc_projected in ret_partitions_to_merge:
                ret_partitions_to_merge[loc_projected].append(joined_partition)
            else:
                ret_partitions_to_merge[loc_projected] = [joined_partition]
    print('}')
    if ret_joinkey_th == -1:
        return ret_CE
    else:
        node.selected_loc_no = [{j: (jth,) for jth, j in enumerate(ret_partitions_to_merge)} if ith == ret_joinkey_th else NONE_LOC_NO for ith, i in enumerate(node.selected_loc_no)]
        node.calculated_partitions = np.full(len(node.selected_loc_no[ret_joinkey_th]), None)
        for loc, no in node.selected_loc_no[ret_joinkey_th].items():
            #node.calculated_partitions[no] = partitions_merge_UniIndep(loc, ret_partitions_to_merge[loc], ret_joinkey_th)
            node.calculated_partitions[no] = partitions_merge_UniIndep_n_ndv_speval(loc, ret_partitions_to_merge[loc], ret_joinkey_th, mqspn)
            #node.calculated_partitions[no] = partitions_merge_Uni_n_maxndv(loc, ret_partitions_to_merge[loc], ret_joinkey_th, mqspn)
        print('merge{')
        node._print()
        print('}')
        print()
        input()
        return

def multi_QSPN_CEtree_calc_opt(node: CEtree_Node, tables_filtering_predicates: dict, mqspn: MultiQSPN):
    if len(node.children) > 0:
        print('Before intersection')
        intersection_selected_loc_no_forward(node)
        print('After intersection')
        print(node.table_models_partitions.name, node.selected_loc_no)
        for i in node.children:
            print(i[0].table_models_partitions.name, i[0].selected_loc_no)
        print()
        # exit(-1)
        for i in node.children:
            multi_QSPN_CEtree_calc_opt(i[0], tables_filtering_predicates, mqspn)
        print('Before intersection')
        intersection_selected_loc_no_back(node)
        print('After intersection')
        print(node.table_models_partitions.name, node.selected_loc_no, node.table_models_partitions.partitions_loc_no)
        print()
        # exit(-1)

    ret_partitions_to_merge = {}
    ret_CE = 0.0

    filtering_predicates = tables_filtering_predicates[node.table_models_partitions.name]
    ret_joinkey_th = node.fa[1]
    print(node.table_models_partitions.name, 'enum{')
    print(filtering_predicates)
    print()
    for loc, no in enum_all_loc_no(node.selected_loc_no, node.table_models_partitions.partitions, 0, (), ()):
        center_partition_singletable_prob = partition_singletable_prob_calc(node.table_models_partitions.qspns[no], filtering_predicates)
        center_partition = node.table_models_partitions.partitions[no]
        print(loc, no, center_partition_singletable_prob)
        if center_partition_singletable_prob == 0:
            continue
        
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_DoubleUniEasyCorr(loc, node.table_models_partitions.partitions[no], center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_DoubleUniEasyCorr_n_ndv(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr_n_min_ndv(loc, node.table_models_partitions.partitions[no], center_partition_singletable_prob, node, mqspn)
        joined_n, joined_ndv = partition_join_calc_Uni_n_min_ndvscale(loc, center_partition, center_partition_singletable_prob, node)
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr_n_ndv(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        print(joined_n, joined_ndv)
        if ret_joinkey_th == -1:
            ret_CE += joined_n
        else:
            joined_partition = Partition(joined_n, joined_ndv)
            loc_projected = loc[ret_joinkey_th]
            if loc_projected in ret_partitions_to_merge:
                ret_partitions_to_merge[loc_projected].append(joined_partition)
            else:
                ret_partitions_to_merge[loc_projected] = [joined_partition]
    print('}')
    if ret_joinkey_th == -1:
        return ret_CE
    else:
        # print('debug', node.selected_loc_no, node.table_models_partitions.partitions_loc_no)
        node.selected_loc_no = [{j: (jth,) for jth, j in enumerate(ret_partitions_to_merge)} if ith == ret_joinkey_th else NONE_LOC_NO for ith, i in enumerate(node.selected_loc_no)]
        # print('debug', node.selected_loc_no, node.table_models_partitions.partitions_loc_no)
        # input()
        node.calculated_partitions = np.full(len(node.selected_loc_no[ret_joinkey_th]), None)
        for loc, no in node.selected_loc_no[ret_joinkey_th].items():
            #node.calculated_partitions[no] = partitions_merge_UniIndep(loc, ret_partitions_to_merge[loc], ret_joinkey_th)
            node.calculated_partitions[no] = partitions_merge_UniIndep_n_ndv(loc, ret_partitions_to_merge[loc], ret_joinkey_th, mqspn)
            #node.calculated_partitions[no] = partitions_merge_Uni_n_maxndv(loc, ret_partitions_to_merge[loc], ret_joinkey_th, mqspn)
        print('merge{')
        node._print()
        print('}')
        print()
        input()
        return

def CEtree_dfs_check(node: CEtree_Node):
    node._print()
    for i in node.children:
        CEtree_dfs_check(i[0])

def CEtree_clean(node: CEtree_Node, speval_table_models_partitions: dict):
    for i in node.children:
        CEtree_clean(i[0], speval_table_models_partitions)
    node.table_models_partitions = speval_table_models_partitions[node.table_models_partitions.name]
    node.selected_loc_no = None
    node.calculated_partitions = None

def CE_multi_QSPN_opt(q: list, mqspn: MultiQSPN):
    root = CEtree_build_opt(q, mqspn)

    root.selected_loc_no = root.table_models_partitions.partitions_loc_no.copy()
    CEtree_dfs_check(root)
    print()
    input()
    #exit(-1)
    ret_CE = multi_QSPN_CEtree_calc_opt(root, q[2], mqspn)
    print(ret_CE)
    print()
    input()
    print('speval')
    print()
    #do again multi_QSPN_CEtree_calc_opt by mqspn.speval_table_models_partitions
    CEtree_clean(root, mqspn.speval_table_models_partitions)
    root.selected_loc_no = root.table_models_partitions.partitions_loc_no.copy()
    CEtree_dfs_check(root)
    print()
    input()
    # input()
    # exit(-1)
    ret_CE_speval = multi_QSPN_CEtree_calc_opt_speval(root, q[2], mqspn)
    print(ret_CE_speval)
    print()
    input()
    # exit(-1)

    return ret_CE+ret_CE_speval