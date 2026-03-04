from Structure.mqspnJoinBaseNew import MultiQSPN, Table_Models_Partitions, Partition, CEtree_Node, loadmodel_MultiQSPN_table_models_partitions_qspns, enum_all_loc_no, intersection_selected_loc_no_forward, intersection_selected_loc_no_back

import numpy as np
import pickle
from copy import copy

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
    # CEtree_dfs_check(opt_root)
    # print()
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

def CEtree_build2(q: list, mqspn: MultiQSPN):
    tree = build_CEtree(q, mqspn.table_models_partitions)
    opt_root, _ = find_CEtree_root_opt(tree, len(mqspn.global_joinkeys_range))
    opt_root.fa = (None, -1)
    CEtree_dfs_init(opt_root)
    return opt_root

def partitions_merge_UniIndep_n_ndv_speval_n_speval_ndv(loc_projected: int, partitions_to_merge: list, joinkey_th: int, mqspn: MultiQSPN):
    assert len(partitions_to_merge) > 0
    if len(partitions_to_merge) == 1:
        return partitions_to_merge[0]

    w = mqspn.calc_partition_width(joinkey_th, loc_projected)
    speval_w = mqspn.speval_joinkey_partition_width[loc_projected] if joinkey_th == mqspn.speval_joinkey else None
    miss_prob = 1.0
    speval_miss_prob = 1.0
    merged_n = None
    merged_ndv = None
    merged_speval_n = None
    merged_speval_ndv = None
    for i in partitions_to_merge:
        if i.n is not None:
            if merged_n is None:
                merged_n = 0
                merged_ndv = [None] * len(mqspn.global_joinkeys_range)
            merged_n += i.n
            miss_prob *= 1 - i.ndv[joinkey_th] / w
        if i.speval_n is not None:
            if merged_speval_n is None:
                merged_speval_n = 0
                merged_speval_ndv = [None] * len(mqspn.global_joinkeys_range)
            merged_speval_n += i.speval_n
            speval_miss_prob *= 1 - i.speval_ndv[joinkey_th] / speval_w
    
    if merged_ndv is not None:
        merged_ndv[joinkey_th] = w * (1 - miss_prob)
    if merged_speval_ndv is not None:
        merged_speval_ndv[joinkey_th] = speval_w * (1 - speval_miss_prob)
    
    return Partition(merged_n, merged_ndv, merged_speval_n, merged_speval_ndv)

def partition_singletable_prob_calc(qspn, filtering_predicates, query_scope):
    if type(filtering_predicates) is tuple:
        if qspn is None:
            return None
        #return min(1.0, qspn.probability(filtering_predicates, calculated=dict(), exist_qsum=True, first_time_recur=True, nasupport=True, query_scope=query_scope)[0])
        return min(1.0, qspn._probability_pbfs_nasupport_opt(filtering_predicates, qspn.model, qspn.model.scope, query_scope)[0])
    else:
        return filtering_predicates

def partition_join_calc_Uni_n_min_ndvscale_speval_UniEasyCorr_n_ndv(loc: tuple, center_partition: Partition, center_partition_singletable_prob, center_partition_singletable_speval_prob, node: CEtree_Node, mqspn: MultiQSPN):
    if center_partition.ndv is None or center_partition_singletable_prob is None:
        joined_n = center_partition.n
        joined_ndv = center_partition.ndv
    elif center_partition_singletable_prob == 0:
        joined_n = None
        joined_ndv = None
    else:
        joined_n = center_partition.n * center_partition_singletable_prob
        joined_ndv = [None if i is None else i * (1 - (1 - center_partition_singletable_prob) ** (center_partition.n / i)) for i in center_partition.ndv]
    
    if center_partition.speval_ndv is None or center_partition_singletable_speval_prob is None:
        joined_speval_n = center_partition.speval_n
        joined_speval_ndv = center_partition.speval_ndv
    elif center_partition_singletable_speval_prob == 0:
        joined_speval_n = None
        joined_speval_ndv = None
    else:
        joined_speval_n = center_partition.speval_n * center_partition_singletable_speval_prob
        joined_speval_ndv = [None if i is None else i * (1 - (1 - center_partition_singletable_speval_prob) ** (center_partition.speval_n / i)) for i in center_partition.speval_ndv]
    
    min_scale, min_scale_joinkey_th = 1, None
    # min_nonspeval_scale, min_nonspeval_scale_joinkey_th = 1, None
    W = []
    for ith, i in enumerate(loc):
        if i is None:
            W.append(1)
        elif ith == mqspn.speval_joinkey:
            W.append(mqspn.speval_joinkey_partition_width[i])
        else:
            W.append(mqspn.calc_partition_width(ith, i))
    speval_scales = [1] * len(W)
    for i in node.children:
        child = i[0]
        child_joinkey_th = i[1]
        child_no = child.selected_loc_no[child_joinkey_th][loc[child_joinkey_th]]
        child_partition = child.calculated_partitions[child_no]
        # child_partition._print()

        if joined_n is not None:
            if child_partition.n is None:
                joined_n = None
                joined_ndv = None
            else:
                child_partition_ndv = child_partition.ndv[child_joinkey_th]
                scale = child_partition_ndv / joined_ndv[child_joinkey_th]
                if scale < min_scale:
                    min_scale = scale
                    min_scale_joinkey_th = child_joinkey_th
                if child_joinkey_th != mqspn.speval_joinkey:
                    speval_scales[child_joinkey_th] *= child_partition_ndv / W[child_joinkey_th]
                cnt_per_ndv = child_partition.n / child_partition_ndv
                joined_n *= cnt_per_ndv
                if child_joinkey_th != mqspn.speval_joinkey and joined_speval_n is not None:
                    joined_speval_n *= cnt_per_ndv
        
        if joined_speval_n is not None and child_joinkey_th == mqspn.speval_joinkey:
            if  child_partition.speval_n is None:
                joined_speval_n = None
                joined_speval_ndv = None
            else:
                child_partition_speval_ndv = child_partition.speval_ndv[child_joinkey_th]
                speval_scales[child_joinkey_th] *= child_partition_speval_ndv / W[child_joinkey_th]
                joined_speval_n *= child_partition.speval_n / child_partition_speval_ndv
    
    if joined_n is not None and min_scale_joinkey_th is not None:
        joined_n *= min_scale
        joined_ndv[min_scale_joinkey_th] *= min_scale
    if joined_speval_n is not None:
        for ith, i in enumerate(speval_scales):
            if joined_speval_ndv[ith] is not None:
                joined_speval_ndv[ith] *= i
        speval_scales = sorted(speval_scales)
        for ith, i in enumerate(speval_scales):
            joined_speval_n *= i ** (1/(ith+1))
    
    return joined_n, joined_ndv, joined_speval_n, joined_speval_ndv

NONE_LOC_NO = {None: ()}
POS_INF = np.inf
NEG_INF = -POS_INF
def multi_QSPN_CEtree_calc_opt(node: CEtree_Node, tables_filtering_predicates: dict, mqspn: MultiQSPN):
    if len(node.children) > 0:
        # print('Before intersection')
        intersection_selected_loc_no_forward(node)
        # print('After intersection')
        # print(node.table_models_partitions.name, node.selected_loc_no)
        # for i in node.children:
        #     print(i[0].table_models_partitions.name, i[0].selected_loc_no)
        # print()
        # exit(-1)
        for i in node.children:
            multi_QSPN_CEtree_calc_opt(i[0], tables_filtering_predicates, mqspn)
        # print('Before intersection')
        intersection_selected_loc_no_back(node)
        # print('After intersection')
        # print(node.table_models_partitions.name, node.selected_loc_no, node.table_models_partitions.partitions_loc_no)
        # print()
        # exit(-1)

    ret_partitions_to_merge = {}
    ret_CE = 0.0

    filtering_predicates = tables_filtering_predicates[node.table_models_partitions.name]
    query_scope = set(ith for ith, (l, r) in enumerate(zip(filtering_predicates[0][0], filtering_predicates[1][0])) if l != NEG_INF or r != POS_INF) if type(filtering_predicates) is tuple else None
    ret_joinkey_th = node.fa[1]
    # print(node.table_models_partitions.name, 'enum{')
    # print(filtering_predicates)
    # print()
    for loc, no in enum_all_loc_no(node.selected_loc_no, node.table_models_partitions.partitions, 0, (), ()):
        no_qspn = node.table_models_partitions.partitions[no].qspn
        no_speval_qspn = node.table_models_partitions.partitions[no].speval_qspn
        center_partition_singletable_prob = partition_singletable_prob_calc(no_qspn, filtering_predicates, query_scope)
        center_partition_singletable_speval_prob = partition_singletable_prob_calc(no_speval_qspn, filtering_predicates, query_scope)
        check_singletable_prob = 0 if center_partition_singletable_prob is None else center_partition_singletable_prob
        check_singletable_speval_prob = 0 if center_partition_singletable_speval_prob is None else center_partition_singletable_speval_prob
        if check_singletable_prob == 0 and check_singletable_speval_prob == 0:
            continue
        center_partition = node.table_models_partitions.partitions[no]
        # print(loc, no, center_partition_singletable_prob, center_partition_singletable_speval_prob, center_partition._partition_info())
        
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_DoubleUniEasyCorr(loc, node.table_models_partitions.partitions[no], center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_DoubleUniEasyCorr_n_ndv(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr_n_min_ndv(loc, node.table_models_partitions.partitions[no], center_partition_singletable_prob, node, mqspn)
        #joined_n, joined_ndv = partition_join_calc_Uni_n_min_ndvscale(loc, center_partition, center_partition_singletable_prob, node)
        #joined_n, joined_ndv = partition_join_calc_UniEasyCorr_n_ndv(loc, center_partition, center_partition_singletable_prob, node, mqspn)
        joined_n, joined_ndv, joined_speval_n, joined_speval_ndv = partition_join_calc_Uni_n_min_ndvscale_speval_UniEasyCorr_n_ndv(loc, center_partition, center_partition_singletable_prob, center_partition_singletable_speval_prob, node, mqspn)
        # print(joined_n, joined_ndv, joined_speval_n, joined_speval_ndv)
        # print()
        if ret_joinkey_th == -1:
            if joined_n is not None:
                ret_CE += joined_n
            if joined_speval_n is not None:
                ret_CE += joined_speval_n
        else:
            loc_projected = loc[ret_joinkey_th]
            if ret_joinkey_th == mqspn.speval_joinkey and (joined_n is not None or joined_speval_n is not None):
                joined_partition = Partition(joined_n, joined_ndv, joined_speval_n, joined_speval_ndv)
                if loc_projected in ret_partitions_to_merge:
                    ret_partitions_to_merge[loc_projected].append(joined_partition)
                else:
                    ret_partitions_to_merge[loc_projected] = [joined_partition]
            else:
                joined_partition = Partition(joined_n, joined_ndv, None, None) if joined_n is not None else None
                joined_partition2 = Partition(joined_speval_n, joined_speval_ndv, None, None) if joined_speval_n is not None else None
                if loc_projected in ret_partitions_to_merge:
                    if joined_partition is not None:
                        ret_partitions_to_merge[loc_projected].append(joined_partition)
                    if joined_partition2 is not None:
                        ret_partitions_to_merge[loc_projected].append(joined_partition2)
                elif joined_partition is not None or joined_partition2 is not None:
                    ret_partitions_to_merge[loc_projected] = []
                    if joined_partition is not None:
                        ret_partitions_to_merge[loc_projected].append(joined_partition)
                    if joined_partition2 is not None:
                        ret_partitions_to_merge[loc_projected].append(joined_partition2)
    # print('}')
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
            node.calculated_partitions[no] = partitions_merge_UniIndep_n_ndv_speval_n_speval_ndv(loc, ret_partitions_to_merge[loc], ret_joinkey_th, mqspn)
            #node.calculated_partitions[no] = partitions_merge_Uni_n_maxndv(loc, ret_partitions_to_merge[loc], ret_joinkey_th, mqspn)
        # print('merge{')
        # node._print()
        # print('}')
        # print()
        # input()
        return

def CE_multi_QSPN_opt(q: list, mqspn: MultiQSPN):
    root = CEtree_build(q, mqspn)
    #root = CEtree_build_opt(q, mqspn)

    root.selected_loc_no = root.table_models_partitions.partitions_loc_no.copy()
    # CEtree_dfs_check(root)
    # print()
    # input()
    #exit(-1)
    ret_CE = multi_QSPN_CEtree_calc_opt(root, q[2], mqspn)
    # exit(-1)

    return ret_CE