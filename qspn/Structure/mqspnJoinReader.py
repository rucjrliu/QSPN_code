import pandas as pd
import csv
import os
from pandas.api.types import is_numeric_dtype
from copy import deepcopy
import numpy as np

from Structure.mqspnJoinBase import MultiQSPN

# def na_fix_wkld(s):
#     for ith, i in enumerate(s):
#         newi = deepcopy(i)
#         if 'votes.BountyAmount' in newi:
#             newi = newi.replace('votes', 'votes3')
#         elif 'votes.UserId' in i:
#             newi = newi.replace('votes', 'votes2')
#         if 'posts.FavoriteCount' in i:
#             newi = newi.replace('posts', 'posts3')
#         elif 'posts.AnswerCount' in i or 'posts.ViewCount' in i:
#             newi = newi.replace('posts', 'posts2')
#         s[ith] = newi

# def na_fix_dc(dc):
#     if 'votes' in dc and 'posts' in dc:
#         dc_votes = (dc['votes'] | dc['votes2'] | dc['votes3'])
#         dc_posts = (dc['posts'] | dc['posts2'] | dc['posts3'])
#         dc['votes'] = deepcopy(dc_votes)
#         dc['votes'].discard('BountyAmount')
#         dc['votes'].discard('UserId')
#         dc['votes2'] = deepcopy(dc_votes)
#         dc['votes2'].discard('BountyAmount')
#         dc['votes3'] = deepcopy(dc_votes)
#         dc['posts'] = deepcopy(dc_posts)
#         dc['posts'].discard('AnswerCount')
#         dc['posts'].discard('ViewCount')
#         dc['posts'].discard('FavoriteCount')
#         dc['posts2'] = deepcopy(dc_posts)
#         dc['posts2'].discard('FavoriteCount')
#         dc['posts3'] = deepcopy(dc_posts)

# def na_fix_join_graph(dc, join_graph):
#     for head, nodes in join_graph.items():
#         nodes_completed = deepcopy(nodes)
#         for i in nodes:
#             t, c = i.split('.')
#             if t in ['votes', 'votes2', 'votes3']:
#                 for j in ['votes', 'votes2', 'votes3']:
#                     k = '.'.join([j, c])
#                     if k not in nodes_completed:
#                         nodes_completed.append(k)
#             elif t in ['posts', 'posts2', 'posts3']:
#                 for j in ['posts', 'posts2', 'posts3']:
#                     k = '.'.join([j, c])
#                     if k not in nodes_completed:
#                         nodes_completed.append(k)

def multi_table_workload_csv_reader_testset(path: str):
    workload = [] #(tables, join_predicates, query), join_predicates=[('table.column', 'table.column)], query=[(table, column, op, value)]
    true_card = []
    with open(path, 'r', encoding='utf-8') as filein:
        s = filein.readlines()
    # if 'stats' in path:
    #     na_fix_wkld(s)
    # for i in s:
    #     print(i, end='')
    # exit(-1)
    for i in s:
        s_tables, s_join_preds, s_query, s_truecard = i.strip('\n').strip('\r').split('#')
        #print(s_tables, s_join_preds, s_query, s_truecard)
        tables = s_tables.split(',')
        if s_join_preds == '':
            join_preds = []
        else:
            join_preds = [tuple(j.split('=')) for j in s_join_preds.split(',')]
        if s_query == '':
            query = []
        else:
            query_preds = s_query.split(',')
            #print(tables, join_preds, query_preds)
            #if len(query_preds) == 1 and query_preds[0] == '':
            #    query_preds = []
            assert len(query_preds) % 3 == 0
            query = []
            for j in range(0, len(query_preds), 3):
                jt, jc = query_preds[j].split('.')
                jop = query_preds[j+1]
                jv = float(query_preds[j+2])
                query.append((jt, jc, jop, jv))
        #print(tables, join_preds, query)
        workload.append((tables, join_preds, query))
        true_card.append(int(s_truecard))
    workload = fix_statsceb_nan_workload_testset(path, workload)
    assert len(workload) == len(true_card), 'len(workload) != len(true_card) in testset'
    return workload, true_card

def multi_table_workload_extract(workload: list, mqspn: MultiQSPN):
    extracted_workload = []
    #join predicates
    for q in workload:
        new_q = [q[0], {i: [] for i in q[0]}, {}]
        for i in q[1]:
            table_a, joinkey_a = i[0].split('.')
            table_b, joinkey_b = i[1].split('.')
            joinkey_a = mqspn.table_joinkeys[table_a][joinkey_a]
            joinkey_b = mqspn.table_joinkeys[table_b][joinkey_b]
            new_q[1][table_a].append((table_b, joinkey_a))
            new_q[1][table_b].append((table_a, joinkey_b))
        extracted_workload.append(new_q)
    #filtering predicates
    for q, new_q in zip(workload, extracted_workload):
        flag_no_filtering_predicates = {}
        for j in q[0]:
            query_ndarray = (np.full((1, len(mqspn.table_filtering_columns[j])), -np.inf), np.full((1, len(mqspn.table_filtering_columns[j])), np.inf))
            new_q[2][j] = query_ndarray
            flag_no_filtering_predicates[j] = True
        for k in q[2]:
            query_ndarray = new_q[2][k[0]]
            flag_no_filtering_predicates[k[0]] = False
            k_scope = mqspn.table_filtering_columns[k[0]][k[1]]
            if k[2] == '=':
                query_ndarray[0][0, k_scope] = max(k[3], query_ndarray[0][0, k_scope])
                query_ndarray[1][0, k_scope] = min(k[3], query_ndarray[1][0, k_scope])
            elif k[2] == '<=':
                query_ndarray[1][0, k_scope] = min(k[3], query_ndarray[1][0, k_scope])
            elif k[2] == '<':
                query_ndarray[1][0, k_scope] = min(k[3] - 1, query_ndarray[1][0, k_scope])
            elif k[2] == '>=':
                query_ndarray[0][0, k_scope] = max(k[3], query_ndarray[0][0, k_scope])
            elif k[2] == '>':
                query_ndarray[0][0, k_scope] = max(k[3] + 1, query_ndarray[0][0, k_scope])
        for j in new_q[0]:
            if flag_no_filtering_predicates[j]:
                new_q[2][j] = 1.0
            elif (new_q[2][j][0] > new_q[2][j][1]).any():
                new_q[2][j] = 0.0
            # else:
            #     print([float(k) for k in new_q[2][j][0][0]], [float(k) for k in new_q[2][j][1][0]])
    return extracted_workload

def multi_table_workload_csv_reader_trainset(path: str):
    workload = [] #(tables, join_predicates, query), join_predicates=[('table.column', 'table.column)], query=[(table, column, op, value)]
    true_card = []
    with open(path, 'r', encoding='utf-8') as filein:
        s = filein.readlines()
    # if 'stats' in path:
    #     na_fix_wkld(s)
    # for i in s:
    #     print(i, end='')
    # exit(-1)
    for i in s:
        s_tables, s_join_preds, s_query, s_truecard = i.strip('\n').strip('\r').split('#')
        #print(s_tables, s_join_preds, s_query, s_truecard)
        tables = s_tables.split(',')
        if s_join_preds == '':
            join_preds = []
        else:
            join_preds = [tuple(j.split('=')) for j in s_join_preds.split(',')]
        if s_query == '':
            query = []
        else:
            query_preds = s_query.split(',')
            #print(tables, join_preds, query_preds)
            #if len(query_preds) == 1 and query_preds[0] == '':
            #    query_preds = []
            assert len(query_preds) % 3 == 0
            query = []
            for j in range(0, len(query_preds), 3):
                jt, jc = query_preds[j].split('.')
                jop = query_preds[j+1]
                jv = float(query_preds[j+2])
                query.append((jt, jc, jop, jv))
        #print(tables, join_preds, query)
        workload.append((tables, join_preds, query))
        true_card.append(int(s_truecard))
    workload = fix_statsceb_nan_workload_trainset(path, workload)
    return workload, true_card

def workload_join_pattern_pairs(workload):
    join_pattern = {}
    for i in workload:
        #print(i)
        #return
        for j in i[1]:
            assert len(j) == 2
            lp = j[0]
            rp = j[1]
            lt, lc = j[0].split('.')
            rt, rc = j[1].split('.')
            if lt > rt:
                lt, rt = rt, lt
                lc, rc = rc, lc
                lp, rp = rp, lp
            if (lt, rt) not in join_pattern:
                join_pattern[(lt, rt)] = {(lp, rp): 1}
            elif (lp, rp) not in join_pattern[(lt, rt)]:
                join_pattern[(lt, rt)][(lp, rp)] = 1
            else:
                join_pattern[(lt, rt)][(lp, rp)] += 1
    return join_pattern

def workload_select_pattern(workload):
    select_pattern = {}
    for i in workload:
        #print(i[2])
        #return
        for j in i[2]:
            assert len(j) == 4
            pt = j[0]
            pc = j[1]
            po = j[2]
            pv = j[3]
            if pt not in select_pattern:
                select_pattern[pt] = {pc: 1}
            elif pc not in select_pattern[pt]:
                select_pattern[pt][pc] = 1
            else:
                select_pattern[pt][pc] += 1
    return select_pattern

def workload_join_pattern_tables(workload):
    join_pattern = {}
    for i in workload:
        for j in i[0]:
            for k in i[0]:
                if j != k:
                    lt = min(j, k)
                    rt = max(j, k)
                    if (lt, rt) not in join_pattern:
                        join_pattern[(lt, rt)] = 1
                    else:
                        join_pattern[(lt, rt)] += 1
    return join_pattern

def ufs(belong: dict, x):
    if belong[x] != x:
        belong[x] = ufs(belong, belong[x])
    return belong[x]

def workload_data_columns_stats(workload):
    dc = {}
    join_belong = {}
    for i in workload:
        for j in i[0]:
            if j not in dc:
                dc[j] = set()
        for j in i[1]:
            for k in j:
                kt, kc = k.split('.')
                dc[kt].add(kc)
            j0 = j[0]
            if j0 not in join_belong:
                join_belong[j0] = j0
            for k in range(1, len(j)):
                if j[k] not in join_belong:
                    join_belong[j[k]] = join_belong[j0]
                else:
                    ufs(join_belong, j[k])
                    ufs(join_belong, j0)
                    join_belong[join_belong[j[k]]] = join_belong[j0]
        for j in i[2]:
            assert len(j) == 4
            dc[j[0]].add(j[1])
    join_graph = {}
    for i in join_belong:
        ufs(join_belong, i)
    for i, b in join_belong.items():
        if b not in join_graph:
            join_graph[b] = [i]
        else:
            join_graph[b].append(i)
    # for i in dc:
    #     print('{}: {}'.format(i, list(dc[i])))
    return dc, join_graph
#path = 'mscn_queries_neurocard_format.csv'
#workload, true_card = multi_table_workload_csv_reader(path)
#for i in range(50):
#    print(workload[i], true_card[i])
#dc = data_columns_stats(workload)

# def na_fix_dataset_reader(data_path):
#     for i in ['votes2', 'votes3']:
#         if i in data_path:
#             data_path = data_path.replace(i, 'votes')
#             return data_path
#     for i in ['posts2', 'posts3']:
#         if i in data_path:
#             data_path = data_path.replace(i, 'posts')
#             return data_path
#     return data_path

def set_joinkeys_th(join_graph):
    global_th_joinkeys = [None] * len(join_graph)
    table_th_joinkeys = {}
    for ith, i in enumerate(join_graph.items()):
        global_th_joinkeys[ith] = i[1]
        for j in i[1]:
            jt, jc = j.split('.')
            if jt not in table_th_joinkeys:
                table_th_joinkeys[jt] = [None] * len(global_th_joinkeys)
            table_th_joinkeys[jt][ith] = jc
    return global_th_joinkeys, table_th_joinkeys

# def fix_statsceb_nan_table(data_root: str, dataset: dict, table_th_joinkeys: dict):
#     fixed_nan_dataset = {}
#     if 'stats' in data_root:
#         for t, df in dataset.items():
#             fixed_nan_dataset[t] = df
#             if t in set({'comments', 'postHistory', 'votes'}):
#                 fixed_nan_dataset[t + '_nouser'] = df.drop('UserId', axis=1)
#             elif t == 'posts':
#                 fixed_nan_dataset[t + '_nouser'] = df.drop('OwnerUserId', axis=1)
#     else:
#         fixed_nan_dataset = dataset
#     return fixed_nan_dataset

def init_p(dataset_name: str, partition_width_limit):
    if dataset_name == 'job':
        if partition_width_limit is None:
            return (1500000, 5000, 1000000)
        else:
            return (partition_width_limit, 5000, 1000000)
    elif dataset_name == 'stats':
        if partition_width_limit is None:
            return (60000, 100, 10000)
        else:
            return (partition_width_limit, 100, 10000)
    else:
        return None

def fix_statsceb_nan_workload_trainset(path: str, workload: list):
    fixed_nan_workload = []
    if 'stats' in path:
        check_table = {'comments': ['UserId'], 'postHistory': ['UserId'], 'votes': ['UserId'], 'posts': ['OwnerUserId']}
        fixed = {'comments': 'comments_nouser', 'postHistory': 'postHistory_nouser', 'votes': 'votes_nouser', 'posts': 'posts_nouser'}
        for qth, q in enumerate(workload):
            fixed_nan_workload.append(q)
            check_res = False
            for i in q[1]:
                for j in i:
                    t, joinkey = j.split('.')
                    if t in check_table:
                        check_res = True
            if check_res:
                new_q = ([], [], [])
                for ith, i in enumerate(q[0]):
                    if check_res:
                        if i in check_table:
                            new_q[0].append(fixed[i])
                        else:
                            new_q[0].append(i)
                for ith, i in enumerate(q[1]):
                    new_j = []
                    for j in i:
                        t, joinkey = j.split('.')
                        if t in check_table:
                            if joinkey not in check_table[t]:
                                new_j.append('.'.join([fixed[t], joinkey]))
                        else:
                            new_j.append(j)
                    if len(new_j) == 2:
                        new_q[1].append(tuple(new_j))
                for ith, i in enumerate(q[2]):
                    if check_res:
                        if i[0] in check_table:
                            new_q[2].append((fixed[i[0]], i[1], i[2], i[3]))
                        else:
                            new_q[2].append(i)
                fixed_nan_workload.append(new_q)
        
        workload = fixed_nan_workload
        fixed_nan_workload = []

        for qth, q in enumerate(workload):
            check2_res = {'PostId': 0, 'RelatedPostId': 0}
            for i in q[1]:
                for j in i:
                    t, joinkey = j.split('.')
                    if t == 'postLinks' and joinkey in check2_res:
                        check2_res[joinkey] = 1
            assert check2_res['PostId'] + check2_res['RelatedPostId'] < 2
            if check2_res['RelatedPostId']:
                new_q_turn_postLinks_to_postLinks_Related = ([], [], [])
                new_q_del_RelatedPostId = ([], [], [])
                for ith, i in enumerate(q[0]):
                    if i == 'postLinks':
                        new_q_turn_postLinks_to_postLinks_Related[0].append('postLinks_Related')
                    else:
                        new_q_turn_postLinks_to_postLinks_Related[0].append(i)
                    new_q_del_RelatedPostId[0].append(i)
                for ith, i in enumerate(q[1]):
                    new_j_turn_postLinks_to_postLinks_Related = []
                    new_j_del_RelatedPostId = []
                    for j in i:
                        t, joinkey = j.split('.')
                        if t == 'postLinks':
                            new_j_turn_postLinks_to_postLinks_Related.append('.'.join(['postLinks_Related', joinkey]))
                        else:
                            new_j_turn_postLinks_to_postLinks_Related.append(j)
                            new_j_del_RelatedPostId.append(j)
                    assert len(new_j_turn_postLinks_to_postLinks_Related) == 2
                    new_q_turn_postLinks_to_postLinks_Related[1].append(tuple(new_j_turn_postLinks_to_postLinks_Related))
                    if len(new_j_del_RelatedPostId) == 2:
                        new_q_del_RelatedPostId[1].append(new_j_del_RelatedPostId)
                for ith, i in enumerate(q[2]):
                    if i[0] == 'postLinks':
                        new_q_turn_postLinks_to_postLinks_Related[2].append(('postLinks_Related', i[1], i[2], i[3]))
                    else:
                        new_q_turn_postLinks_to_postLinks_Related[2].append(i)
                    new_q_del_RelatedPostId[2].append(i)
                fixed_nan_workload.append(new_q_del_RelatedPostId)
                fixed_nan_workload.append(new_q_turn_postLinks_to_postLinks_Related)
            elif check2_res['PostId']:
                fixed_nan_workload.append(q)
                new_q_del_PostId_turn_postLinks_to_postLinks_Related = ([], [], [])
                for ith, i in enumerate(q[0]):
                    if i == 'postLinks':
                        new_q_del_PostId_turn_postLinks_to_postLinks_Related[0].append('postLinks_Related')
                    else:
                        new_q_del_PostId_turn_postLinks_to_postLinks_Related[0].append(i)
                for ith, i in enumerate(q[1]):
                    new_j_del_PostId_turn_postLinks_to_postLinks_Related = []
                    for j in i:
                        t, joinkey = j.split('.')
                        if t != 'postLinks':
                            new_j_del_PostId_turn_postLinks_to_postLinks_Related.append(j)
                    if len(new_j_del_PostId_turn_postLinks_to_postLinks_Related) == 2:
                        new_q_del_PostId_turn_postLinks_to_postLinks_Related[1].append(new_j_del_PostId_turn_postLinks_to_postLinks_Related)
                for ith, i in enumerate(q[2]):
                    if i[0] == 'postLinks':
                        new_q_del_PostId_turn_postLinks_to_postLinks_Related[2].append(('postLinks_Related', i[1], i[2], i[3]))
                    else:
                        new_q_del_PostId_turn_postLinks_to_postLinks_Related[2].append(i)
                fixed_nan_workload.append(new_q_del_PostId_turn_postLinks_to_postLinks_Related)
            else:
                fixed_nan_workload.append(q)
    
    else:
        fixed_nan_workload = workload
    
    return fixed_nan_workload

def fix_statsceb_nan_workload_testset(path: str, workload: list):
    fixed_nan_workload = []
    if 'stats' in path:
        check_cond = {'comments': 'UserId', 'postHistory': 'UserId', 'votes': 'UserId', 'posts': 'OwnerUserId', 'postLinks': 'PostId'}
        fixed = {'comments': 'comments_nouser', 'postHistory': 'postHistory_nouser', 'votes': 'votes_nouser', 'posts': 'posts_nouser', 'postLinks': 'postLinks_Related'}
        for qth, q in enumerate(workload):
            check_res = set()
            for i in q[1]:
                for j in i:
                    t, joinkey = j.split('.')
                    if t in check_cond and joinkey == check_cond[t]:
                        check_res.add(t)
            new_q = deepcopy(q)
            for ith, i in enumerate(q[0]):
                if i in check_cond and i not in check_res:
                    new_q[0][ith] = fixed[i]
            for ith, i in enumerate(q[1]):
                new_j = []
                for j in i:
                    t, joinkey = j.split('.')
                    if t in check_cond and t not in check_res:
                        new_j.append('.'.join([fixed[t], joinkey]))
                    else:
                        new_j.append(j)
                assert len(new_j) == 2
                new_q[1][ith] = tuple(new_j)
            for ith, i in enumerate(q[2]):
                if i[0] in check_cond and i[0] not in check_res:
                    new_q[2][ith] = (fixed[i[0]], i[1], i[2], i[3])
            fixed_nan_workload.append(new_q)
    
    else:
        fixed_nan_workload = workload
    
    return fixed_nan_workload

def fix_statsceb_nan_data_filename(data_root: str, filename: str):
    if 'stats' in data_root:
        if '_nouser' in filename:        
            return filename[0 : len(filename)-len('_nouser')]
        elif '_Related' in filename:
            return filename[0 : len(filename)-len('_Related')]
        else:
            return filename
    else:
        return filename

def multi_table_dataset_csv_reader(data_root: str, dc: dict, table_th_joinkeys: dict):
    #data_root = '/home/liujw/neurocard-master/neurocard/datasets/job'
    dataset = {}
    for i in dc:
        # print(i, end=';')
        filename = fix_statsceb_nan_data_filename(data_root, i)
        data_path = os.path.join(data_root, filename + '.csv')
        # data_path = na_fix_dataset_reader(data_path)
        #print(i)
        with open(data_path, 'r', encoding='utf-8') as filein:
            reader=csv.reader(filein)
            for j in reader:
                header = list(j)
                break
        data = pd.read_csv(data_path, usecols=dc[i])
        # print(data.shape, end=';')
        #data = data.dropna(axis=1)
        #print(data.shape, data.dtypes)
        for j in data.columns:
            if not is_numeric_dtype(data[j]):
                data[j] = pd.to_numeric(data[j], errors='coerce')
        #data = data.dropna()
        # print(data.shape)
        dropna_subset = []
        for j in table_th_joinkeys[i]:
            if j is not None:
                dropna_subset.append(j)
        dataset[i] = data.dropna(subset=dropna_subset)
        #print(i, 'origin_shape:', data.shape, 'joinkey_dropna_shape:', dataset[i].shape)
    # exit(-1)
    #dataset = fix_statsceb_nan_table(data_root, dataset, table_th_joinkeys)
    return dataset