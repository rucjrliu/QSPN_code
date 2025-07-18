import pandas as pd
import csv
import os
from pandas.api.types import is_numeric_dtype

from Learning.qspnJoinBase import ufs

def multi_table_workload_csv_reader(path: str):
    workload = [] #(tables, join_predicates, query), join_predicates=[('table.column', 'table.column)], query=[(table, column, op, value)]
    true_card = []
    with open(path, 'r', encoding='utf-8') as filein:
        s = filein.readlines()
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

def multi_table_dataset_csv_reader(data_root: str, dc: dict):
    dataset = {}
    for i in dc:
        data_path = os.path.join(data_root, i + '.csv')
        #print(i)
        with open(data_path, 'r', encoding='utf-8') as filein:
            reader=csv.reader(filein)
            for j in reader:
                header = list(j)
                break
        data = pd.read_csv(data_path, usecols=dc[i])
        #data = data.dropna(axis=1)
        #print(data.shape, data.dtypes)
        for j in data.columns:
            if not is_numeric_dtype(data[j]):
                data[j] = pd.to_numeric(data[j], errors='coerce')
        data = data.dropna()
        dataset[i] = data
    return dataset