import sys
import pandas
import argparse
import os
import pandas as pd 
import pickle 
import json 
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

import settings
sys.path.append(str(settings.PROJECT_PATH / 'qspn'))

from Learning.learningWrapper import learn_FSPN
from Structure.leaves.parametric.Parametric import Categorical
from Structure.model import FSPN
from Structure.nodes import Context
from Learning.statistics import get_structure_stats
from Learning.updateQSPN import update_QSPN

try:
    from time import perf_counter 
except:
    from time import time
    perf_counter = time

def input_update_wkld(file: str):
    wkld = []
    input_wkld = np.load(file)
    for i in input_wkld:
        if np.all(i[:, 1] == -1):
            wkld.append(('INSERT INTO', i[:, 0]))
        else:
            wkld.append(('SELECT', i))
    q_n = 0
    upd_n = 0
    for i in wkld:
        if i[0] == 'SELECT':
            q_n += 1
        elif i[0] == 'INSERT INTO':
            upd_n += 1
    print('SQLs: {} queries, {} update data rows, {} in total.'.format(q_n, upd_n, len(wkld)))
    return wkld

global origin_data, origin_workload

def run_qspn(qspn, args, sql):
    sel0 = None
    if sql[0] == 'SELECT':
        model = FSPN()
        model.model = qspn
        model.store_factorize_as_dict()
        sel0 = model.probability((sql[1][:,0].reshape(1,-1), sql[1][:,1].reshape(1,-1)),
                                            calculated=dict(), exist_qsum=True, first_time_recur=True)[0]
    return qspn, sel0

def retrain_qspn(qspn, args, cluster, data, workload, update_data, new_queries):
    upded_data = pd.concat([data, update_data], axis=0)
    upded_data = upded_data.values.astype(int)
    upded_workload = np.concatenate((workload, new_queries), axis=0)
    rebuild_parametric_types = [Categorical for i in range(len(origin_data.columns))]
    rebuild_ds_context = Context(parametric_types=rebuild_parametric_types).add_domains(upded_data)
    qspn = learn_FSPN(
        upded_data,
        rebuild_ds_context,
        workload=upded_workload,
        queries=cluster,
        rdc_sample_size=100000,
        rdc_strong_connection_threshold=1.1,
        multivariate_leaf=False,
        threshold=0.3,
        wkld_attr_threshold=0.01,
        wkld_attr_bound=(args.Nx, args.lower, args.upper)
    )
    return qspn

qspn_ada_cache_size = 20
qspn_ada_cache_d = []
qspn_ada_cache_q = []
qspn_ada_upded_data = None
qspn_ada_upded_workload = None
def run_qspn_adaptive_incremental(qspn, args, sql):
    global qspn_ada_cache_size, qspn_ada_cache_d, qspn_ada_cache_q, qspn_ada_upded_data, qspn_ada_upded_workload
    sel0 = None
    if qspn_ada_upded_data is None:
        qspn_ada_upded_data = deepcopy(origin_data)
    if qspn_ada_upded_workload is None:
        qspn_ada_upded_workload = deepcopy(origin_workload)
    if sql[0] == 'SELECT':
        qspn_ada_cache_q.append(sql[1])
    elif sql[0] == 'INSERT INTO':
        qspn_ada_cache_d.append(sql[1])
    #try update qspn
    if len(qspn_ada_cache_q) >= qspn_ada_cache_size:
        upd_data_array = np.array(qspn_ada_cache_d)
        if upd_data_array.shape != (len(qspn_ada_cache_d), len(origin_data.columns)):
            upd_data_array = np.zeros(shape=(len(qspn_ada_cache_d), len(origin_data.columns)), dtype=int)
        upd_data = pd.DataFrame(pd.DataFrame({c: upd_data_array[:, i] for i, c in enumerate(origin_data.columns)}))
        new_queries = np.array(qspn_ada_cache_q)
        print('upd_data:', upd_data.shape)
        print('new_queries:', new_queries.shape)
        #exit(-1)
        qspn = update_QSPN(
            qspn,
            data=qspn_ada_upded_data,
            workload=qspn_ada_upded_workload,
            data_insert=upd_data,
            data_delete = None,
            new_queries = new_queries,
            rdc_sample_size=100000,
            rdc_threshold=0.3,
            wkld_attr_threshold=0.01,
            wkld_attr_bound=(args.Nx, args.lower, args.upper)
        )
        qspn_ada_upded_data = pd.concat([qspn_ada_upded_data, upd_data], axis=0)
        qspn_ada_upded_workload = np.concatenate((qspn_ada_upded_workload, new_queries), axis=0)
        print('qspn_ada_upded_data:', qspn_ada_upded_data.shape)
        print('qspn_ada_upded_workload:', qspn_ada_upded_workload.shape)
        print()
        #exit(-1)
        qspn_ada_cache_d= []
        qspn_ada_cache_q = []
    #run qspn
    if sql[0] == 'SELECT':
        model = FSPN()
        model.model = qspn
        model.store_factorize_as_dict()
        sel0 = model.probability((sql[1][:,0].reshape(1,-1), sql[1][:,1].reshape(1,-1)),
                                            calculated=dict(), exist_qsum=True, first_time_recur=True)[0]
    return qspn, sel0

def update_1by1(run_qspn_func, update_qspn_func, args, cluster, model_save, MODEL_PATH, query_path, update_query_path, model_prefix, data_path, data_path_df, columns):
    global origin_data, origin_workload
    #input origin_data, origin_workload, origin_cardinality
    if args.model == 'qspn' or args.model_path is not None:
        model_save_path = model_save
    else:
        model_save_path = str(MODEL_PATH / f"{model_prefix}.pkl")
    print('original model: {}'.format(model_save_path))
    model_size = os.path.getsize(model_save_path)
    print(f"original Model Size: {model_size/1000} KB")
    with open('qspn_update_expr.log', 'a') as dumplog:
        dumplog.write('Original Model Size: {} KB\n'.format(model_size/1000))
    #load origin model
    qspn = None
    with open(model_save_path, 'rb') as f:
        qspn = pickle.load(f)
    print(get_structure_stats(qspn))
    #model = FSPN()
    #model.model = spn 
    #model.store_factorize_as_dict()
    #read data (.csv)
    data_csv = pd.read_csv(str(data_path), header=None)
    data_csv = data_csv.drop(0)
    origin_data = deepcopy(data_csv)
    origin_cardinality = data_csv.shape[0]
    print(type(data_csv))
    print(f"origin_data shape: {data_csv.shape}")
    print(f"origin_cardinality: {origin_cardinality}")
    #read data (.hdf)
    data_df = pd.read_hdf(data_path_df, key="dataframe") #.hdf
    data_columns = data_df.columns
    data_columns_lower = {c.lower(): i for i, c in enumerate(data_df.columns)}
    columns_lower = {c.lower(): i for i, c in enumerate(columns)}
    print(data_columns)
    print(columns)
    assert data_columns_lower == columns_lower
    origin_cardinality_df = data_df.shape[0]
    print(type(data_df))
    print(f"origin_data(.hdf) shape: {data_df.shape}")
    print(f"origin_cardinality: {origin_cardinality_df}")
    data_list = list(np.array(data_df))
    #read origin workload
    origin_workload_path = os.path.join(query_path, "train_query_sc.npy")
    origin_workload = np.load(origin_workload_path)
    print(f"origin_workload: {origin_workload_path}")
    print("origin_workload_shape:", origin_workload.shape)
    #input sqls
    #print(update_query_path)
    running_sqls_path = os.path.join(update_query_path, "test_query_sc.npy")
    running_sqls = input_update_wkld(running_sqls_path)
    print(f"running_sqls: {running_sqls_path}")
    #exit(-1)
    #run
    interval_sql_num = 0
    est_card = []
    true_card = []
    total_time = 0
    seq_time = 0
    seq_q_errs = []
    interval_update_data_list = []
    interval_update_data_csv = pd.DataFrame({i: [] for i in data_csv.columns})
    interval_running_queries = []
    last_update_data = deepcopy(origin_data)
    last_update_workload = deepcopy(origin_workload)
    interval_times = 0
    for sql in tqdm(running_sqls):
        #interval retrain
        if interval_sql_num > len(running_sqls) // 10:
            if update_qspn_func is not None:
                interval_running_queries = np.array(interval_running_queries)
                #print('data_df:', data_df.shape)
                interval_update_data_array = np.array(interval_update_data_list)
                if interval_update_data_array.shape != (len(interval_update_data_list), len(data_csv.columns)):
                    interval_update_data_array = np.zeros(shape=(len(interval_update_data_list), len(data_csv.columns)), dtype=int)
                interval_update_data_csv = pd.DataFrame({c: interval_update_data_array[:, i] for i, c in enumerate(data_csv.columns)})
                print('Update: {} tuples, {} queries'.format(len(interval_update_data_csv), len(interval_running_queries)))
                start = perf_counter()
                qspn = update_qspn_func(qspn, args, cluster, last_update_data, last_update_workload, interval_update_data_csv, interval_running_queries)
                delta_time = perf_counter() - start
                seq_time += delta_time
                total_time += delta_time
                #print(last_update_data)
                last_update_data = pd.concat([last_update_data, interval_update_data_csv], axis=0)
                last_update_workload = np.concatenate((last_update_workload, interval_running_queries), axis=0)
                print('last_update_data:', last_update_data.shape)
                print('last_update_workload:', last_update_workload.shape)
                #print(last_update_data)
                print()
                #exit(-1)
                interval_update_data_csv = pd.DataFrame({i: [] for i in data_csv.columns})
                interval_running_queries = []
            with open('qspn_update_expr.log', 'a') as dumplog:
                maxerr = max(seq_q_errs)
                sumerr = sum(seq_q_errs)
                smooth_meanerr = max(1.0, (sumerr - maxerr) / max(1, len(seq_q_errs)-1))
                dumplog.write('Seq Time={} min., |Q|={}, Seq Mean Q-err={}, Seq smooth Mean Q-err(NoMAX)={}\n'.format(seq_time / 60, len(seq_q_errs), np.mean(seq_q_errs) if len(seq_q_errs) > 0 else 0, smooth_meanerr))
            seq_time = 0
            seq_q_errs = []
            interval_sql_num = 0
            interval_times += 1
        #calc true_card of sql and prep interval retrain
        if sql[0] == 'SELECT':
            data_array = np.array(data_list)
            data_df = pd.DataFrame({c: data_array[:, i] for i, c in enumerate(data_df.columns)})
            pred = ['{}<={}<={}'.format(sql[1][j, 0], c, sql[1][j, 1]) for j, c in enumerate(data_df.columns)]
            preds = ' and '.join(pred)
            #print(data_df.columns)
            #print(data_csv.columns)
            true_card.append(len(data_df.query(preds)))
            if update_qspn_func is not None:
                interval_running_queries.append(sql[1])
        #update data_df for true_card and prep interval retrain
        elif sql[0] == 'INSERT INTO':
            #print(sql[1])
            #print(data_df)
            #data_df.loc[len(data_df)] = sql[1]
            data_list.append(sql[1])
            if update_qspn_func is not None:
                interval_update_data_list.append(sql[1])
                #interval_update_data_csv.loc[len(interval_update_data_csv)] = sql[1]
        #run qspn on sql
        assert run_qspn_func is not None
        start = perf_counter()
        qspn, sel = run_qspn_func(qspn, args, sql)
        delta_time = perf_counter() - start
        seq_time += delta_time
        total_time += delta_time
        if sql[0] == 'SELECT':
            est = max(1, round(sel * len(data_df)))
            tru = true_card[len(est_card)]
            seq_q_errs.append(max(est / tru, tru / est))
            est_card.append(est)
        interval_sql_num += 1
    #last seq
    if interval_sql_num > 0:
        if update_qspn_func is not None:
            interval_running_queries = np.array(interval_running_queries)
            #print('data_df:', data_df.shape)
            interval_update_data_array = np.array(interval_update_data_list)
            if interval_update_data_array.shape != (len(interval_update_data_list), len(data_csv.columns)):
                interval_update_data_array = np.zeros(shape=(len(interval_update_data_list), len(data_csv.columns)), dtype=int)
            interval_update_data_csv = pd.DataFrame({c: interval_update_data_array[:, i] for i, c in enumerate(data_csv.columns)})
            print('Update: {} tuples, {} queries'.format(len(interval_update_data_csv), len(interval_running_queries)))
            start = perf_counter()
            qspn = update_qspn_func(qspn, args, cluster, last_update_data, last_update_workload, interval_update_data_csv, interval_running_queries)
            delta_time = perf_counter() - start
            seq_time += delta_time
            total_time += delta_time
            #print(last_update_data)
            last_update_data = pd.concat([last_update_data, interval_update_data_csv], axis=0)
            last_update_workload = np.concatenate((last_update_workload, interval_running_queries), axis=0)
            print('last_update_data:', last_update_data.shape)
            print('last_update_workload:', last_update_workload.shape)
            #print(last_update_data)
            print()
            #exit(-1)
            interval_update_data_csv = pd.DataFrame({i: [] for i in data_csv.columns})
            interval_running_queries = []
        with open('qspn_update_expr.log', 'a') as dumplog:
            maxerr = max(seq_q_errs)
            sumerr = sum(seq_q_errs)
            smooth_meanerr = max(1.0, (sumerr - maxerr) / max(1, len(seq_q_errs)-1))
            dumplog.write('Seq Time={} min., |Q|={}, Seq Mean Q-err={}, Seq smooth Mean Q-err(NoMAX)={}\n'.format(seq_time / 60, len(seq_q_errs), np.mean(seq_q_errs) if len(seq_q_errs) > 0 else 0, smooth_meanerr))
        seq_time = 0
        seq_q_errs = []
        interval_sql_num = 0
    # print('Total Time = {} min'.format(total_time / 60))
    # errors = np.maximum(np.divide(est_card, true_card), np.divide(true_card, est_card))
    # print()
    # print("Q-Error:")
    # print(list(errors))
    # print("Q-Error distributions are:")
    # for n in [50, 90, 95, 99, 100]:
    #     print(f"{n}% percentile:", np.percentile(errors, n))
    # print('Mean: {}'.format(np.mean(errors)))
    # print('smooth Mean (NoMAX): {}'.format((sum(errors) - np.percentile(errors, 100)) / (len(errors)-1)))
    # with open('qspn_update_expr.log', 'a') as dumplog:
    #     dumplog.write('Total Time = {} min\n'.format(total_time / 60))
    #     dumplog.write('Q-Error:\n')
    #     dumplog.write('[{}]\n'.format(','.join(list(map(str, list(errors))))))
    #     dumplog.write("Q-Error distributions are:\n")
    #     for n in [50, 90, 95, 99, 100]:
    #         dumplog.write("{}% percentile:{}\n".format(n, np.percentile(errors, n)))
    #     dumplog.write('Mean: {}\n'.format(np.mean(errors)))
    pickle.dump(qspn, open('update_eval_tmp.pkl', "wb"), pickle.HIGHEST_PROTOCOL)
    upded_model_size = os.path.getsize('update_eval_tmp.pkl')
    print(f"Updated Model Size: {upded_model_size/1000} KB")
    with open('qspn_update_expr.log', 'a') as dumplog:
        dumplog.write('Updated Model Size: {} KB\n\n'.format(upded_model_size/1000))
    