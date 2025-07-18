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
import csv

sys.path.append(os.getcwd())

import settings
sys.path.append(str(settings.PROJECT_PATH / 'qspn'))
sys.path.append(str(settings.PROJECT_PATH / 'scripts'))

from Learning.qspnJoinLearning import learn_multi_QSPN
from Learning.qspnJoinReader import multi_table_workload_csv_reader
from Learning.qspnJoinInference import mqspn_probability, DETAIL_PERF
from Learning.qspnJoinReader import multi_table_workload_csv_reader, multi_table_dataset_csv_reader, workload_data_columns_stats
from Learning.qspnJoinBase import SHOW_VE, set_FJBuckets_K

DEBUG_ERR = False
DETAIL_PERF = False

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time


dataset_root = settings.DATA_ROOT
model_save_root = os.path.join(settings.MODEL_ROOT, "multi_tables")

def mqspn_train(dataset_name, workload_train_name, attr=(None,)):
    global dataset_root, model_save_root
    workload_name = workload_train_name + '.csv'
    workload, true_card = multi_table_workload_csv_reader(os.path.join(dataset_root, dataset_name, 'queries', workload_name))
    assert len(workload) == len(true_card)
    set_FJBuckets_K(attr[0])
    mqspn, train_time = learn_multi_QSPN(os.path.join(dataset_root, dataset_name), workload)
    print('----------------------------------------------------------------------------------------------------------------')
    print("Train Time: {} min".format(train_time / 60))
    pickle.dump(mqspn, open(os.path.join(model_save_root, 'mqspn', 'mqspn_{}_{}_{}.pkl'.format(dataset_name, workload_train_name, attr[0])), 'wb'), pickle.HIGHEST_PROTOCOL)

def mqspn_test(dataset_name, workload_train_name, workload_test_name, attr=(None,)):
    global dataset_root, model_save_root
    model_path = os.path.join(model_save_root, 'mqspn', 'mqspn_{}_{}_{}.pkl'.format(dataset_name, workload_train_name, attr[0]))
    set_FJBuckets_K(attr[0])
    model_size = os.path.getsize(model_path)
    print('Loading {}'.format(model_path))
    print(f"Model Size: {model_size/1000/1000} MB")
    with open(model_path, 'rb') as f:
        mqspn = pickle.load(f)
    # for i in mqspn.table_columns:
    #     print(i)
    #     print('\t', mqspn.table_columns[i])
    #     print('\t', mqspn.table_domain[i])
    #     print('\t', mqspn.table_cardinality[i])
    #     print('\t', mqspn.table_rdc_adjacency_matrix[i])
    #     print('\t', mqspn.table_qspn_model[i], mqspn.table_qspn_model[i].scope, mqspn.table_qspn_model[i].children)
    #     print()
    #exit(-1)
    #queries_name = 'mscn_queries_neurocard_format.csv'
    #queries, true_cards = multi_table_workload_csv_reader(os.path.join(dataset_root, dataset_name, 'queries', queries_name))
    #dc, join_graph = workload_data_columns_stats(queries)
    #print("Trainset")
    #print(dc)
    #print(join_graph)
    queries_name = workload_test_name + '.csv'
    queries, true_cards = multi_table_workload_csv_reader(os.path.join(dataset_root, dataset_name, 'queries', queries_name))
    #exit(-1)
    #print(queries)
    #exit(-1)
    #dc, join_graph = workload_data_columns_stats(queries)
    #print("Testset")
    #print(dc)
    #print(join_graph)
    #exit(-1)
    #queries = queries[0:1000]
    #true_cards = true_cards[0:1000]
    #queries = queries[0:15]
    #true_cards = true_cards[0:15]
    #queries = queries[0:100]
    #true_cards = true_cards[0:100]
    #queries = queries[66 : 67]
    #true_cards = true_cards[66 : 67]
    #queries = queries[0 : 1] + queries[66 : 67]
    #true_cards = true_cards[0 : 1] + true_cards[66 : 67]
    #queries = [queries[i-1] for i in [1, 20, 21, 22, 35, 63]]
    #true_cards = [true_cards[i-1] for i in [1, 20, 21, 22, 35, 63]]
    #queries = [queries[i-1] for i in [56, 57, 67, 70]]
    #true_cards = [true_cards[i-1] for i in [56, 57, 67, 70]]
    #queries = [queries[i-1] for i in [56, 57, 62, 66, 67]]
    #true_cards = [true_cards[i-1] for i in [56, 57, 62, 66, 67]]
    #queries = [queries[i-1] for i in [50, 51, 52, 63, 64]]
    #true_cards = [true_cards[i-1] for i in [50, 51, 52, 63, 64]]
    #queries = [queries[i-1] for i in [55, 56, 57, 58]]
    #true_cards = [true_cards[i-1] for i in [55, 56, 57, 58]]
    #queries = [queries[-1]]
    #true_cards = [true_cards[-1]]
    #print(queries)
    #print(true_cards)
    #exit(-1)
    assert len(queries) == len(true_cards)
    qerrs = []
    total_time = 0
    if DEBUG_ERR:
        debug_err_output_header = ['query', 'est_card', 'GT', 'q-err']
        debug_err_output = []
    if DETAIL_PERF:
        de_perf_info = {'qspn_prune': [], 'merge_buckets': [], 'pre_ve': [], 've': []}
    for qth, q in enumerate(queries):
        gt = true_cards[qth]
        #print(q)
        qf = {'join': ['='.join(i) for i in q[1]], 'select': [('.'.join(i[0 : 2]), i[2], i[3]) for i in q[2]]}
        if SHOW_VE:
            print(qf, gt)
        #exit(-1)
        ce_func = mqspn_probability
        tic = perf_counter()
        if DETAIL_PERF:
            est_card, perf_qspn_prune, perf_merge_buckets, perf_pre_ve, perf_ve = ce_func(mqspn, qf, attr)
            de_perf_info['qspn_prune'].append(perf_qspn_prune)
            de_perf_info['merge_buckets'].append(perf_merge_buckets)
            de_perf_info['pre_ve'].append(perf_pre_ve)
            de_perf_info['ve'].append(perf_ve)
            est_card = max(1, est_card)
        else:
            est_card = max(1, ce_func(mqspn, qf, attr))
        #print(est_card)
        #print(qf, est_card)
        total_time += perf_counter() - tic
        if SHOW_VE:
            print(est_card, 'GT:', gt)
        tru_card = max(1, gt)
        qerr = max(est_card/tru_card, tru_card/est_card)
        if DEBUG_ERR:
            debug_err_output.append([qth+1, est_card, tru_card, qerr])
        qerrs.append(qerr)
    qerrs = np.array(qerrs)
    print(f"querying {len(qerrs)} queries takes {total_time} secs (avg. {total_time*1000/len(qerrs)} ms per query)")
    if DETAIL_PERF:
        de_perf_info = {ik: np.mean(iv) for ik, iv in de_perf_info.items()}
        for ik, iv in de_perf_info.items():
            print('{} = {}ms'.format(ik, iv))
    print('----------------------------------------------------------------------------------------------------------------')
    print("Q-Error distributions are:")
    for nth in [50, 90, 95, 99, 100]:
        print(f"{nth}% percentile:", np.percentile(qerrs, nth))
    print('Mean: {}'.format(np.mean(qerrs)))
    if DEBUG_ERR:
        with open('debug_join_err_output.csv', 'w', encoding='utf-8') as dump_debug_err:
            writer = csv.writer(dump_debug_err)
            writer.writerow(debug_err_output_header)
            writer.writerows(debug_err_output)
    #mqspn, train_time = learn_multi_QSPN(os.path.join(dataset_root, dataset_name), workload)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--dataset', type=str, default='job')
    parser.add_argument('--workload-trainset', type=str, default='mscn_queries_neurocard_format')
    parser.add_argument('--workload-testset', type=str, default='job-light')
    parser.add_argument('--model-binning-size', type=int, default=211)
    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset
    workload_train_name = args.workload_trainset
    workload_test_name = args.workload_testset
    model_binning_size = args.model_binning_size

    if args.inference:
        mqspn_test(dataset_name, workload_train_name, workload_test_name, (model_binning_size,))
    elif args.train:
        mqspn_train(dataset_name, workload_train_name, (model_binning_size,))