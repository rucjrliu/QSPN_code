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
import csv

sys.path.append(os.getcwd())

import settings
sys.path.append(str(settings.PROJECT_PATH / 'qspn'))
sys.path.append(str(settings.PROJECT_PATH / 'scripts'))

from Learning.mqspnJoinLearningNew import learn_multi_QSPN
from Inference.mqspnJoinInferenceNew import load_multi_QSPN, CE_multi_QSPN_opt
from Structure.mqspnJoinReader import multi_table_workload_csv_reader_trainset, multi_table_workload_csv_reader_testset, multi_table_workload_extract, init_p
# from Learning.qspnJoinInference import mqspn_probability, DETAIL_PERF
# from Learning.qspnJoinBase import SHOW_VE, set_FJBuckets_K

DEBUG_ERR = False
DETAIL_PERF = False

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time


dataset_root = settings.DATA_ROOT
model_save_root = os.path.join(settings.MODEL_ROOT, "multi_tables")

def mqspn_train(dataset_name, workload_train_name, attr=(None,70)):
    global dataset_root, model_save_root
    workload_name = workload_train_name + '.csv'
    workload, true_card = multi_table_workload_csv_reader_trainset(os.path.join(dataset_root, dataset_name, 'queries', workload_name))
    for i in workload:
        print(i)
    print()
    # exit(-1)
    mqspn, train_time = learn_multi_QSPN(attr[0], os.path.join(dataset_root, dataset_name), workload, speval_top_num=attr[1])
    print('----------------------------------------------------------------------------------------------------------------')
    print("Train Time: {} min".format(train_time / 60))
    model_path = os.path.join(model_save_root, 'mqspn', 'mqspn_{}_{}_{}.pkl'.format(dataset_name, workload_train_name, attr[0]))
    pickle.dump(mqspn, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    model_size = os.path.getsize(model_path)
    print(f"Model Size: {model_size/1000/1000} MB")

def mqspn_test(dataset_name, workload_train_name, workload_test_name, attr=(None,70)):
    global dataset_root, model_save_root

    model_path = os.path.join(model_save_root, 'mqspn', 'mqspn_{}_{}_{}.pkl'.format(dataset_name, workload_train_name, attr[0]))
    model_size = os.path.getsize(model_path)
    print('Loading {}'.format(model_path))
    print(f"Model Size: {model_size/1000/1000} MB")
    mqspn = load_multi_QSPN(model_path)
    # mqspn._print(info_level=2)
    # exit(-1)

    workload_name = workload_test_name + '.csv'
    workload, true_cards = multi_table_workload_csv_reader_testset(os.path.join(dataset_root, dataset_name, 'queries', workload_name))
    workload = multi_table_workload_extract(workload, mqspn)
    est_cards = []
    total_time = 0
    for q, gt in zip(workload, true_cards):
        # print(q, gt)
        tic = perf_counter()
        ce = max(1, round(CE_multi_QSPN_opt(q, mqspn)))
        total_time += perf_counter() - tic
        est_cards.append(ce)
        # print(ce)
        # print()
        #exit(-1)
        # input()
    print(f"querying {len(true_cards)} queries takes {total_time} secs (avg. {total_time*1000/len(true_cards)} ms per query)")
    errors = np.maximum(np.divide(est_cards, true_cards), np.divide(true_cards, est_cards))
    print('----------------------------------------------------------------------------------------------------------------')
    prt_info_df = {'Est. Card': est_cards, 'True Card': true_cards, 'Q-error': errors}
    prt_info_df = pd.DataFrame(prt_info_df)
    pd.set_option('display.max_rows', None)
    print(prt_info_df)
    print('----------------------------------------------------------------------------------------------------------------')
    print("Q-Error distributions are:")
    #dumplog.write("Q-Error distributions are:\n")
    for n in [50, 90, 95, 99, 100]:
        print(f"{n}% percentile:", np.percentile(errors, n))
        #dumplog.write(f"{n}% percentile:\n".format(np.percentile(errors, n)))
    print('Mean: {}'.format(np.mean(errors)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    
    # parser.add_argument('--dataset', type=str, default='job')
    # parser.add_argument('--workload-trainset', type=str, default='mscn_queries_neurocard_format')
    # parser.add_argument('--workload-testset', type=str, default='job-light')

    parser.add_argument('--dataset', type=str, default='stats')
    parser.add_argument('--workload-trainset', type=str, default='stats_CEB_trainset')
    parser.add_argument('--workload-testset', type=str, default='stats_CEB_testset')
    parser.add_argument('--partition-width-limit', type=int, default=0)
    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset
    workload_train_name = args.workload_trainset
    workload_test_name = args.workload_testset
    partition_width_limit = args.partition_width_limit if args.partition_width_limit > 0 else None

    if args.inference:
        mqspn_test(dataset_name, workload_train_name, workload_test_name, init_p(dataset_name, partition_width_limit))
    elif args.train:
        mqspn_train(dataset_name, workload_train_name, init_p(dataset_name, partition_width_limit))