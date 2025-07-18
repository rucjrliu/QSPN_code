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

sys.path.append(os.getcwd())

import settings
sys.path.append(str(settings.PROJECT_PATH / 'qspn'))
sys.path.append(str(settings.PROJECT_PATH / 'scripts'))

from Learning.learningWrapper import learn_FSPN
from Structure.leaves.parametric.Parametric import Categorical
from Structure.model import FSPN
from Structure.nodes import Context
from Learning.statistics import get_structure_stats
from Learning.updateQSPN import update_QSPN
import Learning.splitting.Workload as qspnwkld

from update_eval import update_1by1, run_qspn, retrain_qspn, run_qspn_adaptive_incremental

try:
    from time import perf_counter 
except:
    from time import time
    perf_counter = time

def divide_list(x: list, d: int):
    assert d > 0
    return [x[i*len(x)//min(d,len(x)) : min((i+1)*len(x)//min(d,len(x)),len(x))] for i in range(min(d,len(x)))]

def divide_df(x, d: int):
    assert d > 0
    return [x.iloc[i*len(x)//min(d,len(x)) : min((i+1)*len(x)//min(d,len(x)),len(x))] for i in range(min(d,len(x)))]

def divide_ndarr(x, d: int):
    assert d > 0
    return [x[i*len(x)//min(d,len(x)) : min((i+1)*len(x)//min(d,len(x)),len(x))] for i in range(min(d,len(x)))]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--dataset', type=str, default='gas')
    parser.add_argument('--skew', type=float, default=0.5)
    parser.add_argument('--corr', type=float, default=0.5)
    parser.add_argument('--query-root', type=str, default='template')
    parser.add_argument('--query-path', type=str, required=False)
    parser.add_argument('--model-path', type=str, required=False)
    # FSPN / SPN parameters
    parser.add_argument('--model', type=str, default='qspn')
    parser.add_argument('--multi-leaf', action='store_true')
    parser.add_argument('--rdc-threshold', type=float, default="0.3")
    parser.add_argument('--rdc-strong-connection-threshold', type=float, default="0.7")
    # QSPN parameters
    parser.add_argument('--Nx', type=float, default=5.0)
    parser.add_argument('--lower', type=float, default=0.1)
    parser.add_argument('--upper', type=float, default=0.3)
    parser.add_argument('--qsplit', action='store_true', default=True)
    parser.add_argument('--qdcorr', default=None)
    parser.add_argument('--detail', action='store_true')
    # Inference parameters
    parser.add_argument('--test-query', type=str, required=False)
    # Update parameters
    parser.add_argument('--update-data', type=str, required=False)
    parser.add_argument('--update-meta', type=str, required=False)
    parser.add_argument('--update-query-root', type=str, default='template')
    parser.add_argument('--update-query-path', type=str, required=False)
    parser.add_argument('--update-skew', type=float, default=None)
    parser.add_argument('--update-corr', type=float, default=None)
    parser.add_argument('--update-method', type=str, default='notrain')
    args = parser.parse_args()

    print(args)
    #exit(-1)

    data_path = settings.DATA_ROOT / args.dataset/ "data.csv"
    data_path_df = settings.DATA_ROOT / args.dataset/ "data.hdf"
    meta_path = settings.DATA_ROOT / args.dataset/ "meta.json"
    query_root = settings.DATA_ROOT / args.dataset/ "queries" /args.query_root
    update_data_path = None
    update_data_path_df = None
    if args.update_data is not None:
        update_data_path_df = settings.DATA_ROOT / args.dataset/ (args.update_data + ".hdf")
        update_data_path = settings.DATA_ROOT / args.dataset/ (args.update_data + ".csv")

    print(data_path)
    print('upd:', update_data_path_df)
    print('upd:', update_data_path)
    print(meta_path)
    print(query_root)

    data = pd.read_csv(str(data_path), header=None)
    data = data.drop(0)
    with open(str(meta_path)) as f:
        meta = json.load(f)
    columns = meta['columns']
    cardinality = meta['cardinality']
    print(f"data shape: {data.shape}")
    print(f"Cardinality: {cardinality}, Columns: ", columns)

    #print(columns)
    #print(type(columns))
    #exit(-1)

    MODEL_PATH = settings.MODEL_ROOT / "single_tables"
    if args.model == 'qspn':
        MODEL_PATH = MODEL_PATH / "qspn"
    #print(MODEL_PATH)
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    model_prefix = args.model
    if args.model == 'spn' or args.model == 'fspn':
        if args.multi_leaf:
            model_prefix = model_prefix + "_Multi"
        model_prefix = model_prefix + f"_rdc{args.rdc_threshold}"
    else:
        model_prefix = model_prefix + f"_{args.Nx}_{args.lower}_{args.upper}"
        if args.qsplit:
            model_prefix = model_prefix + "_QSplit"
        if args.multi_leaf:
            model_prefix = model_prefix + "_Multi"
        if args.qdcorr:
            model_prefix = model_prefix + "_qdcorr" + args.qdcorr
    model_prefix = model_prefix + f"_{args.dataset}"
    #print(model_prefix)

    if args.query_path:
        query_path = str(settings.DATA_ROOT / args.dataset/ "queries" /args.query_path)
        model_save = str(MODEL_PATH / f"{model_prefix}_{args.query_path}.pkl")
    if args.skew is not None and args.corr is not None:
        query_path = str(query_root / f"query_{args.skew}_{args.corr}_0.0")
        model_save = str(MODEL_PATH / f"{model_prefix}_{args.query_root}_{args.skew}_{args.corr}.pkl")
    if args.update_query_path is not None:
        update_query_path = str(settings.DATA_ROOT / args.dataset/ "queries" /args.update_query_path)
    if args.update_query_root is not None and args.update_skew is not None and args.update_corr is not None:
        print(type(args.update_query_root))
        update_query_path = os.path.join(settings.DATA_ROOT, args.dataset, "queries", args.update_query_root, f"query_{args.update_skew}_{args.update_corr}_0.0")
        model_save = str(MODEL_PATH / f"{model_prefix}_{args.query_root}_{args.skew}_{args.corr}.pkl")        
    if args.model_path is not None:
        model_save = str(MODEL_PATH / f"{args.model_path}.pkl")
    #print(update_query_path)

    print(model_save)
    print(query_path)

    #exit(-1)
    if args.update:
        if args.qsplit:
            cluster = 'kmeans'
        else:
            cluster = None
        with open('qspn_update_expr.log', 'a') as dumplog:
            dumplog.write('({})({})({})({}):\n'.format(data_path, update_query_path, model_save, args.update_method))
        if args.update_method == 'notrain':
            update_1by1(run_qspn_func=run_qspn, update_qspn_func=None, args=args, cluster=cluster, model_save=model_save, MODEL_PATH=MODEL_PATH, query_path=query_path, update_query_path=update_query_path, model_prefix=model_prefix, data_path=data_path, data_path_df=data_path_df, columns=columns)
        elif args.update_method == 'retrain':
            update_1by1(run_qspn_func=run_qspn, update_qspn_func=retrain_qspn, args=args, cluster=cluster, model_save=model_save, MODEL_PATH=MODEL_PATH, query_path=query_path, update_query_path=update_query_path, model_prefix=model_prefix, data_path=data_path, data_path_df=data_path_df, columns=columns)
        elif args.update_method == 'adaincr':
            update_1by1(run_qspn_func=run_qspn_adaptive_incremental, update_qspn_func=None, args=args, cluster=cluster, model_save=model_save, MODEL_PATH=MODEL_PATH, query_path=query_path, update_query_path=update_query_path, model_prefix=model_prefix, data_path=data_path, data_path_df=data_path_df, columns=columns)
        exit(0)

    if args.train:
        print('Train {} on data:{} workload:{}_{}'.format(args.model, args.dataset, args.corr, args.skew))
        sample_data = data.values.astype(int)
        #print(type(data), data.shape)
        #print(type(sample_data), sample_data.shape)
        #print(sample_data[-1])
        #exit(-1)
        parametric_types = [Categorical for i in range(len(data.columns))]
        ds_context = Context(parametric_types=parametric_types).add_domains(sample_data)

        if args.model == 'qspn':
            workload = np.load(os.path.join(query_path, "train_query_sc.npy"))
            true_card = np.load(os.path.join(query_path, "train_true_sc.npy"))
            print("Train QSPN, workload size: ", workload.shape)
            print(f"Model saved in {model_save}")
            #exit(-1)
            if args.qsplit:
                cluster = 'kmeans'
                #qspnwkld.MAXCUT_K=2
                #print(qspnwkld.MAXCUT_K)
            else:
                cluster = None
            start = perf_counter()
            model = learn_FSPN(
                sample_data,
                ds_context,
                workload=workload,
                queries=cluster,
                rdc_sample_size=100000,
                rdc_strong_connection_threshold=1.1,
                multivariate_leaf=args.multi_leaf,
                threshold=0.3,
                wkld_attr_threshold=0.01,
                wkld_attr_bound=(args.Nx, args.lower, args.upper),
                qdcorr=args.qdcorr
            )
            train_time = perf_counter() - start
            print(f"Model saved in {model_save}")
            pickle.dump(model, open(model_save, "wb"), pickle.HIGHEST_PROTOCOL)

        elif args.model == 'spn':
            model_save_path = str(MODEL_PATH / f"{model_prefix}.pkl")
            print(f"Train SPN, model saved in {model_save_path}")
            start = perf_counter()
            model = learn_FSPN(
                sample_data,
                ds_context,
                workload=None,
                rdc_sample_size=100000,
                rdc_strong_connection_threshold=1.1,
                multivariate_leaf=args.multi_leaf,
                threshold=args.rdc_threshold, 
                wkld_attr_bound=None
            )
            train_time = perf_counter() - start
            pickle.dump(model, open(model_save_path, "wb"), pickle.HIGHEST_PROTOCOL)

        elif args.model == 'fspn':
            model_save_path = str(MODEL_PATH / f"{model_prefix}.pkl")
            print(f"Train FSPN, model saved in {model_save_path}")
            start = perf_counter()
            model = learn_FSPN(
                sample_data,
                ds_context,
                workload=None,
                rdc_sample_size=100000,
                rdc_strong_connection_threshold=args.rdc_strong_connection_threshold,
                multivariate_leaf=args.multi_leaf,
                threshold=args.rdc_threshold, 
                wkld_attr_bound=None
            )
            train_time = perf_counter() - start
            pickle.dump(model, open(model_save_path, "wb"), pickle.HIGHEST_PROTOCOL)
        print('----------------------------------------------------------------------------------------------------------------')
        print('Train Time: {} min.'.format(train_time / 60))

    if args.inference:
        results = []
        workload = np.load(os.path.join(query_path, "test_query_sc.npy"))
        true_card = np.load(os.path.join(query_path, "test_true_sc.npy"))

        if args.test_query is not None:
            test_query = os.path.join(query_root, args.test_query)
            print("Test Query: ", args.test_query)
            workload = np.load(os.path.join(test_query, "test_query_sc.npy"))
            true_card = np.load(os.path.join(test_query, "test_true_sc.npy"))

        workload = workload[true_card!=0,:]
        true_card = true_card[true_card!=0]
        print("Inference, workload size: ", workload.shape)

        if args.model == 'qspn' or args.model_path is not None:
            model_save_path = model_save
        else:
            model_save_path = str(MODEL_PATH / f"{model_prefix}.pkl")
        print('model: {}'.format(model_save_path))
        
        #exit(-1)

        model_size = os.path.getsize(model_save_path)
        print(f"Model Size: {model_size/1000} KB")

        with open(model_save_path, 'rb') as f:
            spn = pickle.load(f)
        print(get_structure_stats(spn))

        model = FSPN()
        #print(model.scope, model.range)
        #exit(-1)
        model.model = spn 
        model.store_factorize_as_dict()

        est_card = []
        if args.model == 'qspn':
            
            #est_card = model.probability((workload[:,:,0],workload[:,:,1]), calculated=dict()) * cardinality
            #total_time = perf_counter() - tic 
            total_time = 0
            #tic = perf_counter()
            for wi in range(workload.shape[0]):
                tic = perf_counter()
                # e = max(1, round(model.probability((workload[wi,:,0].reshape(1,-1), workload[wi,:,1].reshape(1,-1)),
                #                         calculated=dict()) * cardinality))
                e = [max(1, round(ce)) for ce in model.probability((workload[wi,:,0].reshape(1,-1), workload[wi,:,1].reshape(1,-1)),
                                        calculated=dict(), exist_qsum=True, first_time_recur=True) * cardinality]
                total_time += perf_counter() - tic
                #print(wi, e, true_card[wi])
                #exit(-1)
                est_card.append(e[0])
            #total_time = perf_counter() - tic
            est_card = np.array(est_card)
        elif args.model == 'fspn':
            #print(args.model)
            #est_card = model.probability((workload[:,:,0],workload[:,:,1]), calculated=dict()) * cardinality
            #total_time = perf_counter() - tic 
            #est_card = []
            total_time = 0
            for wi in range(workload.shape[0]):
                tic = perf_counter()
                # e = model.probability((workload[wi,:,0].reshape(1,-1), workload[wi,:,1].reshape(1,-1)),
                #                         calculated=dict(), exist_qsum=False) * cardinality
                e = [max(1, round(ce)) for ce in model.probability((workload[wi,:,0].reshape(1,-1), workload[wi,:,1].reshape(1,-1)),
                                        calculated=dict(), exist_qsum=False, first_time_recur=True) * cardinality]
                total_time += perf_counter() - tic
                est_card.append(e[0])
            est_card = np.array(est_card)
        else:
            #print(args.model)
            #est_card = model.probability((workload[:,:,0],workload[:,:,1]), calculated=dict()) * cardinality
            #total_time = perf_counter() - tic 
            #est_card = []
            total_time = 0
            for wi in range(workload.shape[0]):
                tic = perf_counter()
                # e = model.probability((workload[wi,:,0].reshape(1,-1), workload[wi,:,1].reshape(1,-1)),
                #                         calculated=dict(), exist_qsum=False) * cardinality
                e = [max(1, round(ce)) for ce in model.probability((workload[wi,:,0].reshape(1,-1), workload[wi,:,1].reshape(1,-1)),
                                        calculated=dict(), exist_qsum=False, first_time_recur=True) * cardinality]
                total_time += perf_counter() - tic
                est_card.append(e[0])
            est_card = np.array(est_card)
        # for i in est_card:
        #     print(i)
        print(f"querying {len(true_card)} queries takes {total_time} secs (avg. {total_time*1000/len(true_card)} ms per query)")
        errors = np.maximum(np.divide(est_card, true_card), np.divide(true_card, est_card))
        if args.detail:
            print('----------------------------------------------------------------------------------------------------------------')
            prt_info_df = {'Est. Card': est_card, 'True Card': true_card, 'Q-error': errors}
            prt_info_df = pd.DataFrame(prt_info_df)
            pd.set_option('display.max_rows', None)
            print(prt_info_df)

        # for ei, ee in enumerate(errors):
        #     if ee > 10:
        #         print('accu.={}, est.={}, q-err={}'.format(true_card[ei], est_card[ei], ee))

        est_card_zero = est_card[est_card==0]
        #print(est_card_zero)
        #print(true_card)
        #print('Q-Error:', list(errors))
        print('----------------------------------------------------------------------------------------------------------------')
        print("Q-Error distributions are:")
        #dumplog.write("Q-Error distributions are:\n")
        for n in [50, 90, 95, 99, 100]:
            print(f"{n}% percentile:", np.percentile(errors, n))
            #dumplog.write(f"{n}% percentile:\n".format(np.percentile(errors, n)))
        print('Mean: {}'.format(np.mean(errors)))
        #dumplog.write('Mean: {}\n'.format(np.mean(errors)))
        #dumplog.write('\n')
        #dumplog.close()

        results.append([
            query_path,
            np.percentile(errors, 50),
            np.percentile(errors, 90),
            np.percentile(errors, 95),
            np.percentile(errors, 99),
            np.percentile(errors, 100),
            total_time,
            total_time*1000/len(true_card),
            model_size/1000
        ])
        
        # results = pd.DataFrame(results, columns=["workload", 
        #                                          "50th",
        #                                          "90th",
        #                                          "95th",
        #                                          "99th",
        #                                          "Max",
        #                                          "Total Time",
        #                                          "Avg Time",
        #                                          "Model Size"])
        # results.to_csv(os.path.join(settings.RESULT_PATH, f'{args.}{args.model}_results.csv'), index=None)

    update_intervals = 10
    #update_intervals = -1 means one by one
    if args.update and update_intervals > 1:
        #input origin_data, origin_workload, origin_cardinality
        if args.model == 'qspn' or args.model_path is not None:
            model_save_path = model_save
        else:
            model_save_path = str(MODEL_PATH / f"{model_prefix}.pkl")
        print('original model: {}'.format(model_save_path))
        model_size = os.path.getsize(model_save_path)
        print(f"Model Size: {model_size/1000} KB")

        with open(model_save_path, 'rb') as f:
            spn = pickle.load(f)
        print(get_structure_stats(spn))

        model = FSPN()
        model.model = spn 
        model.store_factorize_as_dict()

        origin_workload = np.load(os.path.join(query_path, "train_query_sc.npy"))
        origin_true_card = np.load(os.path.join(query_path, "train_true_sc.npy"))
        print(f"origin_workload shape: {origin_workload.shape}")
        print(f"origin_true_card shape: {origin_true_card.shape}")
        print(type(origin_workload), type(origin_true_card))

        origin_data = pd.read_csv(str(data_path), header=None)
        origin_data = origin_data.drop(0)
        origin_cardinality = origin_data.shape[0]
        print(f"origin_data shape: {origin_data.shape}")
        print(f"origin_cardinality: {origin_cardinality}")
        print(type(origin_data))
        #input new_data (.csv)
        new_data = None
        if update_data_path is not None:
            new_data = pd.read_csv(str(update_data_path), header=None)
            new_data_columns = {c: i for i, c in enumerate(list(new_data.iloc[0]))}
            print(new_data_columns)
            print(columns)
            #exit(-1)
            assert new_data_columns == columns
            new_data = new_data.drop(0)
            new_cardinality = new_data.shape[0]
            print(f"new_data shape: {new_data.shape}")
            print(f"new_cardinality: {new_cardinality}, new_data_columns: ", new_data_columns)
            print(type(new_data))
        #input new_data_df (.hdf)
        new_data_df = None
        if update_data_path_df is not None:
            new_data_df = pd.read_hdf(update_data_path_df, key="dataframe") #.hdf
            new_data_columns = {c: i for i, c in enumerate(new_data_df.columns)}
            print(new_data_columns)
            print(columns)
            assert new_data_columns == columns
            new_cardinality = new_data_df.shape[0]
            print(f"new_data shape: {new_data_df.shape}")
            print(f"new_cardinality: {new_cardinality}")
            print(type(new_data_df))
        #input new_queries
        if args.update_skew is not None and args.update_corr is not None:
            new_query_path = str(query_root / f"query_{args.update_skew}_{args.update_corr}_0.0")
            new_workload = np.load(os.path.join(new_query_path, "train_query_sc.npy"))
            new_true_card = np.load(os.path.join(new_query_path, "train_true_sc.npy"))
            new_queries = np.load(os.path.join(new_query_path, "test_query_sc.npy"))
            new_queries_true_card = np.load(os.path.join(new_query_path, "test_true_sc.npy"))
            print(f"new workload: {new_query_path}")
            print(f"new_queries shape: {new_queries.shape}")
            print(f"new_queries_true_card shape: {new_queries_true_card.shape}")
            print(type(new_queries), type(new_queries_true_card))
        
        #divide new_data_df and new_queries into update_intervals pieces
        new_data_list = divide_df(new_data, update_intervals)
        new_data_df_list = divide_df(new_data_df, update_intervals)
        new_queries_list = divide_ndarr(new_queries, update_intervals)
        new_queries_true_card_list = divide_ndarr(new_queries_true_card, update_intervals)
        updated_cardinality = origin_cardinality
        updated_workload = origin_workload
        update_data_df = None
        last_update_data = None
        new_model_save = str(f"{model_save}_update_{args.update_data}_{args.query_root}_{args.update_skew}_{args.update_corr}.pkl")
        rebuild_model_save = str(f"{model_save}_rebuild_{args.update_data}_{args.query_root}_{args.update_skew}_{args.update_corr}.pkl")
        upd_qspn = deepcopy(spn)
        print(f"updateQSPN model will be saved in {new_model_save}")
        print(f"baseline rebuild QSPN model will be saved in {rebuild_model_save}")
        print('To start Interval Update,')
        print('Press ENTER to continue...')
        input()
        
        #Each interval
        print()
        for inte in range(update_intervals):
            print('[Interval {}]:'.format(inte))
            #new_data_df_interval, new_queries_interval
            new_data_df_inte = new_data_df_list[inte]
            assert new_data_df_inte.shape[0] == new_data_list[inte].shape[0]
            new_queries_inte = new_queries_list[inte]
            new_queries_true_card_inte = new_queries_true_card_list[inte]
            assert new_queries_true_card_inte.shape[0] == new_queries_inte.shape[0]
            #merge to get updated_data_df_interval, calc new_queries_interval_true_card
            if update_data_df is None:
                last_update_data = origin_data
                update_data_df = new_data_df_inte
            else:
                last_update_data = pd.concat([last_update_data, new_data_list[inte - 1]], axis=0)
                update_data_df = pd.concat([update_data_df, new_data_df_inte], axis=0)
            print(f"Interval {inte} last_update_data shape: {last_update_data.shape}")
            print(f"Interval {inte} update_data shape: {update_data_df.shape}")
            updated_cardinality += new_data_df_inte.shape[0]
            print(f"updated_cardinality: {updated_cardinality}")
            print(type(update_data_df))
            print(f"Interval {inte}: new_queries shape: {new_queries_inte.shape}")
            if update_data_df is not None:
                print('Calculating test_true_sc on upded_data...')
                new_queries_update_data_true_card_inte = np.full(new_queries_inte.shape[0], -1)
                for i in tqdm(range(new_queries_inte.shape[0])):
                    pred = ['{}<={}<={}'.format(new_queries_inte[i, j, 0], c, new_queries_inte[i, j, 1]) for j, c in enumerate(new_data_columns)]
                    preds = ' and '.join(pred)
                    new_queries_update_data_true_card_inte[i] = len(update_data_df.query(preds))
                    new_queries_true_card_inte[i] += new_queries_update_data_true_card_inte[i]
            print(f"Interval {inte}: new_queries_true_card shape: {new_queries_true_card_inte.shape}")
            print(type(new_queries_inte), type(new_queries_true_card_inte))
            print('Interval {}: new_data, update_data and new_queries are OK.'.format(inte))
            print('Press ENTER to continue...')
            input()
            #Baseline: Whole rebuild
            if args.qsum:
                cluster = 'kmeans'
            else:
                cluster = None
            rebuild_updated_data = pd.concat([last_update_data, new_data_list[inte]], axis=0)
            rebuild_updated_data = rebuild_updated_data.values.astype(int)
            assert rebuild_updated_data.shape[0] == updated_cardinality
            assert rebuild_updated_data.shape[1] == len(origin_data.columns)
            last_update_workload = deepcopy(updated_workload)
            if new_queries_inte is not None and len(new_queries_inte) > 0:
                updated_workload = np.concatenate((updated_workload, new_queries_inte), axis=0)
            assert updated_workload.shape[0] == last_update_workload.shape[0] + new_queries_inte.shape[0]
            assert updated_workload.shape[1] == origin_workload.shape[1]
            assert updated_workload.shape[2] == origin_workload.shape[2]
            rebuild_parametric_types = [Categorical for i in range(len(origin_data.columns))]
            rebuild_ds_context = Context(parametric_types=rebuild_parametric_types).add_domains(rebuild_updated_data)
            rebuild_qspn = learn_FSPN(
                rebuild_updated_data,
                rebuild_ds_context,
                workload=updated_workload,
                queries=cluster,
                rdc_sample_size=100000,
                rdc_strong_connection_threshold=1.1,
                multivariate_leaf=False,
                threshold=0.3,
                wkld_attr_threshold=0.01,
                wkld_attr_bound=(args.Nx, args.lower, args.upper)
            )
            print('Baseline Rebuild Finished.')
            print(f"Interval {inte}: Baseline Rebuild Model is saved to {rebuild_model_save}")
            pickle.dump(rebuild_qspn, open(rebuild_model_save, "wb"), pickle.HIGHEST_PROTOCOL)
            print('Press ENTER to Continue ...')
            input()
            #updateQSPN
            upd_qspn = update_QSPN(
                upd_qspn,
                data=last_update_data,
                workload=last_update_workload,
                data_insert=new_data_list[inte],
                data_delete = None,
                new_queries = new_queries_inte,
                rdc_sample_size=100000,
                rdc_threshold=0.3,
                wkld_attr_threshold=0.01,
                wkld_attr_bound=(args.Nx, args.lower, args.upper)
            )
            print('Update Finished.')
            print(f"Interval {inte}: updateQSPN Model is saved to {new_model_save}")
            pickle.dump(upd_qspn, open(new_model_save, "wb"), pickle.HIGHEST_PROTOCOL)
            print('Press ENTER to Continue ...')
            input()
            #check origin QSPN on new_queries_interval of updated_data_df_interval CE Accu, Inference Time, Size
            print("Inference origin model, workload size: ", new_queries_inte.shape)
            print(f"Model Path: {model_save}")
            model_size = os.path.getsize(model_save)
            print(f"Model Size: {model_size/1000} KB")
            model = FSPN()
            model.model = spn
            model.store_factorize_as_dict()
            est_origin_model_card = []
            tic = perf_counter()
            for wi in range(new_queries_inte.shape[0]):
                e = model.probability((new_queries_inte[wi,:,0].reshape(1,-1), new_queries_inte[wi,:,1].reshape(1,-1)),
                                        calculated=dict()) * updated_cardinality
                est_origin_model_card.append(e[0])
            total_origin_model_time = perf_counter() - tic 
            est_origin_model_card = np.array(est_origin_model_card)
            print('origin model qspn result:')
            print(f"querying {len(new_queries_true_card_inte)} queries takes {total_origin_model_time} secs (avg. {total_origin_model_time*1000/len(new_queries_true_card_inte)} ms per query)")
            print(new_queries_true_card_inte)
            print(est_origin_model_card)
            errors = np.maximum(np.divide(est_origin_model_card, new_queries_true_card_inte), np.divide(new_queries_true_card_inte, est_origin_model_card))
            print("Q-Error distributions are:")
            for n in [50, 90, 95, 99, 100]:
                print(f"{n}% percentile:", np.percentile(errors, n))
            print('Mean: {}'.format(np.mean(errors)))
            print('Interval {}: origin_model on update_data is tested.'.format(inte))
            print('Press ENTER to continue...')
            input()
            #check baseline rebuild QSPN on new_queries_interval of updated_data_df_interval CE Accu, Inference Time, Size
            print("Inference baseline rebuild QSPN model, workload size: ", new_queries_inte.shape)
            print(f"Model Path: {rebuild_model_save}")
            model_size = os.path.getsize(rebuild_model_save)
            print(f"Model Size: {model_size/1000} KB")
            model = FSPN()
            model.model = rebuild_qspn
            model.store_factorize_as_dict()
            est_origin_model_card = []
            tic = perf_counter()
            for wi in range(new_queries_inte.shape[0]):
                e = model.probability((new_queries_inte[wi,:,0].reshape(1,-1), new_queries_inte[wi,:,1].reshape(1,-1)),
                                        calculated=dict()) * updated_cardinality
                est_origin_model_card.append(e[0])
            total_origin_model_time = perf_counter() - tic 
            est_origin_model_card = np.array(est_origin_model_card)
            print('baseline rebuild qspn result:')
            print(f"querying {len(new_queries_true_card_inte)} queries takes {total_origin_model_time} secs (avg. {total_origin_model_time*1000/len(new_queries_true_card_inte)} ms per query)")
            print(new_queries_true_card_inte)
            print(est_origin_model_card)
            errors = np.maximum(np.divide(est_origin_model_card, new_queries_true_card_inte), np.divide(new_queries_true_card_inte, est_origin_model_card))
            print("Q-Error distributions are:")
            for n in [50, 90, 95, 99, 100]:
                print(f"{n}% percentile:", np.percentile(errors, n))
            print('Mean: {}'.format(np.mean(errors)))
            print('Interval {}: baseline_rebuild_model on update_data is tested.'.format(inte))
            print('Press ENTER to continue...')
            input()
            #check updateQSPN on new_queries_interval of updated_data_df_interval CE Accu, Inference Time, Size
            print("Inference updateQSPN model, workload size: ", new_queries_inte.shape)
            print(f"Model Path: {new_model_save}")
            model_size = os.path.getsize(new_model_save)
            print(f"Model Size: {model_size/1000} KB")
            model = FSPN()
            model.model = upd_qspn
            model.store_factorize_as_dict()
            est_origin_model_card = []
            tic = perf_counter()
            for wi in range(new_queries_inte.shape[0]):
                e = model.probability((new_queries_inte[wi,:,0].reshape(1,-1), new_queries_inte[wi,:,1].reshape(1,-1)),
                                        calculated=dict()) * updated_cardinality
                est_origin_model_card.append(e[0])
            total_origin_model_time = perf_counter() - tic 
            est_origin_model_card = np.array(est_origin_model_card)
            print('update qspn result:')
            print(f"querying {len(new_queries_true_card_inte)} queries takes {total_origin_model_time} secs (avg. {total_origin_model_time*1000/len(new_queries_true_card_inte)} ms per query)")
            print(new_queries_true_card_inte)
            print(est_origin_model_card)
            errors = np.maximum(np.divide(est_origin_model_card, new_queries_true_card_inte), np.divide(new_queries_true_card_inte, est_origin_model_card))
            print("Q-Error distributions are:")
            for n in [50, 90, 95, 99, 100]:
                print(f"{n}% percentile:", np.percentile(errors, n))
            print('Mean: {}'.format(np.mean(errors)))
            print('Interval {}: update_model on update_data is tested.'.format(inte))
            print('Press ENTER to continue...')
            input()
    
    if args.update and update_intervals <= 1:
        if args.model == 'qspn' or args.model_path is not None:
            model_save_path = model_save
        else:
            model_save_path = str(MODEL_PATH / f"{model_prefix}.pkl")
        print('original model: {}'.format(model_save_path))
        model_size = os.path.getsize(model_save_path)
        print(f"Model Size: {model_size/1000} KB")

        with open(model_save_path, 'rb') as f:
            spn = pickle.load(f)
        print(get_structure_stats(spn))

        model = FSPN()
        model.model = spn 
        model.store_factorize_as_dict()

        origin_workload = np.load(os.path.join(query_path, "train_query_sc.npy"))
        origin_true_card = np.load(os.path.join(query_path, "train_true_sc.npy"))
        
        origin_data = pd.read_csv(str(data_path), header=None)
        origin_data = origin_data.drop(0)
        origin_cardinality = origin_data.shape[0]
        print(f"origin_data shape: {origin_data.shape}")
        print(f"origin_cardinality: {origin_cardinality}")
        print(type(origin_data))
        
        
        new_data_df = None
        if update_data_path_df is not None:
            #new_data = pd.read_csv(str(update_data_path), header=None)
            new_data_df = pd.read_hdf(update_data_path_df, key="dataframe") #.hdf
            new_data_columns = {c: i for i, c in enumerate(new_data_df.columns)}
            print(new_data_columns)
            print(columns)
            #exit(-1)
            assert new_data_columns == columns
            #new_data = new_data.drop(0)
            new_cardinality = new_data_df.shape[0]
            #print(f"new_data shape: {new_data_df.shape}")
            #print(f"new_cardinality: {new_cardinality}, new_data_columns: ", new_data_columns)
            #print(type(new_data_df))
            #print(new_data_df)
            #new_data_df_list = divide_df(new_data_df, 10)
            #xx = new_data_df_list[0]
            #for i in range(1, len(new_data_df_list)):
            #    xx = pd.concat([xx, new_data_df_list[i]], axis=0)
            #print(xx)
            #exit(-1)
        
        if new_data_df is None:
            upded_cardinality = origin_data.shape[0]
        else:
            upded_cardinality = origin_data.shape[0] + new_cardinality
        print(f"upded_cardinality={upded_cardinality}")

        new_model_save = str(f"{model_save}_update_{args.update_data}_{args.query_root}_{args.update_skew}_{args.update_corr}.pkl")
        rebuild_model_save = str(f"{model_save}_rebuild_{args.update_data}_{args.query_root}_{args.update_skew}_{args.update_corr}.pkl")
        print(f"updated model will be saved in {new_model_save}")
        print(f"baseline rebuild model will be saved in {rebuild_model_save}")
        #exit(-1)
        if args.update_skew is not None and args.update_corr is not None:
            new_query_path = str(query_root / f"query_{args.update_skew}_{args.update_corr}_0.0")
            new_workload = np.load(os.path.join(new_query_path, "train_query_sc.npy"))
            new_true_card = np.load(os.path.join(new_query_path, "train_true_sc.npy"))
            new_queries = np.load(os.path.join(new_query_path, "test_query_sc.npy"))
            new_queries_true_card = np.load(os.path.join(new_query_path, "test_true_sc.npy"))
            # print(new_queries)
            # print(new_queries.shape)
            # print(new_queries_true_card)
            # print(new_queries_true_card.shape)
            # new_queries_list = divide_ndarr(new_queries, 10)
            # new_queries_true_card_list = divide_ndarr(new_queries_true_card, 10)
            # for i, qli in enumerate(new_queries_list):
            #     print(qli)
            #     print(qli.shape)
            #     print(new_queries_true_card_list[i])
            #     print(new_queries_true_card_list[i].shape)
            # exit(-1)
            print(f"new workload: {new_query_path}")
        
        if new_data_df is not None:
            print('Calculating test_true_sc on upded_data...')
            # #new_data: csv2df
            # csv_data = origin_data
            # new_data_columns_order = [None] * len(new_data_columns)
            # for i in new_data_columns:
            #     new_data_columns_order[new_data_columns[i]] = i
            # new_data_dd = {new_data_columns_order[i]: [csv_data.iloc[j][i] for j in range(len(csv_data))] for i in tqdm(range(len(new_data_columns_order)))}
            # new_data_index = [i for i in range(len(csv_data))]
            # df = pd.DataFrame(new_data_dd, index=new_data_index)
            #calc new_queries on new_data(df)
            new_queries_new_data_true_card = np.full(new_queries.shape[0], -1)
            for i in tqdm(range(new_queries.shape[0])):
                pred = ['{}<={}<={}'.format(new_queries[i, j, 0], c, new_queries[i, j, 1]) for j, c in enumerate(new_data_columns)]
                preds = ' and '.join(pred)
                new_queries_new_data_true_card[i] = len(new_data_df.query(preds))
                new_queries_true_card[i] += new_queries_new_data_true_card[i]
            #print(new_queries_true_card)
            #print(new_queries_true_card < upded_cardinality)
            #exit(-1)
        #print(origin_data)

        new_queries = new_queries[new_queries_true_card!=0]
        new_queries_true_card = new_queries_true_card[new_queries_true_card!=0]

        print("Inference origin model, workload size: ", new_queries.shape)
        model = FSPN()
        model.model = spn
        model.store_factorize_as_dict()
        est_origin_model_card = []
        if args.model == 'qspn':
            tic = perf_counter()
            #est_card = model.probability((workload[:,:,0],workload[:,:,1]), calculated=dict()) * cardinality
            #total_time = perf_counter() - tic 
            for wi in range(new_queries.shape[0]):
                e = model.probability((new_queries[wi,:,0].reshape(1,-1), new_queries[wi,:,1].reshape(1,-1)),
                                        calculated=dict()) * upded_cardinality
                #print(wi, e, true_card[wi])
                #exit(-1)
                est_origin_model_card.append(e[0])
            total_origin_model_time = perf_counter() - tic 
            est_origin_model_card = np.array(est_origin_model_card)
            print('origin model qspn result:')
            print(f"querying {len(new_queries_true_card)} queries takes {total_origin_model_time} secs (avg. {total_origin_model_time*1000/len(new_queries_true_card)} ms per query)")
            errors = np.maximum(np.divide(est_origin_model_card, new_queries_true_card), np.divide(new_queries_true_card, est_origin_model_card))
            #est_card_zero = est_card[est_card==0]
            #print(est_card_zero)
            print(new_queries_true_card)
            print("Q-Error distributions are:")
            for n in [50, 90, 95, 99, 100]:
                print(f"{n}% percentile:", np.percentile(errors, n))
            print('Mean: {}'.format(np.mean(errors)))
        
        new_data = None #.csv
        if update_data_path is not None:
            new_data = pd.read_csv(str(update_data_path), header=None)
            new_data_columns = {c: i for i, c in enumerate(list(new_data.iloc[0]))}
            print(new_data_columns)
            print(columns)
            #exit(-1)
            assert new_data_columns == columns
            new_data = new_data.drop(0)
            new_cardinality = new_data.shape[0]
            print(f"new_data shape: {new_data.shape}")
            print(f"new_cardinality: {new_cardinality}, new_data_columns: ", new_data_columns)
            #exit(-1)
            #print(new_data)
            print(type(new_data))
        
        #sample new_workload
        new_workload_sample_rate = 0.2
        new_workload_sample_size = int(new_workload_sample_rate * new_workload.shape[0])
        new_workload_sample_index = random.sample([i for i in range(new_workload.shape[0])], new_workload_sample_size)
        new_workload = new_workload[new_workload_sample_index]
        new_true_card = new_true_card[new_workload_sample_index]

        print("Update QSPN, origin workload size: ", origin_workload.shape)
        print("Update QSPN, new workload size(sample): ", new_workload.shape)
        print('Press ENTER to Continue ...')
        input()

        #sample_origin_data = origin_data.values.astype(int)
        #print(type(sample_origin_data), sample_origin_data.shape)
        #print(sample_origin_data[-1])
        #sample_new_data = new_data.values.astype(int)
        #print(type(sample_new_data), sample_new_data.shape)
        #print(sample_new_data[-1])
        #exit(-1)

        if args.qsum:
            cluster = 'kmeans'
        else:
            cluster = None

        #Baseline: Whole rebuild
        if origin_data is not None and len(origin_data) > 0:
            sample_origin_data = data.values.astype(int)
        else:
            sample_origin_data = None
        if new_data is not None and len(new_data) > 0:
            sample_new_data = new_data.values.astype(int)
        else:
            sample_new_data = None
        if new_data is not None and len(new_data) > 0:
            updated_data = np.concatenate((sample_origin_data, sample_new_data), axis=0)
        else:
            updated_data = sample_origin_data
        assert updated_data.shape[1] == len(origin_data.columns)
        if new_workload is not None and len(new_workload) > 0:
            updated_workload = np.concatenate((origin_workload, new_workload), axis=0)
        else:
            updated_workload = origin_workload
        print(f"updated_data shape: {updated_data.shape}")
        print(f"updated_cardinality: {updated_data.shape[0]}")
        print(f"updated_workload shape: {updated_workload.shape}")
        
        rebuild_parametric_types = [Categorical for i in range(len(origin_data.columns))]
        rebuild_ds_context = Context(parametric_types=rebuild_parametric_types).add_domains(updated_data)
        rebuild_qspn = learn_FSPN(
            updated_data,
            rebuild_ds_context,
            workload=updated_workload,
            queries=cluster,
            rdc_sample_size=100000,
            rdc_strong_connection_threshold=1.1,
            multivariate_leaf=True,
            threshold=0.3,
            wkld_attr_threshold=0.01,
            wkld_attr_bound=(args.Nx, args.lower, args.upper)
        )
        print('Baseline Rebuild Finished.')
        print(f"Baseline Rebuild Model is saved to {rebuild_model_save}")
        pickle.dump(rebuild_qspn, open(rebuild_model_save, "wb"), pickle.HIGHEST_PROTOCOL)
        print('Press ENTER to Continue ...')
        input()

        upd_qspn = update_QSPN(
            spn,
            data=origin_data,
            workload=origin_workload,
            data_insert=new_data,
            data_delete = None,
            new_queries = new_workload,
            rdc_sample_size=100000,
            rdc_threshold=0.3,
            wkld_attr_threshold=0.01,
            wkld_attr_bound=(args.Nx, args.lower, args.upper)
        )
        print('Update Finished.')
        print(f"updateQSPN Model is saved to {new_model_save}")
        pickle.dump(upd_qspn, open(new_model_save, "wb"), pickle.HIGHEST_PROTOCOL)
        print('Press ENTER to Continue ...')
        input()

        print("Inference rebuild_model:",  ", workload size: ", new_queries.shape)
        rebuild_model = FSPN()
        rebuild_model.model = rebuild_qspn
        rebuild_model.store_factorize_as_dict()
        est_rebuild_model_card = []
        if args.model == 'qspn':
            tic = perf_counter()
            #est_card = model.probability((workload[:,:,0],workload[:,:,1]), calculated=dict()) * cardinality
            #total_time = perf_counter() - tic 
            for wi in range(new_queries.shape[0]):
                e = rebuild_model.probability((new_queries[wi,:,0].reshape(1,-1), new_queries[wi,:,1].reshape(1,-1)),
                                        calculated=dict()) * upded_cardinality
                #print(wi, e, true_card[wi])
                #exit(-1)
                est_rebuild_model_card.append(e[0])
            total_rebuild_model_time = perf_counter() - tic 
            est_rebuild_model_card = np.array(est_rebuild_model_card)

        print("Inference upd_model, workload size: ", new_queries.shape)
        upd_model = FSPN()
        upd_model.model = upd_qspn
        upd_model.store_factorize_as_dict()
        est_upd_model_card = []
        if args.model == 'qspn':
            tic = perf_counter()
            #est_card = model.probability((workload[:,:,0],workload[:,:,1]), calculated=dict()) * cardinality
            #total_time = perf_counter() - tic 
            for wi in range(new_queries.shape[0]):
                e = upd_model.probability((new_queries[wi,:,0].reshape(1,-1), new_queries[wi,:,1].reshape(1,-1)),
                                        calculated=dict()) * upded_cardinality
                #print(wi, e, true_card[wi])
                #exit(-1)
                est_upd_model_card.append(e[0])
            total_upd_model_time = perf_counter() - tic 
            est_upd_model_card = np.array(est_upd_model_card)
        
        print('rebuild_qspn(baseline) result:')
        print(f"Model Path: {rebuild_model_save}")
        model_size = os.path.getsize(rebuild_model_save)
        print(f"Model Size: {model_size/1000} KB")
        print(f"querying {len(new_queries_true_card)} queries takes {total_rebuild_model_time} secs (avg. {total_rebuild_model_time*1000/len(new_queries_true_card)} ms per query)")
        errors = np.maximum(np.divide(est_rebuild_model_card, new_queries_true_card), np.divide(new_queries_true_card, est_rebuild_model_card))
        #est_card_zero = est_card[est_card==0]
        #print(est_card_zero)
        print(new_queries_true_card)
        print("Q-Error distributions are:")
        for n in [50, 90, 95, 99, 100]:
            print(f"{n}% percentile:", np.percentile(errors, n))
        print('Mean: {}'.format(np.mean(errors)))
        print()
        
        print('update_qspn result:')
        print(f"Model Path: {new_model_save}")
        model_size = os.path.getsize(new_model_save)
        print(f"Model Size: {model_size/1000} KB")
        print(f"querying {len(new_queries_true_card)} queries takes {total_upd_model_time} secs (avg. {total_upd_model_time*1000/len(new_queries_true_card)} ms per query)")
        errors = np.maximum(np.divide(est_upd_model_card, new_queries_true_card), np.divide(new_queries_true_card, est_upd_model_card))
        #est_card_zero = est_card[est_card==0]
        #print(est_card_zero)
        print(new_queries_true_card)
        print("Q-Error distributions are:")
        for n in [50, 90, 95, 99, 100]:
            print(f"{n}% percentile:", np.percentile(errors, n))
        print('Mean: {}'.format(np.mean(errors)))