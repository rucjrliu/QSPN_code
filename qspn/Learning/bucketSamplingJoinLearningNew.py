from Structure.bucketSamplingJoinBaseNew import MultiQSPN, Table_Models_Partitions, Partition, BucketSample, loc2no
# from Learning.qspnJoinlearningWrapper import learn_FSPN
from Structure.mqspnJoinReader import multi_table_dataset_csv_reader, workload_data_columns_stats, workload_join_pattern_pairs, set_joinkeys_th

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np
import pandas as pd
import math

def set_table_columns(table_columns: dict, table_name: str, columns: list, data):
    assert data.shape[1] == len(columns)
    table_columns[table_name] = {c: i for i, c in enumerate(columns)}

def calc_tables_domain(table_domain: dict, table_names: list, datas: list):
    for table_name, data in zip(table_names, datas):
        data_min = data.min(axis=0, skipna=True)
        data_max = data.max(axis=0, skipna=True)
        table_domain[table_name] = [(data_min[i], data_max[i]) for i in range(data.shape[1])]

def partition_table_by_joinkeys(name: str, data_table: pd.DataFrame, JOINKEY_PARTITION_WIDTH_LIMIT, global_joinkeys_cuts: list, th_joinkeys: list, speval_joinkey: int, top_count_spevals: list):
    #print(name)
    #print(data_table)
    cuts = [None if i is None else pd.cut(data_table[i], bins=global_joinkeys_cuts[ith], right=False, labels=False) for ith, i in enumerate(th_joinkeys)]
    partitions_loc_no = [{None: ()} if i is None else {j: (jth,) for jth, j in enumerate(i.unique())} for i in cuts]
    #print(partitions_loc_no)
    #print()
    #return None, None
    real_cuts = []
    real_cuts_th = []
    partitions_shape = []
    for ith, i in enumerate(cuts):
        nos_i = partitions_loc_no[ith]
        if i is not None:
            real_cuts.append(i)
            real_cuts_th.append(ith)
            partitions_shape.append(len(nos_i))
    partitions_shape = tuple(partitions_shape)
    #print(partitions_shape)
    #print(data_table.shape[0])
    #print(real_cuts)
    #return None, None
    partitions_data = np.full(partitions_shape, None)
    partitions_speval_data = np.full(partitions_shape, None)
    partitions = np.full(partitions_shape, None)
    grouped = data_table.groupby(real_cuts, observed=True)
    lloc = [None] * len(partitions_loc_no)
    check_n = 0
    check_speval_n = 0
    for k, d in grouped:
        tuple_k = k if type(k) is tuple else (k,)
        for ith, i in zip(real_cuts_th, tuple_k):
            lloc[ith] = i
        loc = tuple(lloc)
        no = loc2no(partitions_loc_no, loc)
        #print(lloc, loc, no)
        #continue
        if th_joinkeys[speval_joinkey] is None:
            partitions_data[no] = d
        else:
            speval_d = d[d[th_joinkeys[speval_joinkey]].isin(top_count_spevals)]
            common_d = d[~d[th_joinkeys[speval_joinkey]].isin(top_count_spevals)]
            partitions_speval_data[no] = speval_d if speval_d.shape[0] > 0 else None
            partitions_data[no] = common_d if common_d.shape[0] > 0 else None
        #print(partitions_data[no])
        if partitions_data[no] is not None:
            check_n += partitions_data[no].shape[0]
        if partitions_speval_data[no] is not None:
            check_speval_n += partitions_speval_data[no].shape[0]
    assert check_n+check_speval_n == data_table.shape[0], "table {} after partition_table_by_joinkeys, check_n={}+check_speval_n={} while should be {}".format(name, check_n, check_speval_n, data_table.shape[0])
    #print('check_n',check_n)
    #print()
    #return None, None
    return Table_Models_Partitions(name, partitions_loc_no, partitions), partitions_data, partitions_speval_data

def calc_partition(partition_data, partition_speval_data, th_joinkeys: list):
    n, ndvs, speval_n, speval_ndvs = None, None, None, None
    if partition_data is not None:
        n = partition_data.shape[0]
        ndvs = [None if i is None else partition_data[i].nunique() for i in th_joinkeys]
    if partition_speval_data is not None:
        speval_n = partition_speval_data.shape[0]
        speval_ndvs = [None if i is None else partition_speval_data[i].nunique() for i in th_joinkeys]
    # for i in th_joinkeys:
    #     print(i)
    #     if i is not None:
    #         print(partition_data[i].value_counts())
    # print()
    return Partition(n, ndvs, speval_n, speval_ndvs)

def filtering_workload_pattern(wkld: np.ndarray, queries):
    wkld_pattern = np.zeros((wkld.shape[0], wkld.shape[1]))
    all_patterns = set()
    for ith, i in enumerate(wkld):
        wkld_pattern[ith] = ((i[: , 0] != -np.inf) | (i[: , 1] != np.inf))
        all_patterns.add(tuple(wkld_pattern[ith]))
    return queries if len(all_patterns) < 10 else None

def train_partition_qspn(partition_data: pd.DataFrame, filtering_columns: dict, workload_filtering: np.ndarray, queries, rdc_sample_size, rdc_strong_connection_threshold, multivariate_leaf, threshold, wkld_attr_threshold, wkld_attr_bound, qdcorr, amis):
    filtering_columns_list = [None] * len(filtering_columns)
    for i, ith in filtering_columns.items():
        filtering_columns_list[ith] = i
    sample_data = partition_data[filtering_columns_list].values.astype(np.float64)
    model = BucketSample(sample_data)
    # model = learn_FSPN(
    #     sample_data,
    #     ds_context,
    #     workload=None,
    #     rdc_sample_size=100000,
    #     rdc_strong_connection_threshold=1.1,
    #     multivariate_leaf=multivariate_leaf,
    #     threshold=threshold, 
    #     wkld_attr_bound=None
    # )
    return model

def jsum_partition_tables(mqspn: MultiQSPN, data_tables: dict, table_th_joinkeys: dict, speval_joinkey: int, top_count_spevals: list):
    global_joinkeys_partition_nums = [math.ceil((i[1] - i[0] + 1) / mqspn.JOINKEY_PARTITION_WIDTH_LIMIT) for i in mqspn.global_joinkeys_range]
    global_joinkeys_cuts = [[min(minmax[0] + i * mqspn.JOINKEY_PARTITION_WIDTH_LIMIT, minmax[1] + 1)  for i in range(0, n + 1)] for n, minmax in zip(global_joinkeys_partition_nums, mqspn.global_joinkeys_range)]
    # print(global_joinkeys_partition_nums)
    # for i in global_joinkeys_cuts:
    #     print(i)
    # print()
    #exit(-1)
    table_partitions_data = {}
    table_partitions_speval_data = {}
    for name, data_table in data_tables.items():
        # print('table:', name)
        # for i in table_th_joinkeys[name]:
        #     print('joinkey:', i)
        #     if i is not None:
        #         print(data_table[i].value_counts())
        #         print(np.mean(data_table[i].value_counts()), '+/-', np.std(data_table[i].value_counts()))
        # print()
        mqspn.table_models_partitions[name], table_partitions_data[name], table_partitions_speval_data[name] = partition_table_by_joinkeys(name, data_table, mqspn.JOINKEY_PARTITION_WIDTH_LIMIT, global_joinkeys_cuts, table_th_joinkeys[name], speval_joinkey, top_count_spevals)
        # partitions_data_check = np.full(table_partitions_data[name].shape, None)
        # check_cell_n = 0
        # for ith, i in np.ndenumerate(table_partitions_data[name]):
        #     if i is not None:
        #         partitions_data_check[ith] = 'df'
        #         check_cell_n += 1
        # print(partitions_data_check)
        # print(mqspn.table_models_partitions[name])
        # print(partitions_data_check.shape, np.prod(partitions_data_check.shape), check_cell_n)
    #exit(-1)
    return table_partitions_data, table_partitions_speval_data

# def speval_jsum_partition_tables(mqspn: MultiQSPN, speval_data_tables: dict, table_th_joinkeys: dict):
#     global_joinkeys_partition_nums = [math.ceil((i[1] - i[0] + 1) / mqspn.JOINKEY_PARTITION_WIDTH_LIMIT) for i in mqspn.global_joinkeys_range]
#     global_joinkeys_cuts = [[min(minmax[0] + i * mqspn.JOINKEY_PARTITION_WIDTH_LIMIT, minmax[1] + 1)  for i in range(0, n + 1)] for n, minmax in zip(global_joinkeys_partition_nums, mqspn.global_joinkeys_range)]
#     # print(global_joinkeys_partition_nums)
#     # for i in global_joinkeys_cuts:
#     #     print(i)
#     # print()
#     #exit(-1)
#     speval_table_partitions_data = {}
#     for name, data_table in speval_data_tables.items():
#         if data_table is None:
#             mqspn.speval_table_models_partitions[name], speval_table_partitions_data[name] = mqspn.table_models_partitions[name], None
#         else:
#             mqspn.speval_table_models_partitions[name], speval_table_partitions_data[name] = partition_table_by_joinkeys(name, data_table, mqspn.JOINKEY_PARTITION_WIDTH_LIMIT, global_joinkeys_cuts, table_th_joinkeys[name])
#     #exit(-1)
#     return speval_table_partitions_data

def jproduct_join_filtering(models_partitions: Table_Models_Partitions, partitions_data: np.ndarray, partitions_speval_data: np.ndarray, th_joinkeys: list, filtering_columns: dict, workload_filtering: np.ndarray, queries, rdc_sample_size, rdc_strong_connection_threshold, multivariate_leaf, threshold, wkld_attr_threshold, wkld_attr_bound, qdcorr, amis):
    for no, partition_data in np.ndenumerate(partitions_data):
        partition_speval_data = partitions_speval_data[no]
        if partition_data is not None or partition_speval_data is not None:
            models_partitions.partitions[no] = calc_partition(partition_data, partition_speval_data, th_joinkeys)
            # print(no, models_partitions.partitions[no]._partition_info())
            # continue
            if len(filtering_columns) > 0:
                if partition_data is not None:
                    models_partitions.partitions[no].qspn = train_partition_qspn(partition_data, filtering_columns, workload_filtering, queries, rdc_sample_size, rdc_strong_connection_threshold, multivariate_leaf, threshold, wkld_attr_threshold, wkld_attr_bound, qdcorr, amis)
                    assert models_partitions.partitions[no].qspn is not None, 'patition{} on data.shape={} qspn training FAILED!'.format(no, partition_data.shape)
                if partition_speval_data is not None:
                    models_partitions.partitions[no].speval_qspn = train_partition_qspn(partition_speval_data, filtering_columns, workload_filtering, queries, rdc_sample_size, rdc_strong_connection_threshold, multivariate_leaf, threshold, wkld_attr_threshold, wkld_attr_bound, qdcorr, amis)
                    assert models_partitions.partitions[no].speval_qspn is not None, 'patition{} on data.shape={} speval_qspn training FAILED!'.format(no, partition_data.shape)

def extract_speval(global_joinkeys_range: list, data_tables: dict, table_th_joinkeys: dict, speval_top_num=70):
    joinkey_th_value_counts = [[] for i in range(len(global_joinkeys_range))]
    for name, data in data_tables.items():
        for ith, i in enumerate(table_th_joinkeys[name]):
            if i is not None:
                joinkey_th_value_counts[ith].append(dict(data[i].value_counts()))
    joinkey_special_value = [{} for i in range(len(joinkey_th_value_counts))]
    for ith, i in enumerate(joinkey_th_value_counts):
        for j in i:
            joinkey_special_value[ith] = joinkey_special_value[ith] | j.keys()
    joinkey_special_value_join_counts = [{j: 1 for j in i} for i in joinkey_special_value]
    for ith, i in enumerate(joinkey_special_value):
        for j in i:
            for k in joinkey_th_value_counts[ith]:
                if j in k:
                    joinkey_special_value_join_counts[ith][j] *= k[j]
    for ith, i in enumerate(joinkey_special_value_join_counts):
        # print(ith)
        joinkey_special_value_join_counts[ith] = sorted([(val, cnts) for val, cnts in i.items()], key=lambda t: t[1], reverse=True)
        # print(joinkey_special_value_join_counts[ith])
    # exit(-1)
    top_freq = speval_top_num
    speval_joinkey, speval_max_count = None, 0
    for ith, i in enumerate(joinkey_special_value_join_counts):
        if i[0][1] > speval_max_count:
            speval_joinkey, speval_max_count = ith, i[0][1]
    top_count_spevals = [joinkey_special_value_join_counts[speval_joinkey][i][0] for i in range(min(len(joinkey_special_value_join_counts[speval_joinkey]), top_freq))]
    return speval_joinkey, top_count_spevals

def learn_multi_QSPN(
        partition_width_limit,
        dataset_root,
        workload,
        cols="rdc",
        rows="grid_naive",
        queries="kmeans",
        threshold=0.3,
        rdc_sample_size=100000,
        rdc_strong_connection_threshold=1.1,
        wkld_attr_threshold=0.01,
        wkld_attr_bound=(0.5, 0.1, 0.3),
        multivariate_leaf=False,
        ohe=False,
        leaves=None,
        leaves_corr=None,
        memory=None,
        rand_gen=None,
        cpus=-1,
        updateQSPN_scope=None,
        updateQSPN_workload_all_n=None,
        qdcorr=None,
        qspn_multihist_max_scope_n=1,
        speval_top_num = 100,
        amis=50
):
    #workload: (tables, join_predicates, query), join_predicates=[('table.column', 'table.column)], query=[(table, column, op, value)]
    dc, join_graph = workload_data_columns_stats(workload)
    # na_fix_dc(dc)
    # print(dc)
    # print()
    # print(join_graph)
    # print()
    global_th_joinkeys, table_th_joinkeys = set_joinkeys_th(join_graph)
    # print(global_th_joinkeys)
    # print()
    # print(table_th_joinkeys)
    # print()
    # exit(-1)

    #print(join_graph)
    #exit(-1)
    #read all .csvs (including header in .csv)
    #print(dc)
    #print(len(workload))
    #print(workload[0])
    #print(workload[1])
    #print(workload[-1])
    data_tables = multi_table_dataset_csv_reader(dataset_root, dc, table_th_joinkeys) #table_name: data(DataFrames)
    # exit(-1)
    # for i, t in data_tables.items():
    #     print(i, len(t))
    # exit(-1)
    # for i, ii in data_tables.items():
    #     print(i, ii.shape, ii.dtypes)
    #print(data_tables)
    #exit(-1)
    #class MultiQSPN
    train_time = 0
    mqspn_init_start = perf_counter()

    table_columns = {}
    table_domain = {}
    for i, data in data_tables.items():
        set_table_columns(table_columns, i, list(data.columns), data)
    calc_tables_domain(table_domain, list(data_tables.keys()), list(data_tables.values()))
    assert len(table_columns) == len(table_domain)
    # for i, c in table_columns.items():
    #     print(i)
    #     print(c)
    #     print(table_domain[i])
    #     print()
    # exit(-1)
    global_joinkeys_range_lb = [[] for i in global_th_joinkeys]
    global_joinkeys_range_ub = [[] for i in global_th_joinkeys]
    for it, ics in table_th_joinkeys.items():
        for jth, jc in enumerate(ics):
            if jc is not None:
                jsc = table_columns[it][jc]
                global_joinkeys_range_lb[jth].append(table_domain[it][jsc][0])
                global_joinkeys_range_ub[jth].append(table_domain[it][jsc][1])
    # print(global_joinkeys_range_lb)
    # print(global_joinkeys_range_ub)
    global_joinkeys_range = [(min(lbs), max(ubs))for lbs, ubs in zip(global_joinkeys_range_lb, global_joinkeys_range_ub)]
    print(global_joinkeys_range)
    print(partition_width_limit)
    # print()
    # exit(-1)
    # for belong_i, i in join_graph.items():
    #     print(belong_i)
    #     mini = float('inf')
    #     maxi = float('-inf')
    #     for j in i:
    #         jt, jc = j.split('.')
    #         jsc = table_columns[jt][jc]
    #         jmin, jmax = table_domain[jt][jsc]
    #         mini = min(mini, jmin)
    #         maxi = max(maxi, jmax)
    #         #print('\t', j, mqspn.table_domain[jt][jsc])
    #     for j in i:
    #         jt, jc = j.split('.')
    #         jsc = table_columns[jt][jc]
    #         table_domain[jt][jsc] = (mini, maxi)
    #         mqspn.set_table_column_domain(jt, jc, mini, maxi)
    # for i, c in table_columns.items():
    #     print(i)
    #     print(c)
    #     print(table_domain[i])
    #     print()
    # print(table_th_joinkeys)
    # print()
    speval_joinkey, top_count_spevals = extract_speval(global_joinkeys_range, data_tables, table_th_joinkeys, speval_top_num)
    print()
    print('speval', speval_joinkey, top_count_spevals)
    speval_global_joinkeys_partition_num = math.ceil((global_joinkeys_range[speval_joinkey][1] - global_joinkeys_range[speval_joinkey][0] + 1) / partition_width_limit)
    speval_global_joinkeys_cut = [min(global_joinkeys_range[speval_joinkey][0] + i * partition_width_limit, global_joinkeys_range[speval_joinkey][1] + 1)  for i in range(0, speval_global_joinkeys_partition_num + 1)]
    # print(speval_global_joinkeys_cut)
    cut_spevals = pd.cut(pd.Series(top_count_spevals), bins=speval_global_joinkeys_cut, right=False, labels=False)
    # print(cut_spevals)
    speval_joinkey_partition_width = [0] * speval_global_joinkeys_partition_num
    for val, cnt in cut_spevals.value_counts().iteritems():
        speval_joinkey_partition_width[val] = cnt
    print('speval_joinkey_partition_width', speval_joinkey_partition_width)
    print()
    #exit(-1)
    mqspn = MultiQSPN(table_columns, table_th_joinkeys, partition_width_limit, global_joinkeys_range, table_th_joinkeys, speval_joinkey, speval_joinkey_partition_width)
    mqspn._print()
    # exit(-1)
    # print(mqspn.table_columns)
    # print()
    # print(mqspn.table_filtering_columns)
    # print()
    # print(mqspn.JOINKEY_PARTITION_WIDTH_LIMIT)
    # print(mqspn.global_joinkeys_range)
    # print()
    #exit(-1)
    # for i, domi in mqspn.table_domain.items():
    #     print(i)
    #     for j, jsc in mqspn.table_columns[i].items():
    #         print('\t', j, 'scope =', jsc, domi[jsc])
    # exit(-1)
    # print(mqspn.table_columns)
    # print(mqspn.table_domain)
    # print(mqspn.table_cardinality)
    #exit(-1)
    #print(train_time, 'sec')
    #exit(-1)
    # print('ori')
    # for table_name, table_data in data_tables.items():
    #     print(table_name)
    #     print(table_data)
    #     print()
    # print()
    print('divide speval on joinkey_{} top_{}'.format(speval_joinkey, len(top_count_spevals)))
    print(top_count_spevals)
    print('----------------------------')
    # exit(-1)
    # for table_name, table_data in data_tables.items():
        # if table_th_joinkeys[table_name][speval_joinkey] is None:
        #     data_speval_tables[table_name] = None
        # else:
        #     data_speval_tables[table_name] = table_data[table_data[table_th_joinkeys[table_name][speval_joinkey]].isin(top_count_spevals)]
        #     data_tables[table_name] = table_data[~table_data[table_th_joinkeys[table_name][speval_joinkey]].isin(top_count_spevals)]
    # for table_name in data_tables:
    #     print(table_name, table_th_joinkeys[table_name])
    #     print(data_speval_tables[table_name])
    #     print(data_tables[table_name])
    #     print()
    #     print()
    # exit(-1)
    table_partitions_data, table_partitions_speval_data = jsum_partition_tables(mqspn, data_tables, table_th_joinkeys, speval_joinkey, top_count_spevals)
    # for i in mqspn.table_models_partitions:
    #     print(i)
    #     print(mqspn.table_models_partitions[i].partitions_loc_no)
    #     assert table_partitions_data[i].shape == table_partitions_speval_data[i].shape
    #     for jth, _ in np.ndenumerate(mqspn.table_models_partitions[i].partitions):
    #         print(jth)
    #         print(table_partitions_data[i][jth])
    #         print('speval')
    #         print(table_partitions_speval_data[i][jth])
    #         print()
    #     print('----------------------------')
    #exit(-1)
    #gen sub-workload for each table
    # mqspn._print(info_level=0)
    # exit(-1)
    table_workloads = {i: [] for i in data_tables} #table_name: [(join_scope_list, query_ndarray)]
    table_workloads_patterns = {i: None for i in data_tables}
    for i in workload:
        for j in i[0]:
            # workload_i_join = set()
            # for k in i[1]:
            #     for l in k:
            #         lt, lc = l.split('.')
            #         if lt == j:
            #             workload_i_join.add(mqspn.table_columns[j][lc])
            query_ndarray = np.zeros((len(mqspn.table_filtering_columns[j]), 2))
            query_ndarray[:, 0] = float('-inf')
            query_ndarray[:, 1] = float('inf')
            table_workloads[j].append(query_ndarray)
        for k in i[2]:
            query_ndarray = table_workloads[k[0]][-1]
            k_scope = mqspn.table_filtering_columns[k[0]][k[1]]
            if k[2] == '=':
                query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
                query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
            elif k[2] == '<=':
                query_ndarray[k_scope, 1] = min(k[3], query_ndarray[k_scope, 1])
            elif k[2] == '<':
                query_ndarray[k_scope, 1] = min(k[3] - 1, query_ndarray[k_scope, 1])
            elif k[2] == '>=':
                query_ndarray[k_scope, 0] = max(k[3], query_ndarray[k_scope, 0])
            elif k[2] == '>':
                query_ndarray[k_scope, 0] = max(k[3] + 1, query_ndarray[k_scope, 0])
    print()
    print('filtering workloads:')
    print(len(workload))
    for t, wkld in table_workloads.items():
        table_workloads[t] = np.array(wkld)
    for t, wkld in table_workloads.items():
        print(t, wkld.shape)
        table_workloads_patterns[t] = filtering_workload_pattern(wkld, queries)
    # print(table_workloads_patterns)
    # exit(-1)
    #calc partition and train qspn on each partition_data
    for t, partitions_data in table_partitions_data.items():
        partitions_speval_data = table_partitions_speval_data[t]
        print(t)
        jproduct_join_filtering(mqspn.table_models_partitions[t], partitions_data, partitions_speval_data, table_th_joinkeys[t], mqspn.table_filtering_columns[t], table_workloads[t], table_workloads_patterns[t], rdc_sample_size, rdc_strong_connection_threshold, multivariate_leaf, threshold, wkld_attr_threshold, wkld_attr_bound, qdcorr, amis)
    # print()
    # for t, models_partitions in mqspn.table_models_partitions.items():

    #     partitions_output = np.full(models_partitions.partitions.shape, None)
    #     for ith, i in np.ndenumerate(models_partitions.partitions):
    #         if i is not None:
    #             partitions_output[ith] = i._partition_info()
    #     print(t)
    #     print(partitions_output)
    #exit(-1)
    #speval
    # print()
    # print('speval')
    # print()
    # speval_table_partitions_data = speval_jsum_partition_tables(mqspn, data_speval_tables, table_th_joinkeys)
    # for t, partitions_data in speval_table_partitions_data.items():
    #     if partitions_data is not None:
    #         print('speval {}'.format(t))
    #         jproduct_join_filtering(mqspn.speval_table_models_partitions[t], partitions_data, table_th_joinkeys[t], mqspn.table_filtering_columns[t], table_workloads[t], table_workloads_patterns[t], rdc_sample_size, rdc_strong_connection_threshold, multivariate_leaf, threshold, wkld_attr_threshold, wkld_attr_bound, qdcorr)
    train_time += perf_counter() - mqspn_init_start

    mqspn._print(info_level=1)
    # exit(-1)
    # for i in data_workload:
    #     print(i)
    #     for j in range(0, 5):
    #         print(data_workload[i][j])
    #     print()
    # exit(-1)
    #train one model on each data-workload (build_fjbuckets=domain[table])
    # for i, data in data_tables.items():
    #     print(i)
    #     # if i != 'badges' and i != 'users':
    #     #     continue
    #     joined_scope_i = []
    #     for jc, jsc in mqspn.table_columns[i].items():
    #         jt = i
    #         j = '.'.join([jt, jc])
    #         for k in join_graph.values():
    #             if j in k:
    #                 joined_scope_i.append(jsc)
    #                 break
    #     print('table {}, mqspn.table_columns[i]: {}'.format(i, mqspn.table_columns[i]))
    #     #continue
    #     if len(joined_scope_i) > 0:
    #         joined_scope_i = set(joined_scope_i)
    #     else:
    #         joined_scope_i = None
    #     #continue
    #     workload_i = [j[1] for j in data_workload[i]]
    #     workload_i = np.array(workload_i)
    #     workload_i_join = [j[0] for j in data_workload[i]]
    #     sample_data = data.values.astype(np.float64)
    #     # print(sample_data.shape)
    #     # print(workload_i.shape, len(workload_i_join), workload_i_join[0 : 10])
    #     # for i in range(len(data)):
    #     #     print(data.iloc[i])
    #     # print('after astype')
    #     # for i in sample_data:
    #     #     print(i)
    #     # exit(-1)
    #     parametric_types = [Categorical for i in range(len(data.columns))]
    #     ds_context = Context(parametric_types=parametric_types).add_domains(sample_data)
    #     qspn = None
    #     train_i_time = 0
    #     #print join info on table i
    #     # print(i, workload_i)
    #     # print(i, workload_i_join)
    #     # print(i, joined_scope_i)
    #     # print()
    #     # continue
    #     qspn, train_i_time = learn_FSPN(
    #         sample_data,
    #         ds_context,
    #         workload=workload_i,
    #         queries=queries,
    #         rdc_sample_size=rdc_sample_size,
    #         rdc_strong_connection_threshold=rdc_strong_connection_threshold,
    #         multivariate_leaf=multivariate_leaf,
    #         threshold=threshold,
    #         wkld_attr_threshold=wkld_attr_threshold,
    #         wkld_attr_bound=wkld_attr_bound,
    #         qspn_multihist_max_scope_n=qspn_multihist_max_scope_n,
    #         build_fjbuckets=mqspn.table_domain[i],
    #         workload_join=workload_i_join,
    #         joined_scope=joined_scope_i
    #     )
    #     mqspn.set_table_qspn_model(i, qspn)
    #     train_time += train_i_time
    #     print()
    #exit(-1)
    # print(mqspn.table_qspn_model)
    # for i, im in mqspn.table_rdc_adjacency_matrix.items():
    #     print(i)
    #     print(im)
    #exit(-1)
    #return MultiQSPN
    #exit(-1)
    return mqspn, train_time
