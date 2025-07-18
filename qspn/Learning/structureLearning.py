import logging
import copy
import multiprocessing
import os
import time
import pdb
from copy import deepcopy
from collections import deque
from enum import Enum
from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
from Learning.utils import convert_to_scope_domain, get_matached_domain
from Learning.statistics import get_structure_stats
from sklearn.metrics import silhouette_score
from Learning.splitting.Workload import split_queries_by_clusters
from Learning.splitting.Workload import split_queries_by_maxcut_clusters
from Learning.splitting.Workload import get_split_queries_MaxCut_new, get_split_queries_MaxCut_old
from Learning.splitting.Workload import MAXCUT_K
from scipy.cluster import vq
import xgboost as xgb
#from Learning.qspnJoinBase import FJBuckets

logger = logging.getLogger(__name__)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

from Learning.validity import is_valid
from Learning.splitting.RDC import rdc_test
from Learning.splitting.Workload import *
from Structure.nodes import Product, Sum, Factorize, QSum, assign_ids

parallel = True

QSPLIT_MAXCUT = True

if parallel:
    cpus = max(1, os.cpu_count() - 2)
else:
    cpus = 1
pool = multiprocessing.Pool(processes=cpus)


def calculate_RDC(data, ds_context, scope, condition, sample_size):
    """
    Calculate the RDC adjacency matrix using the data
    """
    tic = time.time()
    scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
    meta_types = ds_context.get_meta_types_by_scope(scope_range)
    domains = ds_context.get_domains_by_scope(scope_range)
    #print('RDC: domains:', domains)
    #exit(-1)

    # calculate the rdc scores, the parameter k to this function are taken from SPFlow original code
    if len(data) <= sample_size:
        rdc_adjacency_matrix = rdc_test(
            data, meta_types, domains, k=10
        )
    else:
        local_data_sample = data[np.random.randint(data.shape[0], size=sample_size)]
        #print('local_data_sample len=', len(local_data_sample))
        #print(domains)
        #exit(-1)
        rdc_adjacency_matrix = rdc_test(
            local_data_sample, meta_types, domains, k=10
        )
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    logging.debug(f"calculating pairwise RDC on sample {sample_size} takes {time.time() - tic} secs")
    return rdc_adjacency_matrix, scope_loc, condition_loc


class Operation(Enum):
    CREATE_LEAF = 1  # A leaf node
    SPLIT_COLUMNS = 2  # A product node
    SPLIT_ROWS = 3  # A sum node
    NAIVE_FACTORIZATION = 4  # This refers to consider the variables in the scope as independent
    REMOVE_UNINFORMATIVE_FEATURES = 5  # If all data of certain attribute are the same
    FACTORIZE = 6  # A factorized node
    REMOVE_CONDITION = 7  # Remove independent set from the condition
    SPLIT_ROWS_CONDITION = 8  # A Split node

    SPLIT_COLUMNS_CONDITION = 9  # NOT IMPLEMENTED Split rows when there is condition, using conditional independence
    FACTORIZE_CONDITION = 10  # NOT IMPLEMENTED Factorize columns when there is condition

    SPLIT_QUERIES = 11 # A QSUM node


def get_next_operation(ds_context, min_instances_slice=100, min_features_slice=1, multivariate_leaf=True, split_queries=None,
                       threshold=0.3, rdc_sample_size=50000, rdc_strong_connection_threshold=0.75, wkld_attr_threshold=0.01, wkld_attr_bound=(0.2, 0.5)):
    """
    :param ds_context: A context specifying the type of the variables in the dataset
    :param min_instances_slice: minimum number of rows to stop splitting by rows
    :param min_features_slice: minimum number of feature to stop splitting by columns usually 1
    :param multivariate_leaf: If true, we fit joint distribution with multivariates.
                              This only controls the SPN branch.
    :return: return a function, call to which will generate the next operation to perform
    """

    def next_operation(
            data,
            workload,
            num_queries,
            scope,
            condition,
            no_clusters=False,
            no_independencies=False,
            no_condition=False,
            is_strong_connected=False,
            workload_join=None,
            rdc_threshold=threshold,
            rdc_strong_connection_threshold=rdc_strong_connection_threshold,
            wkld_attr_threshold=wkld_attr_threshold,
            wkld_attr_bound=wkld_attr_bound
    ):
        """
        :param data: local data set
        :param scope: scope of parent node
        :param workload: local workload set
        :param num_queries: num of queries in original training workload set
        :param condition: scope of conditional of parent node
        :param no_clusters: if True, we are assuming the highly correlated and unseparable data
        :param no_independencies: if True, cannot split by columns
        :param no_condition: if True, cannot remove conditions
        :param rdc_threshold: Under which, we will consider the two variables as independent
        :param rdc_strong_connection_threshold: Above which, we will consider the two variables as strongly related
        :return: Next Operation and parameters to call if have any
        """

        assert len(set(scope).intersection(set(condition))) == 0, "scope and condition mismatch"
        assert (len(scope) + len(condition)) == data.shape[1], "Redundant data columns"
        if wkld_attr_bound is not None:
            assert len(wkld_attr_bound) == 3 and wkld_attr_bound[1] < wkld_attr_bound[2] and workload is not None
            u = wkld_attr_bound[2]
            l = wkld_attr_bound[1]
            e = np.exp(1)
            Nx = wkld_attr_bound[0]
            Ny = 1.0
            b = Ny*(u-(e**Nx)*l)/(1-(e**Nx))
            k = np.log(u-b)
            print(f"Nx: {Nx}, l:{l}, u:{u}")

        minimalFeatures = len(scope) == min_features_slice
        minimalInstances = data.shape[0] <= min_instances_slice

        if minimalFeatures and len(condition) == 0:
            return Operation.CREATE_LEAF, None

        query_attr_in_condition = [i for i in condition if i not in ds_context.fanout_attr]
        if is_strong_connected and (len(condition) == 0 or len(query_attr_in_condition) == 0):
            # the case of strongly connected components, directly model them
            #print('EEEEEEEEEEERRRRRRRRRR')
            return Operation.CREATE_LEAF, None

        if (minimalInstances and len(condition) == 0) or (no_clusters and len(condition) <= 1):
            #print('RRRRRRRRRRREEEEEEEEEE')
            if (multivariate_leaf or is_strong_connected):
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        #print('Check if all data of an attribute has the same value (very possible for categorical data)...')
        # Check if all data of an attribute has the same value (very possible for categorical data)
        uninformative_features_idx = np.var(data, 0) == 0
        #print(np.var(data, 0))
        #print(uninformative_features_idx)
        #exit(-1)
        ncols_zero_variance = np.sum(uninformative_features_idx)
        if ncols_zero_variance > 0:
            if ncols_zero_variance == data.shape[1]:
                if multivariate_leaf:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            else:
                feature_idx = np.asarray(sorted(scope + condition))
                uninformative_features = list(feature_idx[uninformative_features_idx])
                if len(condition) == 0 or len(set(uninformative_features).intersection(set(condition))) != 0:
                    # This is very messy here but essentially realigning the scope and condition with the data column
                    return (
                        Operation.REMOVE_UNINFORMATIVE_FEATURES,
                        (get_matached_domain(uninformative_features_idx, scope, condition))
                    )
        #print('condition:', condition)
        #print(no_clusters, minimalInstances)
        #exit(-1)
        if len(condition) != 0 and no_condition:
            """
                In this case, we have no condition to remove. Must split rows or create leaf.
            """
            if minimalInstances:
                if multivariate_leaf or is_strong_connected:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.NAIVE_FACTORIZATION, None
            elif not no_clusters:
                return Operation.SPLIT_ROWS_CONDITION, calculate_RDC(data, ds_context, scope, condition,
                                                                     rdc_sample_size)

        elif len(condition) != 0:
            """Try to eliminate some of condition, which are independent of scope
            """
            rdc_adjacency_matrix, scope_loc, condition_loc = calculate_RDC(data, ds_context, scope, condition,
                                                                           rdc_sample_size)
            independent_condition = []
            remove_cols = []
            for i in range(len(condition_loc)):
                cond = condition_loc[i]
                is_indep = True
                for s in scope_loc:
                    if rdc_adjacency_matrix[cond][s] > rdc_threshold:
                        is_indep = False
                        continue
                if is_indep:
                    remove_cols.append(cond)
                    independent_condition.append(condition[i])

            if len(independent_condition) != 0:
                return Operation.REMOVE_CONDITION, (independent_condition, remove_cols)

            else:
                # If there is nothing to eliminate from conditional set, we split rows
                if minimalInstances:
                    return Operation.CREATE_LEAF, None
                else:
                    return Operation.SPLIT_ROWS_CONDITION, (rdc_adjacency_matrix, scope_loc, condition_loc)


        elif not no_clusters and not minimalInstances:
            """In this case:  len(condition) == 0 and not minimalFeatures and not no_clusters
               So we try to split rows or factorize
            """
            #pdb.set_trace()
            #print(workload)
            if workload is not None:
                #print(workload)
                #print(workload.shape)
                #print(type(scope))
                wkld_attr_adjacency_matrix = get_workload_attr_matrix(workload, scope)
                #print(wkld_attr_adjacency_matrix)
                wkld_attr_adjacency_matrix = wkld_attr_adjacency_matrix / num_queries
                #print(wkld_attr_adjacency_matrix)
            print('calc RDC...')
            #exit(-1)
            #print(type(data))
            #print(data.shape)
            #print(ds_context)
            #print(scope)
            #print(condition)
            #print(rdc_sample_size)
            rdc_adjacency_matrix, scope_loc, _ = calculate_RDC(data, ds_context, scope, condition, rdc_sample_size)
            #exit(-1)
            #print(rdc_adjacency_matrix)
            #for i in rdc_adjacency_matrix:
            #    print(list(i))
            #print(scope_loc)
            #print(_)
            #exit(-1)

            if not no_independencies:
                # test independence
                if wkld_attr_bound is None:
                    #print(rdc_threshold)
                    rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0
                else:
                    new_threshold = (np.power(e, -Nx*wkld_attr_adjacency_matrix+k) + b)/Ny
                    print('rdc_threshold={}'.format(rdc_threshold))
                    print('new_threshold={}'.format(new_threshold))
                    new_threshold_array = new_threshold.reshape(np.prod(new_threshold.shape))
                    new_threshold_stderr = np.std(new_threshold_array)
                    new_threshold_mean = np.mean(new_threshold_array)
                    print(new_threshold_mean, new_threshold_stderr)
                    print(rdc_threshold)
                    #exit(-1)
                    if new_threshold_mean + new_threshold_stderr >= wkld_attr_bound[2]:
                        rdc_adjacency_matrix[rdc_adjacency_matrix < new_threshold] = 0
                    else:
                        rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0
                #print(rdc_adjacency_matrix)
                num_connected_comp = 0
                indep_res = np.zeros(data.shape[1])
                for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
                    #print(i, c)
                    indep_res[list(c)] = i + 1
                    num_connected_comp += 1
                #exit(-1)
                if num_connected_comp > 1:
                    # there exists independent sets, split by columns
                    node_error = [None, None]
                    node_error[0] = {'rdc_adjacency_matrix': rdc_adjacency_matrix, 'typ': 'D'}
                    node_error[1] = {'data_n': data.shape[0], 'rdc_adjacency_matrix': deepcopy(rdc_adjacency_matrix)}
                    if workload is not None:
                        node_error[0]['wkld_attr_adjacency_matrix'] = wkld_attr_adjacency_matrix
                        node_error[1]['queries_n'] = workload.shape[0]
                        node_error[1]['wkld_attr_adjacency_matrix'] = deepcopy(wkld_attr_adjacency_matrix)
                    #print(node_error)
                    #exit(-1)
                    #if node_error[1]['queries_n'] == 0:
                    #    print(scope, 'Q', node_error)
                    #    exit(-1)
                    return Operation.SPLIT_COLUMNS, (indep_res, node_error, 'D')
            
            if workload is not None:
                #print(wkld_attr_threshold)
                #assert wkld_attr_threshold == 0.001
                wkld_attr_adjacency_matrix[wkld_attr_adjacency_matrix < wkld_attr_threshold] = 0
                wkld_num_connected_comp = 0
                wkld_indep_res = np.zeros(data.shape[1])
                for i, c in enumerate(
                    connected_components(from_numpy_matrix(wkld_attr_adjacency_matrix))
                ):
                    wkld_indep_res[list(c)] = i + 1
                    wkld_num_connected_comp += 1
                if wkld_num_connected_comp > 1:
                    node_error = [None, None]
                    node_error[0] = {'rdc_adjacency_matrix': rdc_adjacency_matrix, 'typ': 'Q'}
                    node_error[1] = {'data_n': data.shape[0], 'rdc_adjacency_matrix': deepcopy(rdc_adjacency_matrix)}
                    node_error[0]['wkld_attr_adjacency_matrix'] = wkld_attr_adjacency_matrix
                    node_error[1]['queries_n'] = workload.shape[0]
                    node_error[1]['wkld_attr_adjacency_matrix'] = deepcopy(wkld_attr_adjacency_matrix)
                    #print(node_error)
                    #exit(-1)
                    return Operation.SPLIT_COLUMNS, (wkld_indep_res, node_error, 'Q')

            rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_strong_connection_threshold] = 0
            strong_connected_comp = []  # strongly connected components
            for c in connected_components(from_numpy_matrix(rdc_adjacency_matrix)):
                if len(c) > 1:
                    component = list(c)
                    component.sort()
                    for i in range(len(c)):
                        component[i] = scope[component[i]]
                    strong_connected_comp.append(component)

            if len(strong_connected_comp) != 0:
                #print(strong_connected_comp)
                #print(scope)
                if strong_connected_comp[0] == scope:
                    # the whole scope is actually strongly connected
                    return Operation.CREATE_LEAF, None
                #exit(-1)
                # there exists sets of strongly connect component, must factorize them out
                return Operation.FACTORIZE, strong_connected_comp

        elif minimalInstances:
            if (multivariate_leaf or is_strong_connected):
                return Operation.CREATE_LEAF, None
            else:
                return Operation.NAIVE_FACTORIZATION, None

        # if none of the above conditions follows, we split by row or query and try again.
        if len(condition) == 0:
            if split_queries is not None and workload is not None and len(workload) > 1:
                #pdb.set_trace()
                print('next_operation: considering QSPLIT on scope{} with workload{}'.format(scope, workload.shape))
                if QSPLIT_MAXCUT:
                    print('MAXCUT!')
                    score, clusters, centers, maxcut_point = get_split_queries_MaxCut_new(workload, scope)
                    #print(score)
                    #print(centers)
                    #print(rdc_threshold)
                    #print(wkld_attr_bound)
                    #print([len(i) for i in centers])
                    print('MAXCUT.')
                    #exit(-1)
                    #queries_points_too_much = False
                    #for i in centers:
                    #    if len(i) > MAXCUT_K:
                    #        queries_points_too_much = True
                    #exit(-1)
                    if len(centers) > 1 and score >= 0.7:
                        #print(score)
                        #print(clusters)
                        #print(scope)
                        #print(centers)
                        #print(maxcut_point)
                        node_error = [{'maxcut_opt_score': score}, {'maxcut_cutset_point': maxcut_point}]
                        #print(node_error)
                        #exit(-1)
                        return Operation.SPLIT_QUERIES, (clusters, centers, node_error)
                else:
                    queries, clusters, centers = split_queries(workload, scope, return_clusters=True)
                    unique_clusters = np.sort(np.unique(clusters))
                    if len(unique_clusters) > 1 and silhouette_score(queries, clusters) > 0.2:
                        print(clusters)
                        #print(type(clusters))
                        #print(type(centers))
                        print(scope)
                        print(centers)
                        #exit(-1)
                        return Operation.SPLIT_QUERIES, (clusters, centers, None)

            return Operation.SPLIT_ROWS, None
        else:
            return Operation.SPLIT_ROWS_CONDITION, calculate_RDC(data, ds_context, scope, condition, rdc_sample_size)

    return next_operation


def default_slicer(data, cols, num_cond_cols=None):
    if num_cond_cols is None:
        if len(cols) == 1:
            return data[:, cols[0]].reshape((-1, 1))

        return data[:, cols]
    else:
        return np.concatenate((data[:, cols], data[:, -num_cond_cols:]), axis=1)


def learn_structure(
        dataset,
        ds_context,
        workload,
        split_rows,
        split_rows_condition,
        split_cols,
        split_queries,
        create_leaf,
        create_leaf_multi,
        threshold,
        rdc_sample_size,
        next_operation=None,
        min_row_ratio=0.01,
        rdc_strong_connection_threshold=0.75,
        wkld_attr_threshold=0.01, #origin 0.01
        wkld_attr_bound=(0.2, 0.5),
        multivariate_leaf=True,
        create_leaf_fanout=None,
        initial_scope=None,
        data_slicer=default_slicer,
        debug=True,
        updateQSPN_scope=None,
        updateQSPN_workload_all_n=None,
        qdcorr=None,
        build_fjbuckets=None,
        workload_join=None
):
    assert dataset is not None
    assert ds_context is not None
    assert split_rows is not None
    assert split_cols is not None
    assert create_leaf is not None
    assert create_leaf_multi is not None
    #pdb.set_trace()

    train_start = perf_counter()
    #print(next_operation)
    #exit(-1)
    if next_operation == None:
        next_operation = get_next_operation(ds_context, int(min_row_ratio * dataset.shape[0]),
                                            threshold=threshold, rdc_sample_size=rdc_sample_size,
                                            rdc_strong_connection_threshold=rdc_strong_connection_threshold,
                                            wkld_attr_threshold=wkld_attr_threshold,
                                            wkld_attr_bound=wkld_attr_bound,
                                            multivariate_leaf=multivariate_leaf,
                                            split_queries=split_queries)
    
    num_queries = len(workload) if workload is not None else None
    if updateQSPN_workload_all_n is not None:
        num_queries = updateQSPN_workload_all_n
    print(dataset.shape)
    print('num_queries=', num_queries)
    if workload is not None:
        print(workload.shape)
    #exit(-1)
    
    root = Product()
    root.children.append(None)
    #print(ds_context.meta_types, ds_context.domains, ds_context.parametric_types, ds_context.feature_names, ds_context.fanout_attr, ds_context.fanout_attr_positive, ds_context.fanout_attr_inverse)
    #exit(-1)

    if initial_scope is None:
        if updateQSPN_scope is not None:
            assert dataset.shape[1] == len(updateQSPN_scope)
            initial_scope = deepcopy(updateQSPN_scope)
        else:
            initial_scope = list(range(dataset.shape[1]))
        initial_cond = []
        num_conditional_cols = None
    elif len(initial_scope) < dataset.shape[1]:
        num_conditional_cols = dataset.shape[1] - len(initial_scope)
        initial_cond = [item for item in list(range(dataset.shape[1])) if item not in initial_scope]
    else:
        num_conditional_cols = None
        initial_cond = []
        assert len(initial_scope) > dataset.shape[1], "check initial scope: %s" % initial_scope
    
    tasks = deque()
    print(initial_scope)
    print(type(initial_scope))
    #exit(-1)
    tasks.append((dataset, workload, root, 0, initial_scope, initial_cond, None, None, False, False, False, False, None))

    while tasks:

        local_data, local_workload, parent, children_pos, scope, condition, cond_fanout_data, rect_range, no_clusters,\
        no_independencies, no_condition, is_strong_connected, right_most_branch = tasks.popleft()

        if debug:
            logging.debug(f"Current task with data {local_data.shape} scope {scope} and condition {condition}")
        print(f"Current task with data {local_data.shape} scope {scope} and condition {condition}")
        #num_queries = len(local_workload)
        operation, op_params = next_operation(
            local_data,
            local_workload,
            num_queries,
            scope,
            condition,
            no_clusters=no_clusters,
            no_independencies=no_independencies,
            no_condition=no_condition,
            is_strong_connected=is_strong_connected
        )
        if local_workload is not None:
            wshape = local_workload.shape
        else:
            wshape = "No"
        if debug:
            logging.debug("OP: {} on data slice {}, workload slice {} (remaining tasks {})".format(operation, local_data.shape, wshape, len(tasks)))
        print("OP: {} on data slice {}, workload slice {} (remaining tasks {})".format(operation, local_data.shape, wshape, len(tasks)))

        if operation == Operation.REMOVE_UNINFORMATIVE_FEATURES:
            # Very messy because of the realignment from scope domain, condition domain and data column domain.
            (scope_rm, scope_rm2, scope_keep, condition_rm, condition_keep) = op_params
            new_condition = [condition[i] for i in condition_keep]
            keep_all = [item for item in range(local_data.shape[1]) if item not in condition_rm + scope_rm]

            if len(new_condition) != len(condition) and debug:
                logging.debug(
                    f"find uninformation condition, keeping only condition {new_condition}")
            if len(scope_rm) == 0 and len(new_condition) != 0:
                # only condition variables have been removed
                assert (len(scope) + len(new_condition)) == len(
                    keep_all), f"Redundant data columns, {scope}, {new_condition}, {keep_all}"
                tasks.append(
                    (
                        data_slicer(local_data, keep_all, num_conditional_cols),
                        local_workload,
                        parent,
                        children_pos,
                        scope,
                        new_condition,
                        cond_fanout_data,
                        rect_range,
                        no_clusters,
                        no_independencies,
                        True,
                        is_strong_connected,
                        right_most_branch
                    )
                )
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
            else:
                # we need to create product node if scope variables have been removed
                node = Product()
                node.typ = 'RUF'
                node.scope = copy.deepcopy(scope)
                node.condition = copy.deepcopy(new_condition)
                node.range = copy.deepcopy(rect_range)
                parent.children[children_pos] = node

                rest_scope = copy.deepcopy(scope)
                for i in range(len(scope_rm)):
                    col = scope_rm[i]
                    new_scope = scope[scope_rm2[i]]
                    rest_scope.remove(new_scope)
                    node.children.append(None)
                    assert col not in keep_all
                    if debug:
                        logging.debug(
                            f"find uninformative scope {new_scope}")
                    tasks.append(
                        (
                            data_slicer(local_data, [col], num_conditional_cols),
                            local_workload,
                            node,
                            len(node.children) - 1,
                            [new_scope],
                            [],
                            cond_fanout_data,
                            rect_range,
                            True,
                            True,
                            True,
                            False,
                            right_most_branch
                        )
                    )

                next_final = False

                if len(rest_scope) == 0:
                    continue
                elif len(rest_scope) == 1:
                    next_final = True

                node.children.append(None)
                c_pos = len(node.children) - 1

                if debug:
                    logging.debug(
                        f"The rest scope {rest_scope} and condition {new_condition} keep"
                    )
                    assert (len(rest_scope) + len(new_condition)) == len(keep_all), "Redundant data columns"
                tasks.append(
                    (
                        data_slicer(local_data, keep_all, num_conditional_cols),
                        local_workload,
                        node,
                        c_pos,
                        rest_scope,
                        new_condition,
                        cond_fanout_data,
                        rect_range,
                        next_final,
                        next_final,
                        False,
                        is_strong_connected,
                        right_most_branch
                    )
                )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.REMOVE_CONDITION:
            (independent_condition, remove_cols) = op_params
            new_condition = [item for item in condition if item not in independent_condition]
            keep_cols = [item for item in range(local_data.shape[1]) if item not in remove_cols]
            if debug:
                logging.debug(
                    f"Removed uniformative condition {independent_condition}")
                assert (len(scope) + len(new_condition)) == len(keep_cols), "Redundant data columns"
            tasks.append(
                (
                    data_slicer(local_data, keep_cols, num_conditional_cols),
                    local_workload,
                    parent,
                    children_pos,
                    scope,
                    new_condition,
                    cond_fanout_data,
                    rect_range,
                    no_clusters,
                    no_independencies,
                    True,
                    is_strong_connected,
                    right_most_branch
                )
            )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.SPLIT_ROWS_CONDITION:
            query_attr = [i for i in condition if i not in ds_context.fanout_attr]
            if len(query_attr) == 0 and right_most_branch:
                logging.debug(
                    f"\t\tcreate multi-leaves for scope {scope} and {condition}"
                )
                #if we only have fanout attr left in condition, there is no need to split by row
                node = create_leaf_fanout(local_data, ds_context, scope, condition, cond_fanout_data)
                node.range = rect_range
                parent.children[children_pos] = node
                continue

            split_start_t = perf_counter()
            data_slices = split_rows_condition(local_data, ds_context, scope, condition,
                                               op_params, cond_fanout_data=cond_fanout_data)
            split_end_t = perf_counter()

            if debug:
                logging.debug(
                    "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                )
                if cond_fanout_data is not None:
                    assert len(local_data) == len(cond_fanout_data[1]), \
                    f"mismatched data length of {len(local_data)} and {len(cond_fanout_data[1])}"

            if len(data_slices) == 1:
                tasks.append((local_data, local_workload, parent, children_pos, scope, condition, cond_fanout_data,
                              rect_range, True, False, False, is_strong_connected, right_most_branch))
                continue

            node = Sum()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            for data_slice, range_slice, proportion, fanout_data_slice in data_slices:
                assert (len(scope) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                node.children.append(None)
                node.weights.append(proportion)
                new_rect_range = dict()
                for c in rect_range:
                    if c not in range_slice:
                        new_rect_range[c] = rect_range[c]
                    else:
                        new_rect_range[c] = range_slice[c]
                if debug and fanout_data_slice is not None:
                    assert len(data_slice) == len(fanout_data_slice[1]), \
                        f"mismatched data length of {len(data_slice)} and {len(fanout_data_slice[1])}"
                tasks.append((data_slice, local_workload, node, len(node.children) - 1, scope, condition,
                              fanout_data_slice, new_rect_range, False, False, False,
                              is_strong_connected, right_most_branch))
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.SPLIT_ROWS:
            #pdb.set_trace()
            split_start_t = perf_counter()
            data_slices = split_rows(local_data, ds_context, scope)
            split_end_t = perf_counter()

            if debug:
                logging.debug(
                    "\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                )
            print("\t\tfound {} row clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t))

            if len(data_slices) == 1:
                tasks.append((local_data, local_workload, parent, children_pos, scope, condition, cond_fanout_data,
                              rect_range, False, True, False, is_strong_connected, right_most_branch))
                continue

            node = Sum()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node
            node.cardinality = len(local_data)
            print('|local_data|={}'.format(len(local_data)))
            node.node_error = [{'cluster_err': 0.0, 'centers': []}, {'cluster_err': 0.0, 'data_n': len(local_data)}]
            node.node_error[1]['data_max'] = []
            node.node_error[1]['data_min'] = []
            for data_slice, scope_slice, proportion, center in data_slices:
                print(len(data_slice), scope_slice, proportion, center)
                node.node_error[0]['centers'].append(center)
                _, vqerr = vq.vq(data_slice, np.array([center]))
                node.node_error[0]['cluster_err'] += proportion * np.mean(vqerr)
                assert isinstance(scope_slice, list), "slice must be a list"
                assert (len(scope) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                node.children.append(None)
                node.weights.append(proportion)
                node.cluster_centers.append(center)
                node.node_error[1]['data_max'].append(np.max(data_slice, axis=0))
                node.node_error[1]['data_min'].append(np.min(data_slice, axis=0))
                if local_workload is not None:
                    workload_slice = get_workload_by_data(data_slice, scope_slice, local_workload)
                else:
                    workload_slice = None
                print(len(data_slice))
                tasks.append((data_slice, workload_slice, node, len(node.children) - 1, scope, condition,
                              cond_fanout_data, rect_range, False, False, False,
                              is_strong_connected, right_most_branch))
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            #print(node.node_error[0]['cluster_err'])
            node.node_error[0]['centers'] = np.array(node.node_error[0]['centers'])
            node.node_error[1]['cluster_err'] = node.node_error[0]['cluster_err']
            #print(node.node_error)
            #print(node.children)
            #exit(-1)
            continue

        elif operation == Operation.SPLIT_QUERIES:
            clusters, centers, node_error = op_params
            #print(type(local_workload), len(local_workload))
            workload_slices = None
            if QSPLIT_MAXCUT:
                workload_slices = split_queries_by_maxcut_clusters(local_workload, clusters, scope, centers)
            else:
                workload_slices = split_queries_by_clusters(local_workload, clusters, scope, centers)
            #print('workload_slices:')
            #print(workload_slices)
            #exit(-1)

            if debug:
                logging.debug(
                    "\t\tfound {} workload clusters".format(len(workload_slices))
                )
            #print("\t\tfound {} workload clusters".format(len(workload_slices)))

            # if len(workload_slices) == 1:
            #     tasks.append((local_data, local_workload, parent, children_pos, scope, condition, cond_fanout_data,
            #                   rect_range, False, True, False, is_strong_connected, right_most_branch))
            #     continue

            node = QSum()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            #print(node.scope, node.condition, node.range)
            #exit(-1)
            parent.children[children_pos] = node
            node.cardinality = len(local_data)
            node.queries_n = len(local_workload)
            node.node_error = node_error
            for workload_slice, scope_slice, proportion, center in workload_slices:
                assert isinstance(scope_slice, list), "slice must be a list"
                assert (len(scope) + len(condition)) == local_data.shape[1], "Redundant data columns"
                node.children.append(None)
                node.weights.append(proportion)
                node.cluster_centers.append(center)
                #print(type(workload_slice), workload_slice.shape)
                tasks.append((local_data, workload_slice, node, len(node.children) - 1, scope, condition,
                              cond_fanout_data, rect_range, False, False, False,
                              is_strong_connected, right_most_branch))
                assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                    "node %s has same attribute in both condition and range"
                continue
            print(node.scope, node.condition, node.range)
            print(node.children)
            print(node.weights)
            print(node.cluster_centers)
            print(node.node_error)
            #exit(-1)

        elif operation == Operation.SPLIT_COLUMNS:
            split_start_t = perf_counter()
            print(op_params[0])
            print(local_data.shape)
            print(op_params[2])
            #print(local_data)
            #exit(-1)
            data_slices = split_cols(local_data, ds_context, scope, clusters=op_params[0])
            split_end_t = perf_counter()

            if debug:
                logging.debug(
                    "\t\tfound {} col clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t)
                )
            print("\t\tfound {} col clusters (in {:.5f} secs)".format(len(data_slices), split_end_t - split_start_t))

            if len(data_slices) == 1:
                tasks.append((local_data, local_workload, parent, children_pos, scope, condition, cond_fanout_data,
                              rect_range, False, True, False, is_strong_connected, right_most_branch))
                assert np.shape(data_slices[0][0]) == np.shape(local_data)
                assert data_slices[0][1] == scope
                continue

            node = Product()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            node.node_error = op_params[1]
            node.typ = op_params[2]
            print(node.node_error)
            print(node.scope)
            #if node.node_error[0]['typ'] == 'Q':
            #    exit(-1)
            parent.children[children_pos] = node
            #print(local_workload.shape)
            #print('slices:')

            for data_slice, scope_slice, _ in data_slices:
                print(scope_slice)
                #print(data_slice.shape)
                #print(data_slice)
                assert isinstance(scope_slice, list), "slice must be a list"
                assert (len(scope_slice) + len(condition)) == data_slice.shape[1], "Redundant data columns"
                node.children.append(None)
                if debug:
                    logging.debug(
                        f'Create an independent component with scope {scope_slice} and condition {condition}'
                    )
                if local_workload is not None:
                    workload_slice = get_workload_by_scope(scope_slice, local_workload)
                else:
                    workload_slice = None
                #print(workload_slice.shape)
                tasks.append((data_slice, workload_slice, node, len(node.children) - 1, scope_slice, condition,
                              cond_fanout_data, rect_range, False, True, False,
                              is_strong_connected, right_most_branch))
            #if node.scope[0] != 0 and local_workload.shape[0] > 0:
            #exit(-1)
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            #if node.scope == [4,5,6,7] and node.typ == 'Q':
            #    exit(-1)
            #exit(-1)
            continue

        elif operation == Operation.FACTORIZE:
            # condition should be [] when we do factorize
            node = Factorize()
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node
            index_list = sorted(scope + condition)

            # if there are multiple components we left it for the next round
            if debug:
                for comp in op_params:
                    logging.debug(
                        f'Factorize node found the strong connected component{comp}'
                    )
                logging.debug(
                    f'We only factor out {op_params[0]}'
                )

            strong_connected = op_params[0]
            other_connected = [item for item in scope if item not in strong_connected]

            assert len(other_connected) != 0, "factorize results in only one strongly connected"
            assert cond_fanout_data is None, "conditional data exists"
            node.children.append(None)
            data_copy = copy.deepcopy(local_data)
            if debug:
                logging.debug(
                    f'Factorize node factor out weak connected component{other_connected}'
                )
            keep_cols = [index_list.index(i) for i in sorted(other_connected + condition)]
            tasks.append(
                (
                    data_slicer(data_copy, keep_cols, num_conditional_cols),
                    local_workload,
                    node,
                    0,
                    other_connected,
                    condition,
                    None,
                    rect_range,
                    False,
                    False,
                    False,
                    False,
                    False
                )
            )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            new_condition = sorted(condition + other_connected)
            node.children.append(None)
            new_scope = strong_connected
            keep_cols = [index_list.index(i) for i in sorted(new_scope + new_condition)]
            if debug:
                logging.debug(
                    f'Factorize node found a strongly connect component{new_scope}, '
                    f'condition on {new_condition}'
                )
                assert (len(new_scope) + len(new_condition)) == len(keep_cols), "Redundant data columns"
            if rect_range is None:
                new_rect_range = dict()
            else:
                new_rect_range = copy.deepcopy(rect_range)
            for i, c in enumerate(new_condition):
                condition_idx = []
                for j in new_condition:
                    condition_idx.append(index_list.index(j))
                data_attr = local_data[:, condition_idx[i]]
                new_rect_range[c] = [(np.nanmin(data_attr), np.nanmax(data_attr))]
            cond_fanout_attr = [i for i in new_condition if i in ds_context.fanout_attr]
            if len(cond_fanout_attr) == 0:
                new_condition_fanout_data = None
            else:
                cond_fanout_keep_cols = [index_list.index(i) for i in cond_fanout_attr]
                new_condition_fanout_data = (cond_fanout_attr,
                                             data_slicer(local_data, cond_fanout_keep_cols, num_conditional_cols))
            if right_most_branch is None:
                new_right_most_branch = True
            else:
                new_right_most_branch = False
            tasks.append(
                (
                    data_slicer(local_data, keep_cols, num_conditional_cols),
                    local_workload,
                    node,
                    1,
                    new_scope,
                    new_condition,
                    new_condition_fanout_data,
                    new_rect_range,
                    False,
                    True,
                    False,
                    True,
                    new_right_most_branch
                )
            )
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue

        elif operation == Operation.NAIVE_FACTORIZATION:
            # This is assuming the remaining attributes as independent. FSPN will probably never get here.
            node = Product()
            node.typ = 'N'
            node.scope = copy.deepcopy(scope)
            node.condition = copy.deepcopy(condition)
            node.range = copy.deepcopy(rect_range)
            parent.children[children_pos] = node

            scope_range, scope_loc, condition_loc = convert_to_scope_domain(scope, condition)
            local_tasks = []
            local_children_params = []
            split_start_t = perf_counter()
            # print(scope_loc)
            # print(local_data.shape)
            # if local_workload is not None:
            #     print(local_workload.shape)
            #exit(-1)
            for i, col in enumerate(scope_loc):
                #print(col)
                node.children.append(None)
                local_tasks.append(len(node.children) - 1)
                child_data_slice = data_slicer(local_data, [col], num_conditional_cols)
                local_children_params.append((child_data_slice, ds_context, [scope[i]], []))
            # print('qdcorr =', qdcorr)
            # if local_workload is not None and local_workload.shape[0] > 0 and local_data.shape[0] > 0 and qdcorr == 'XGB':
            #     xgb_trainX = []
            #     xgb_trainY = []
            #     for i in range(local_workload.shape[0]):
            #         Xi = []
            #         Yi = None
            #         qwherei_indep = []
            #         qwherei_all = []
            #         for j, predj in enumerate(node.scope):
            #             qijl = local_workload[i][predj][0]
            #             qijr = local_workload[i][predj][1]
            #             if qijl == float('-inf') and qijr == float('inf'):
            #                 Xi.append(0)
            #             else:
            #                 Xi.append(1)
            #                 qwherei_indep.append((local_data[:, j] >= qijl) & (local_data[:, j] <= qijr))
            #         if len(qwherei_indep) == 0:
            #             Yi = 1.0
            #         else:
            #             qwherei_indep = np.array(qwherei_indep)
            #             qwherei_all = qwherei_indep.all(axis=0)
            #             sel_est = max(1e-9, np.prod([len(local_data[j]) / len(local_data) for j in qwherei_indep]))
            #             sel_accu = max(1e-9, len(local_data[qwherei_all]) / len(local_data))
            #             #print(Xi, sel_est, sel_accu)
            #             Yi = sel_accu / sel_est
            #         xgb_trainX.append(np.array(Xi))
            #         xgb_trainY.append(Yi)
            #     xgb_trainX = np.array(xgb_trainX)
            #     xgb_trainY = np.array(xgb_trainY)
            #     node.qdcorr = xgb.XGBRegressor(objective ='reg:squarederror')
            #     node.qdcorr.fit(xgb_trainX, xgb_trainY)
                #print(xgb_trainX)
                #print(xgb_trainY)
            #exit(-1)
            result_nodes = pool.starmap(create_leaf, local_children_params)

            for child_pos, child in zip(local_tasks, result_nodes):
                node.children[child_pos] = child

            split_end_t = perf_counter()

            logging.debug(
                "\t\tnaive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t)
            )
            print("\t\tnaive factorization {} columns (in {:.5f} secs)".format(len(scope), split_end_t - split_start_t))
            continue

        elif operation == Operation.CREATE_LEAF:
            leaf_start_t = perf_counter()
            if (cond_fanout_data is None or len(cond_fanout_data) == 0) and len(scope) == 1:
                print('create_leaf(single)...')
                node = create_leaf(local_data, ds_context, scope, condition)
            elif create_leaf_fanout is None:
                node = create_leaf_multi(local_data, ds_context, scope, condition)
            elif right_most_branch:
                node = create_leaf_fanout(local_data, ds_context, scope, condition, cond_fanout_data)
            else:
                curr_fanout_attr = [i for i in scope+condition if i in ds_context.fanout_attr]
                if cond_fanout_data is None and len(curr_fanout_attr) == 0:
                    node = create_leaf_multi(local_data, ds_context, scope, condition)
                else:
                    prob_mhl = create_leaf_multi(local_data, ds_context, scope, condition)
                    exp_mhl = create_leaf_fanout(local_data, ds_context, scope, condition, cond_fanout_data)
                    node = Multi_histogram_full(prob_mhl, exp_mhl, scope)
            node.range = rect_range
            parent.children[children_pos] = node
            node.cardinality = len(local_data)
            if build_fjbuckets is not None:
                node.join_buckets = FJBuckets()
                node.join_buckets.calc_from_data(local_data, scope, build_fjbuckets)
            leaf_end_t = perf_counter()

            logging.debug(
                "\t\t created leaf {} for scope={} and condition={} (in {:.5f} secs)".format(
                    node.__class__.__name__, scope, condition, leaf_end_t - leaf_start_t
                )
            )
            print("\t\t created leaf {} for scope={} and condition={} (in {:.5f} secs)".format(
                    node.__class__.__name__, scope, condition, leaf_end_t - leaf_start_t
                ))
            assert len(set(parent.condition).intersection(set(parent.scope))) == 0, \
                "node %s has same attribute in both condition and range"
            continue
        else:
            raise Exception("Invalid operation: " + operation)
    
    train_end = perf_counter()
    train_total_time = train_end - train_start
    node = root.children[0]
    assign_ids(node)
    print(get_structure_stats(node))
    print("training cost: {:.5f} secs".format(train_end-train_start))
    valid, err = is_valid(node)
    assert valid, "invalid fspn: " + err
    #node = Prune(node)
    valid, err = is_valid(node)
    assert valid, "invalid fspn: " + err
    #exit(-1)
    if build_fjbuckets is None:
        return node
    return node, train_total_time

