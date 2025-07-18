import time
import copy
import numpy as np
from Learning.utils import convert_to_scope_domain
from sklearn.cluster import KMeans
from scipy.cluster import vq
try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time
from Learning.splitting.RDC import rdc_test
from Structure.nodes import Product, Sum, Factorize, Leaf, QSum
from Structure.leaves.fspn_leaves.Multi_Histograms import Multi_histogram, multidim_cumsum
from Structure.leaves.fspn_leaves.Histograms import Histogram
from Structure.leaves.fspn_leaves.Merge_leaves import Merge_leaves
from Structure.nodes import qsplit_maxcut_which_childi
from Learning.splitting.Workload import get_workload_attr_matrix, get_workload_by_scope
from Learning.splitting.Workload import get_workload_by_datadom, get_workload_by_data
from networkx.convert_matrix import from_numpy_matrix
from Inference.inference import EPSILON
from Learning.splitting.Workload import split_queries_by_maxcut_clusters, split_queries_by_maxcut_point_encoder, qsplit_train_cluster_decoder, bitset_intersectbits, qsplit_train_cluster_encoder, qsplit_qspnupdate_add_cluster_center_encoder
from Learning.learningWrapper import learn_FSPN
from Structure.nodes import Context
from Structure.leaves.parametric.Parametric import Categorical
from Learning.structureLearning import calculate_RDC
from Learning.statistics import get_structure_stats

def print_graph(G):
    GE = ['{}-{}={}'.format(u,v,w) for u,v,w in G.edges(data='weight')]
    print(GE)

need_rebuild_nodes = []
total_nodes_n = 0
update_performance_boost = True

def top_down_adaptive_rebuild_get_slices(node, ds_context, data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold=0.01, wkld_attr_bound=(0.2, 0.5), cluster_err_threshold=0.3, maxcut_err_threshold=0.3):
    slices = None
    
    if isinstance(node, QSum):
        pass

    elif isinstance(node, Sum):
        pass

    return slices

def top_down_adaptive_rebuild_err_check(node, ds_context, data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold=0.01, wkld_attr_bound=(0.2, 0.5), cluster_err_threshold=0.3, maxcut_err_threshold=0.3):
    global need_rebuild_nodes, total_nodes_n
    #assert isinstance(node, Product)
    #first try ProductD else try ProductQ
    if wkld_attr_bound is not None:
        #print(wkld_attr_bound)
        #print(workload.shape)
        assert len(wkld_attr_bound) == 3 and wkld_attr_bound[1] < wkld_attr_bound[2] and workload is not None
        u = wkld_attr_bound[2]
        l = wkld_attr_bound[1]
        e = np.exp(1)
        Nx = wkld_attr_bound[0]
        Ny = 1.0
        b = Ny*(u-(e**Nx)*l)/(1-(e**Nx))
        k = np.log(u-b)
        #print(f"Nx: {Nx}, l:{l}, u:{u}")

    if isinstance(node, Product):
        print(node.typ)
        #Product node
        #get wkld_attr_adjacency of data(updated)
        if workload is not None:
            wkld_attr_adjacency_matrix = get_workload_attr_matrix(workload, node.scope)
            workload_n = len(workload)
            wkld_attr_adjacency_matrix = wkld_attr_adjacency_matrix / max(1, workload_all_n)
        else:
            wkld_attr_adjacency_matrix = np.zeros(node.node_error[1]['wkld_attr_adjacency_matrix'].shape)
            workload_n = 0
        #print('wkld_rdc', wkld_attr_adjacency_matrix)
        #if ProductD
        if node.node_error[0]['typ'] == 'D':
            #matrix(data_insert)
            #print('calc RDC...')
            #exit(-1)
            if data is not None and len(data) > 1:
                rdc_adjacency_matrix, scope_loc, _ = calculate_RDC(data, ds_context, node.scope, node.condition, rdc_sample_size)
            else:
                rdc_adjacency_matrix = np.zeros(node.node_error[1]['rdc_adjacency_matrix'].shape)
            if wkld_attr_bound is None:
                rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0
            else:
                new_threshold = (np.power(e, -Nx*wkld_attr_adjacency_matrix+k) + b)/Ny
                new_threshold_array = new_threshold.reshape(np.prod(new_threshold.shape))
                new_threshold_stderr = np.std(new_threshold_array)
                new_threshold_mean = np.mean(new_threshold_array)
                print(new_threshold_mean, new_threshold_stderr)
                print(rdc_threshold)
                if new_threshold_mean + new_threshold_stderr >= wkld_attr_bound[2]:
                    rdc_adjacency_matrix[rdc_adjacency_matrix < new_threshold] = 0
                else:
                    rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0
            #print('rdc', rdc_adjacency_matrix)
            #calc G0 and G1 of RDC to check if the RDC of node still holds
            G0 = from_numpy_matrix(node.node_error[0]['rdc_adjacency_matrix'])
            G1 = from_numpy_matrix(rdc_adjacency_matrix)
            #if any edge in G1 but not in G0, need to rebuild
            print_graph(G0)
            print_graph(G1)
            for u,v,w in G1.edges(data='weight'):
                if not G0.has_edge(u, v):
                    return True
        
        #if ProductQ
        elif node.node_error[0]['typ'] == 'Q':
            #print('calc RDC(wkld)...')
            #matrix(new_queries)
            wkld_attr_adjacency_matrix[wkld_attr_adjacency_matrix < wkld_attr_threshold] = 0
            #calc G0 and G1 of wkld_RDC to check if the wkld_RDC of node still holds
            G0 = from_numpy_matrix(node.node_error[0]['wkld_attr_adjacency_matrix'])
            G1 = from_numpy_matrix(wkld_attr_adjacency_matrix)
            #if any edge in G1 but not in G0, need to rebuild
            print_graph(G0)
            print_graph(G1)
            for u,v,w in G1.edges(data='weight'):
                if not G0.has_edge(u, v):
                    return True
    
    elif isinstance(node, QSum):
        #QSum node
        #encode types of new queries to maxcut_point
        workload_point = {}
        for i, q in enumerate(workload):
            pointi = ['1'] * len(node.scope)
            for j, c in enumerate(node.scope):
                if q[c][0] == float('-inf') and q[c][1] == float('inf'):
                    pointi[j] = '0'
            spointi = split_queries_by_maxcut_point_encoder(pointi)
            if spointi not in workload_point:
                workload_point[spointi] = [i]
            else:
                workload_point[spointi].append(i)
        #re-statistic cutset by workload(updated)
        maxcut_cutset_point = copy.deepcopy(node.node_error[1]['maxcut_cutset_point'])
        for i in maxcut_cutset_point:
            for j in i:
                i[j] = 0
        workload_new_points = {}
        for i in workload_point:
            existed = False
            for j in maxcut_cutset_point:
                for k in j:
                    if k == i:
                        j[k] += len(workload_point[i])
                        existed = True
                        break
                if existed:
                    break
            if not existed:
                workload_new_points[i] = {'queries_n': len(workload_point[i]), 'V_weight': 0}
        #initialize E_sum_opt
        E_sum_opt = 0
        for i in maxcut_cutset_point:
            for j in i:
                for k in i:
                    if j != k:
                        E_sum_opt += bitset_intersectbits(j, k) * (i[j] + i[k])
        E_sum_opt //= 2
        #calc V_weight of workload_new_points to decide the order of greedy algorithm
        for i in workload_new_points:
            for j in workload_new_points:
                if j != i:
                    workload_new_points[i]['V_weight'] += bitset_intersectbits(i, j) * (workload_new_points[i]['queries_n'] + workload_new_points[j]['queries_n'])
            for j in maxcut_cutset_point:
                for k in j:
                    workload_new_points[i]['V_weight'] += bitset_intersectbits(i, k) * (workload_new_points[i]['queries_n'] + j[k])
        workload_new_points = sorted([{'V': i, 'queries_n': workload_new_points[i]['queries_n'], 'V_weight': workload_new_points[i]['V_weight']} for i in workload_new_points], reverse=True, key=lambda t : t['V_weight'])
        #new_queries_new_points choose cutset to add by order and calc E_sum_opt
        for i in workload_new_points:
            opt_j = None
            opt = None
            for j in maxcut_cutset_point:
                cost = 0
                for k in j:
                    cost += bitset_intersectbits(i['V'], k) * (i['queries_n'] + j[k])
                if opt is None or opt > cost:
                    opt_j = j
                    opt = cost
            opt_j[i['V']] = i['queries_n']
            E_sum_opt += opt
        #calc E_sum
        Esum = 0
        for i1 in maxcut_cutset_point:
            for j1 in i1:
                for i2 in maxcut_cutset_point:
                    for j2 in i2:
                        if j2 != j1:
                            Esum += bitset_intersectbits(j1, j2) * (i1[j1] + i2[j2])
        Esum //= 2
        #print(E_sum_opt, Esum)
        #calc r_score, the optimization ratio of maxcut_opt to decide whether need to rebuild
        origin_r_score = node.node_error[0]['maxcut_opt_score']
        r_score = (Esum - E_sum_opt) / max(1, Esum)
        print('r_score:', r_score)
        workload_slices = [[] for i in maxcut_cutset_point]
        for q in workload_point:
            for i, c in enumerate(maxcut_cutset_point):
                if q in c:
                    for j in workload_point[q]:
                        workload_slices[i].append(workload[j])
                    break
        for i in range(len(workload_slices)):
            if len(workload_slices[i]) > 0:
                workload_slices[i] = np.array(workload_slices[i])
            else:
                workload_slices[i] = np.zeros((0, workload.shape[1], workload.shape[2]))
        if r_score < (1 - maxcut_err_threshold) * origin_r_score:
            return True, workload_slices
        else:
            return False, workload_slices
    
    elif isinstance(node, Sum):
        #Sum node
        #try to classfy data_insert by node.cluster_centers of node and check K-Means SSE to decide whether need to rebuild
        _, vqerr = vq.vq(data, node.node_error[0]['centers'])
        data_slices = [[] for i in range(len(node.cluster_centers))]
        for i, cci in enumerate(_):
            data_slices[cci].append(data[i])
        for i in range(len(data_slices)):
            if len(data_slices[i]) > 0:
                data_slices[i] = np.array(data_slices[i])
            else:
                data_slices[i] = np.zeros((0, data.shape[1]))
        new_data_cluster_err = np.mean(vqerr)
        return (new_data_cluster_err > (1 + cluster_err_threshold) * node.node_error[0]['cluster_err']), data_slices
    
    return False

def top_down_adaptive_rebuild(node, ds_context, data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold=0.01, wkld_attr_bound=(0.2, 0.5)):
    print(node, data.shape if data is not None else None, workload.shape if workload is not None else None, node.scope)
    print(node.node_error)
    
    if isinstance(node, Leaf):
        #Leaf never needs to rebuild
        return False

    elif isinstance(node, Factorize):
        # a factorize node, not in QSPN
        assert 'Factorize' is None
        left_cols = [node.scope.index(i) for i in node.children[0].scope]
        if data:
            left_data = data[:, left_cols]
        else:
            left_data = None
            return False
        left_ret = top_down_adaptive_rebuild(node.children[0], ds_context, left_data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        right_ret = top_down_adaptive_rebuild(node.children[1], ds_context, data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        return left_ret or right_ret

    elif isinstance(node, QSum):
        #QSum
        #check if the subtree of this node need to build
        rebuild_flag, workload_slices = top_down_adaptive_rebuild_err_check(node, ds_context, data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        #print([i.shape for i in workload_slices])
        if rebuild_flag:
            return True
        #dfs children
        for i, child in enumerate(node.children):
            child_ret = top_down_adaptive_rebuild(child, ds_context, data, workload_slices[i], workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            if child_ret:
                print('Rebuild {}...'.format(child))
                need_rebuild_nodes.append(child)
                #continue
                new_child = learn_FSPN(
                    data,
                    ds_context,
                    workload=workload_slices[i],
                    queries='kmeans',
                    rdc_sample_size=rdc_sample_size,
                    rdc_strong_connection_threshold=1.1,
                    multivariate_leaf=False,
                    threshold=rdc_threshold,
                    wkld_attr_threshold=wkld_attr_threshold,
                    wkld_attr_bound=wkld_attr_bound,
                    updateQSPN_scope=child.scope,
                    updateQSPN_workload_all_n=workload_all_n
                )
                node.children[i] = new_child

    elif isinstance(node, Sum) and node.range is not None:
        # a split node, not in QSPN
        assert node.cluster_centers == [], node
        for child in node.children:
            assert child.range is not None, child
            if data:
                new_data = split_data_by_range(data, child.range, child.scope)
            else:
                new_data = None
            child_ret = top_down_adaptive_rebuild(child, ds_context, new_data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            if child_ret:
                return True
    
    elif isinstance(node, Sum):
        #Sum
        #check if the subtree of this node need to build
        rebuild_flag, data_slices = top_down_adaptive_rebuild_err_check(node, ds_context, data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        #print([i.shape for i in data_slices])
        if rebuild_flag:
            return True
        #dfs children
        for i, child in enumerate(node.children):
            if workload is not None:
                workload_slice = get_workload_by_data(data_slices[i], child.scope, workload)
            else:
                workload_slice = None
            child_ret = top_down_adaptive_rebuild(child, ds_context, data_slices[i], workload_slice, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            if child_ret:
                print('Rebuild {}...'.format(child))
                need_rebuild_nodes.append(child)
                #continue
                new_child = learn_FSPN(
                    data_slices[i],
                    ds_context,
                    workload=workload_slice,
                    queries='kmeans',
                    rdc_sample_size=rdc_sample_size,
                    rdc_strong_connection_threshold=1.1,
                    multivariate_leaf=False,
                    threshold=rdc_threshold,
                    wkld_attr_threshold=wkld_attr_threshold,
                    wkld_attr_bound=wkld_attr_bound,
                    updateQSPN_scope=child.scope,
                    updateQSPN_workload_all_n=workload_all_n
                )
                node.children[i] = new_child
    
    elif isinstance(node, Product):
        # Product (Q or D or N)
        #check if the subtree of this node need to build
        if node.typ == 'N' or node.typ == 'RUF' or node.node_error is None:
            rebuild_flag = False
        else:
            if update_performance_boost:
                rebuild_flag = (hasattr(node, 'rebuild_f') and node.rebuild_f)
            else:
                rebuild_flag = top_down_adaptive_rebuild_err_check(node, ds_context, data, workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        if rebuild_flag:
            return True
        #dfs children
        for i, child in enumerate(node.children):
            index = [node.scope.index(s) for s in child.scope]
            new_data = data[:, index]
            workload_slice = get_workload_by_scope(child.scope, workload)
            child_ret = top_down_adaptive_rebuild(child, ds_context, new_data, workload_slice, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            if child_ret:
                print('Rebuild {}...'.format(child))
                need_rebuild_nodes.append(child)
                #continue
                new_child = learn_FSPN(
                    new_data,
                    ds_context,
                    workload=workload_slice,
                    queries='kmeans',
                    rdc_sample_size=rdc_sample_size,
                    rdc_strong_connection_threshold=1.1,
                    multivariate_leaf=False,
                    threshold=rdc_threshold,
                    wkld_attr_threshold=wkld_attr_threshold,
                    wkld_attr_bound=wkld_attr_bound,
                    updateQSPN_scope=child.scope,
                    updateQSPN_workload_all_n=workload_all_n
                )
                node.children[i] = new_child
                #exit(-1)
    
    return False

def top_down_update_err_check(node, ds_context, data_insert, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold=0.01, wkld_attr_bound=(0.2, 0.5), cluster_err_threshold=0.3, maxcut_err_threshold=0.3):
    global need_rebuild_nodes, total_nodes_n
    #assert isinstance(node, Product)
    rebuild_flag = False
    #first try ProductD else try ProductQ
    if wkld_attr_bound is not None:
        assert len(wkld_attr_bound) == 3 and wkld_attr_bound[1] < wkld_attr_bound[2] and new_queries is not None
        u = wkld_attr_bound[2]
        l = wkld_attr_bound[1]
        e = np.exp(1)
        Nx = wkld_attr_bound[0]
        Ny = 1.0
        b = Ny*(u-(e**Nx)*l)/(1-(e**Nx))
        k = np.log(u-b)
        #print(f"Nx: {Nx}, l:{l}, u:{u}")
    print('Before err_check:', node.node_error)    

    if isinstance(node, Product):
        print(node.typ)
        #get wkld_attr_adjacency of new_queries, new_queries_n and data_n_sum
        if new_queries is not None:
            wkld_attr_adjacency_matrix = get_workload_attr_matrix(new_queries, node.scope)
            new_queries_n = len(new_queries)
            wkld_attr_adjacency_matrix = wkld_attr_adjacency_matrix / max(1, new_queries_all_n)
        else:
            wkld_attr_adjacency_matrix = np.zeros(node.node_error[1]['wkld_attr_adjacency_matrix'].shape)
            new_queries_n = 0
        if 'queries_n' in node.node_error[1]:
            queries_n_sum = node.node_error[1]['queries_n'] + new_queries_n
        else:
            queries_n_sum = new_queries_n
        new_queries_ratio = new_queries_n / max(1, queries_n_sum)
        data_insert_n = data_insert.shape[0] if data_insert is not None else 0
        data_n_sum = node.node_error[1]['data_n'] + data_insert_n
        data_insert_ratio = data_insert_n / data_n_sum
        #merge wkld_attr_adjacency and that of new_queries by simple weighted mean
        #RDC(A+B) ≈ (RDC(A) * |A| + RDC(B) * |B|) / (|A| + |B|) for A is unseen, only see RDC(A)
        if 'wkld_attr_adjacency_matrix' in node.node_error[1]:
            wkld_attr_adjacency_matrix = new_queries_ratio * wkld_attr_adjacency_matrix + (1 - new_queries_ratio) * node.node_error[1]['wkld_attr_adjacency_matrix']
        else:
            wkld_attr_adjacency_matrix = wkld_attr_adjacency_matrix
        #print('wkld_rdc', wkld_attr_adjacency_matrix)
        #get rdc_adjacency_matrix of data_insert (can be None)
        if data_insert is not None and len(data_insert) > 1:
            #print(type(data_insert), data_insert.shape)
            #print(data_insert.shape, data_insert.columns, data_insert.index)
            #print(ds_context)
            #print(node.scope)
            #print(node.condition)
            #print(rdc_sample_size)
            rdc_adjacency_matrix, scope_loc, _ = calculate_RDC(data_insert, ds_context, node.scope, node.condition, rdc_sample_size)
            #exit(-1)
        else:
            rdc_adjacency_matrix = np.zeros(node.node_error[1]['rdc_adjacency_matrix'].shape)
        #merge rdc_adjacency_matrix and that of data_insert by simple weighted mean
        rdc_adjacency_matrix = data_insert_ratio * rdc_adjacency_matrix + (1 - data_insert_ratio) * node.node_error[1]['rdc_adjacency_matrix']
        #print(rdc_adjacency_matrix)
        if wkld_attr_bound is None:
            rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0
        else:
            new_threshold = (np.power(e, -Nx*wkld_attr_adjacency_matrix+k) + b)/Ny
            new_threshold_array = new_threshold.reshape(np.prod(new_threshold.shape))
            new_threshold_stderr = np.std(new_threshold_array)
            new_threshold_mean = np.mean(new_threshold_array)
            print(new_threshold_mean, new_threshold_stderr)
            print(rdc_threshold)
            if new_threshold_mean + new_threshold_stderr >= wkld_attr_bound[2]:
                rdc_adjacency_matrix[rdc_adjacency_matrix < new_threshold] = 0
            else:
                rdc_adjacency_matrix[rdc_adjacency_matrix < rdc_threshold] = 0
        #print('rdc', rdc_adjacency_matrix)
        #exit(-1)
        #if ProductD
        if node.node_error[0]['typ'] == 'D':
            #matrix(data_insert)
            #print('calc RDC...')
            #exit(-1)
            #calc G0 and G1 of RDC to check if the RDC of node still holds
            G0 = from_numpy_matrix(node.node_error[0]['rdc_adjacency_matrix'])
            G1 = from_numpy_matrix(rdc_adjacency_matrix)
            print_graph(G0)
            print_graph(G1)
            for u,v,w in G1.edges(data='weight'):
                if not G0.has_edge(u, v):
                    rebuild_flag = True
                    break
        
        wkld_attr_adjacency_matrix[wkld_attr_adjacency_matrix < wkld_attr_threshold] = 0
        #if ProductQ
        if node.node_error[0]['typ'] == 'Q':
            #calc G0 and G1 of wkld_RDC to check if the wkld_RDC of node still holds
            G0 = from_numpy_matrix(node.node_error[0]['wkld_attr_adjacency_matrix'])
            G1 = from_numpy_matrix(wkld_attr_adjacency_matrix)
            print_graph(G0)
            print_graph(G1)
            rebuild_flag = False
            for u,v,w in G1.edges(data='weight'):
                if not G0.has_edge(u, v):
                    rebuild_flag = True
                    break
        
        #update node.node_error regardless of whether need to rebuild
        node.node_error[1]['data_n'] = data_n_sum
        node.node_error[1]['rdc_adjacency_matrix'] = rdc_adjacency_matrix
        node.node_error[1]['queries_n'] = queries_n_sum
        node.node_error[1]['wkld_attr_adjacency_matrix'] = wkld_attr_adjacency_matrix
    
    elif isinstance(node, QSum):
        #print('hello')
        #QSum node
        #encode types of new queries to maxcut_point
        new_queries_point = {}
        for i, q in enumerate(new_queries):
            pointi = ['1'] * len(node.scope)
            for j, c in enumerate(node.scope):
                if q[c][0] == float('-inf') and q[c][1] == float('inf'):
                    pointi[j] = '0'
            spointi = split_queries_by_maxcut_point_encoder(pointi)
            if spointi not in new_queries_point:
                new_queries_point[spointi] = [i]
            else:
                new_queries_point[spointi].append(i)
        #merge cutset with types of new queries
        new_queries_new_points = {}
        for i in new_queries_point:
            existed = False
            for j in node.node_error[1]['maxcut_cutset_point']:
                for k in j:
                    if k == i:
                        j[k] += len(new_queries_point[i])
                        existed = True
                        break
                if existed:
                    break
            if not existed:
                new_queries_new_points[i] = {'queries_n': len(new_queries_point[i]), 'V_weight': 0}
        #initialize E_sum_opt
        E_sum_opt = 0
        for i in node.node_error[1]['maxcut_cutset_point']:
            for j in i:
                for k in i:
                    if j != k:
                        E_sum_opt += bitset_intersectbits(j, k) * (i[j] + i[k])
        E_sum_opt //= 2
        #calc V_weight of new_queries_new_points to decide the order of greedy algorithm
        for i in new_queries_new_points:
            for j in new_queries_new_points:
                if j != i:
                    new_queries_new_points[i]['V_weight'] += bitset_intersectbits(i, j) * (new_queries_new_points[i]['queries_n'] + new_queries_new_points[j]['queries_n'])
            for j in node.node_error[1]['maxcut_cutset_point']:
                for k in j:
                    new_queries_new_points[i]['V_weight'] += bitset_intersectbits(i, k) * (new_queries_new_points[i]['queries_n'] + j[k])
        new_queries_new_points = sorted([{'V': i, 'queries_n': new_queries_new_points[i]['queries_n'], 'V_weight': new_queries_new_points[i]['V_weight']} for i in new_queries_new_points], reverse=True, key=lambda t : t['V_weight'])
        #new_queries_new_points choose cutset to add by order and calc E_sum_opt
        for i in new_queries_new_points:
            opt_j = None
            opt = None
            for j in node.node_error[1]['maxcut_cutset_point']:
                cost = 0
                for k in j:
                    cost += bitset_intersectbits(i['V'], k) * (i['queries_n'] + j[k])
                if opt is None or opt > cost:
                    opt_j = j
                    opt = cost
            opt_j[i['V']] = i['queries_n']
            E_sum_opt += opt
        #update node.cluster_centers
        node.cluster_centers = []
        for i in node.node_error[1]['maxcut_cutset_point']:
            node.cluster_centers.append(qsplit_train_cluster_encoder(node.scope, i))
        #calc E_sum
        Esum = 0
        for i1 in node.node_error[1]['maxcut_cutset_point']:
            for j1 in i1:
                for i2 in node.node_error[1]['maxcut_cutset_point']:
                    for j2 in i2:
                        if j2 != j1:
                            Esum += bitset_intersectbits(j1, j2) * (i1[j1] + i2[j2])
        Esum //= 2
        #print(E_sum_opt, Esum)
        #calc r_score, the optimization ratio of maxcut_opt
        origin_r_score = node.node_error[0]['maxcut_opt_score']
        r_score = (Esum - E_sum_opt) / max(1, Esum)
        print('r_score:', r_score)
        new_queries_slices = [[] for i in range(len(node.node_error[1]['maxcut_cutset_point']))]
        for q in new_queries_point:
            for i, c in enumerate(node.node_error[1]['maxcut_cutset_point']):
                if q in c:
                    for j in new_queries_point[q]:
                        new_queries_slices[i].append(new_queries[j])
                    break
        for i in range(len(new_queries_slices)):
            if len(new_queries_slices[i]) > 0:
                new_queries_slices[i] = np.array(new_queries_slices[i])
            else:
                new_queries_slices[i] = np.zeros((0, new_queries.shape[1], new_queries.shape[2]))
        if r_score < (1 - maxcut_err_threshold) * origin_r_score:
            rebuild_flag = True, new_queries_slices
        else:
            rebuild_flag = False, new_queries_slices
    
    elif isinstance(node, Sum):
        #Sum node
        #try to classfy data_insert by node.cluster_centers of node and check K-Means SSE
        _, vqerr = vq.vq(data_insert, node.node_error[0]['centers'])
        new_data_insert = [[] for i in range(len(node.cluster_centers))]
        for i, cci in enumerate(_):
            new_data_insert[cci].append(data_insert[i])
        for i in range(len(new_data_insert)):
            if len(new_data_insert[i]) > 0:
                new_data_insert[i] = np.array(new_data_insert[i])
                origin_data_slice_i_max = node.node_error[1]['data_max'][i]
                new_data_insert_i_max = np.max(new_data_insert[i], axis=0)
                node.node_error[1]['data_max'][i] = np.max(np.array([origin_data_slice_i_max, new_data_insert_i_max]), axis=0)
                origin_data_slice_i_min = node.node_error[1]['data_min'][i]
                new_data_insert_i_min = np.min(new_data_insert[i], axis=0)
                node.node_error[1]['data_min'][i] = np.min(np.array([origin_data_slice_i_min, new_data_insert_i_min]), axis=0)
            else:
                new_data_insert[i] = np.zeros((0, data_insert.shape[1]))
        data_insert_cluster_err = np.mean(vqerr)
        data_insert_n = data_insert.shape[0] if data_insert is not None else 0
        data_n_sum = node.node_error[1]['data_n'] + data_insert_n
        data_insert_ratio = data_insert_n / max(1, data_n_sum)
        node.node_error[1]['cluster_err'] = data_insert_ratio * data_insert_cluster_err + (1 - data_insert_ratio) * node.node_error[1]['cluster_err']
        node.node_error[1]['data_n'] = data_n_sum
        rebuild_flag = (node.node_error[1]['cluster_err'] > (1 + cluster_err_threshold) * node.node_error[0]['cluster_err']), new_data_insert
    
    print('After err_check:', node.node_error) 
    #if isinstance(node, QSum):
        #for i in new_queries_slices:
        #    print(i)
        #exit(-1)
    #total_nodes_n += 1
    if isinstance(rebuild_flag, tuple):
        if rebuild_flag[0]:
            need_rebuild_nodes.append(node) 
    else:
        if rebuild_flag:
            need_rebuild_nodes.append(node) 
    return rebuild_flag

def top_down_update(fspn, ds_context, data_insert=None, data_delete=None, new_queries=None, new_queries_all_n=None, rdc_sample_size=50000, rdc_threshold=0.3, wkld_attr_threshold=0.01, wkld_attr_bound=(0.2, 0.5)):
    global need_rebuild_nodes, total_nodes_n
    print(fspn, data_insert.shape if data_insert is not None else None, new_queries.shape if new_queries is not None else None, new_queries_all_n, fspn.scope)
    """
        Updates the FSPN when a new dataset arrives. The function recursively traverses the
        tree and inserts the different values of a dataset at the according places.
        At every sum node, the child node is selected, based on the minimal euclidian distance to the
        cluster_center of on of the child-nodes.
    """
    '''
    #fspn means node
    if fspn.range:
        if data_insert:
            assert data_insert.shape[1] == len(fspn.scope) + len(fspn.range), \
                f"mismatched data shape {data_insert.shape[1]} and {len(fspn.scope) + len(fspn.condition)}"
        if data_delete:
            assert data_delete.shape[1] == len(fspn.scope) + len(fspn.range), \
                f"mismatched data shape {data_delete.shape[1]} and {len(fspn.scope) + len(fspn.condition)}"
    else:
        if data_insert:
            assert data_insert.shape[1] == len(fspn.scope), \
                f"mismatched data shape {data_insert.shape[1]} and {len(fspn.scope)}"
        if data_delete:
            assert data_delete.shape[1] == len(fspn.scope), \
                f"mismatched data shape {data_delete.shape[1]} and {len(fspn.scope)}"
    '''
    ret = False

    #data_insert and new_queries have both been empty, which means the subtree from this node is nothing to update..
    if (data_insert is None or len(data_insert) == 0) and (new_queries is None or len(new_queries) == 0):
        return ret

    if isinstance(fspn, Leaf):
        # TODO: indepence test along with original_dataset for multi-leaf nodes
        #Leaf never needs to rebuild check, update is enough.
        update_leaf(fspn, ds_context, data_insert, data_delete)

    elif isinstance(fspn, Factorize):
        #Factorize, not in QSPN
        assert 'Factorize' is None
        left_cols = [fspn.scope.index(i) for i in fspn.children[0].scope]
        if data_insert:
            left_insert = data_insert[:, left_cols]
        else:
            left_insert = None
        if data_delete:
            left_delete = data_delete[:, left_cols]
        else:
            left_delete = None
        ret_left = top_down_update(fspn.children[0], ds_context, left_insert, left_delete, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        ret_right = top_down_update(fspn.children[1], ds_context, data_insert, data_delete, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        ret = (ret_left or ret_right)

    elif isinstance(fspn, QSum):
        # a qsum node
        origin_queries_n = fspn.queries_n
        origin_cardinality = fspn.cardinality
        #update the information of this QSum node, meanwhile check if this node needs to rebuild but not rebuild now.
        if new_queries is not None:
            ret, new_queries_slices = top_down_update_err_check(fspn, ds_context, data_insert, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            fspn.queries_n += len(new_queries)
        else:
            new_queries_slices = [None for i in range(len(fspn.children))]
        #update node.cardinality
        if data_insert is not None and len(data_insert) > 0:
            fspn.cardinality += len(data_insert)
        if data_delete is not None:
            fspn.cardinality -= len(data_delete)
        
        #dfs children
        for i, child in enumerate(fspn.children):
            child_queries_n = origin_queries_n * fspn.weights[i]
            if new_queries_slices[i] is not None:
                child_queries_n += len(new_queries_slices[i])
            fspn.weights[i] = child_queries_n / fspn.queries_n
            ret_child = top_down_update(child, ds_context, data_insert, data_delete, new_queries_slices[i], new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            ret = (ret or ret_child)

    elif isinstance(fspn, Sum) and fspn.range is not None:
        #a split node, not in QSPN
        assert fspn.cluster_centers == [], fspn
        for child in fspn.children:
            assert child.range is not None, child
            if data_insert:
                new_data_insert = split_data_by_range(data_insert, child.range, child.scope)
            else:
                new_data_insert = None
            if data_delete:
                new_data_delete = split_data_by_range(data_delete, child.range, child.scope)
            else:
                new_data_delete = None
            ret_child = top_down_update(child, ds_context, new_data_insert, new_data_delete, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            ret = (ret or ret_child)

    elif isinstance(fspn, Sum):
        # a sum node
        origin_cardinality = fspn.cardinality
        #update the information of this Sum node, meanwhile check if this node needs to rebuild but not rebuild now.
        if data_insert is not None:
            ret, new_data_insert = top_down_update_err_check(fspn, ds_context, data_insert, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            fspn.cardinality += len(data_insert)
        else:
            new_data_insert = [None for i in range(len(fspn.children))]
        if data_delete is not None:
            new_data_delete = split_data_by_cluster_center(data_delete, fspn.cluster_centers)
            fspn.cardinality -= len(data_delete)
        else:
            new_data_delete = [None for i in range(len(fspn.children))]
        
        #dfs children
        for i, child in enumerate(fspn.children):
            dl_insert = len(new_data_insert[i]) if new_data_insert[i] is not None else 0
            dl_delete = len(new_data_delete[i]) if new_data_delete[i] is not None else 0
            child_cardinality = origin_cardinality * fspn.weights[i]
            child_cardinality += dl_insert
            child_cardinality -= dl_delete
            fspn.weights[i] = child_cardinality / fspn.cardinality
            if new_queries is not None:
                new_queries_slice = get_workload_by_datadom(fspn.node_error[1]['data_min'][i], fspn.node_error[1]['data_max'][i], fspn.scope, new_queries)
            else:
                new_queries_slice = None
            ret_child = top_down_update(child, ds_context, new_data_insert[i], new_data_delete[i], new_queries_slice, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            ret = (ret or ret_child)
    
    elif isinstance(fspn, Product):
        # Product Q or D or N
        #update the information of this Product node, meanwhile check if this node needs to rebuild but not rebuild now.        
        if fspn.typ == 'N' or fspn.typ == 'RUF' or fspn.node_error is None:
            ret = False
        else:
            ret = top_down_update_err_check(fspn, ds_context, data_insert, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
        if update_performance_boost and ret:
            fspn.rebuild_f = True
        #dfs children and divide data_insert and new_queries to new_data_insert and new_queries_slices
        for child in fspn.children:
            index = [fspn.scope.index(s) for s in child.scope]
            if data_insert is not None:
                new_data_insert = data_insert[:, index]
            else:
                new_data_insert = None
            if data_delete is not None:
                new_data_delete = data_delete[:, index]
            else:
                new_data_delete = None
            if new_queries is not None:
                new_queries_slice = get_workload_by_scope(child.scope, new_queries)
            else:
                new_queries_slice = None
            #print(new_queries_slice.shape)
            ret_child = top_down_update(child, ds_context, new_data_insert, new_data_delete, new_queries_slice, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound)
            ret = (ret or ret_child)
    
    total_nodes_n += 1
    return ret


def update_leaf(fspn, ds_context, data_insert, data_delete):
    """
    update the parameter of leaf distribution, currently only support histogram.
    """
    if isinstance(fspn, Histogram):
        if data_insert is not None and len(data_insert) > 0:
            insert_leaf_Histogram(fspn, ds_context, data_insert)
        if data_delete is not None and len(data_delete) > 0:
            delete_leaf_Histogram(fspn, ds_context, data_delete)
    elif isinstance(fspn, Multi_histogram):
        if data_insert is not None and len(data_insert) > 0:
            insert_leaf_Multi_Histogram(fspn, ds_context, data_insert)
        if data_delete is not None and len(data_delete) > 0:
            delete_leaf_Multi_Histogram(fspn, ds_context, data_delete)
    elif isinstance(fspn, Merge_leaves):
        if data_insert is not None and len(data_insert) > 0:
            insert_leaf_Merge(fspn, ds_context, data_insert)
        if data_delete is not None and len(data_delete) > 0:
            delete_leaf_Merge(fspn, ds_context, data_delete)
    else:
        # TODO: implement the update of other leaf nodes
        assert False, "update of other node type is not yet implemented!!!!"


def insert_leaf_Histogram(fspn, ds_context, dataset):
    """
    Insert the new data into the original histogram and update the parameter.
    """
    if fspn.range:
        if len(fspn.scope + fspn.condition) == dataset.shape[1]:
            idx = sorted(fspn.scope + fspn.condition)
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        elif len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]:
            idx = sorted(fspn.scope + list(fspn.range.keys()))
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        else:
            assert False

    new_card = len(dataset)
    if new_card == 0:
        return
    dataset = dataset[~np.isnan(dataset)]
    new_card_actual = len(dataset)  # the cardinality without nan.
    new_nan_perc = new_card_actual / new_card

    old_card = fspn.cardinality
    fspn.cardinality = old_card + new_card
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_weight = old_card / (new_card + old_card)
    new_weight = new_card / (new_card + old_card)
    fspn.nan_perc = old_weight * fspn.nan_perc + new_weight * new_nan_perc  # update nan_perc

    if new_card_actual == 0:
        return
    old_weight = old_card_actual / (new_card_actual + old_card_actual)
    new_weight = new_card_actual / (new_card_actual + old_card_actual)

    new_breaks = list(fspn.breaks)
    left_added = False
    right_added = False
    # new value out of bound of original breaks, adding new break
    if np.min(dataset) < new_breaks[0]:
        new_breaks = [np.min(dataset) - EPSILON] + new_breaks
        left_added = True
    if np.max(dataset) > new_breaks[-1]:
        new_breaks = new_breaks + [np.max(dataset) + EPSILON]
        right_added = True

    new_pdf, new_breaks = np.histogram(dataset, bins=new_breaks)
    new_pdf = new_pdf / np.sum(new_pdf)
    old_pdf = fspn.pdf.tolist()
    if left_added:
        old_pdf = [0.0] + old_pdf
    if right_added:
        old_pdf = old_pdf + [0.0]
    old_pdf = np.asarray(old_pdf)

    assert len(new_pdf) == len(old_pdf) == len(new_breaks) - 1, "lengths mismatch"
    new_pdf = old_pdf * old_weight + new_pdf * new_weight
    new_cdf = np.zeros(len(new_pdf) + 1)
    for i in range(len(new_pdf)):
        if i == 0:
            new_cdf[i + 1] = new_pdf[i]
        else:
            new_cdf[i + 1] = new_pdf[i] + new_cdf[i]
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"
    assert np.isclose(new_cdf[-1], 1), f"incorrect cdf, with max {new_cdf[-1]}"

    fspn.breaks = new_breaks
    fspn.pdf = new_pdf
    fspn.cdf = new_cdf


def delete_leaf_Histogram(fspn, ds_context, dataset):
    """
    Insert the new data into the original histogram and update the parameter.
    """
    if fspn.range:
        if len(fspn.scope + fspn.condition) == dataset.shape[1]:
            idx = sorted(fspn.scope + fspn.condition)
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        elif len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]:
            idx = sorted(fspn.scope + list(fspn.range.keys()))
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        else:
            assert False

    new_card = len(dataset)
    if new_card == 0:
        return
    dataset = dataset[~np.isnan(dataset)]
    new_card_actual = len(dataset)  # the cardinality without nan.

    old_card = fspn.cardinality
    fspn.cardinality = old_card - new_card
    assert fspn.cardinality >= 0, f"not enough data to delete"
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_card_nan = old_card * (1-fspn.nan_perc)
    new_card_nan = new_card - new_card_actual
    fspn.nan_perc = (old_card_nan-new_card_nan) / fspn.cardinality  # update nan_perc

    if new_card_actual == 0:
        return
    delete_weight = new_card_actual / old_card_actual
    remain_weight = 1 - delete_weight

    if np.min(dataset) < fspn.breaks[0] or np.max(dataset) > fspn.breaks[-1]:
        assert False, "deleted value out of bound of original breaks"

    delete_pdf, new_breaks = np.histogram(dataset, bins=fspn.breaks)
    delete_pdf = delete_pdf / np.sum(delete_pdf)
    old_pdf = fspn.pdf

    assert len(delete_pdf) == len(old_pdf) == len(new_breaks) - 1, "lengths mismatch"
    new_pdf = (old_pdf - delete_pdf * delete_weight) / remain_weight
    assert np.sum(new_pdf < 0) == 0, f"incorrect pdf, with negative entree {new_pdf[new_pdf < 0]}"
    new_cdf = np.zeros(len(new_pdf) + 1)
    for i in range(len(new_pdf)):
        if i == 0:
            new_cdf[i + 1] = new_pdf[i]
        else:
            new_cdf[i + 1] = new_pdf[i] + new_cdf[i]
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"
    assert np.isclose(new_cdf[-1], 1), f"incorrect cdf, with max {new_cdf[-1]}"

    fspn.pdf = new_pdf
    fspn.cdf = new_cdf

def insert_leaf_Multi_Histogram(fspn, ds_context, dataset):
    """
        Insert the new data into the original multi-histogram and update the parameter.
    """
    if fspn.range:
        if len(fspn.scope + fspn.condition) == dataset.shape[1]:
            idx = sorted(fspn.scope + fspn.condition)
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        elif len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]:
            idx = sorted(fspn.scope + list(fspn.range.keys()))
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        else:
            assert False

    shape = dataset.shape
    new_card = len(dataset)
    if new_card == 0:
        return
    dataset = dataset[~np.isnan(dataset)]
    dataset = dataset.reshape(shape)
    new_card_actual = len(dataset)  # the cardinality without nan.
    new_nan_perc = new_card_actual / new_card

    old_card = fspn.cardinality
    fspn.cardinality = old_card + new_card
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_weight = old_card / (new_card + old_card)
    new_weight = new_card / (new_card + old_card)
    fspn.nan_perc = old_weight * fspn.nan_perc + new_weight * new_nan_perc  # update nan_perc

    if new_card_actual == 0:
        return
    old_weight = old_card_actual / (new_card_actual + old_card_actual)
    new_weight = new_card_actual / (new_card_actual + old_card_actual)

    new_breaks_list = list(fspn.breaks)
    left_added = [False] * len(new_breaks_list)
    right_added = [False] * len(new_breaks_list)
    assert len(new_breaks_list) == dataset.shape[1], "mismatch number of breaks and data dimension"
    for i in range(len(new_breaks_list)):
        new_breaks = list(new_breaks_list[i])
        # new value out of bound of original breaks, adding new break
        if np.min(dataset[:, i]) < new_breaks[0]:
            new_breaks = [np.min(dataset[:, i]) - EPSILON] + new_breaks
            left_added[i] = True
        if np.max(dataset[:, i]) > new_breaks[-1]:
            new_breaks = new_breaks + [np.max(dataset[:, i]) + EPSILON]
            right_added[i] = True
        new_breaks_list[i] = np.asarray(new_breaks)

    new_pdf, new_breaks_list = np.histogramdd(dataset, bins=new_breaks_list)
    new_pdf = new_pdf / np.sum(new_pdf)
    old_pdf = np.zeros(new_pdf.shape)
    assert len(new_pdf.shape) == len(new_breaks_list)
    index = []
    for i in range(len(new_pdf.shape)):
        start = 0
        end = new_pdf.shape[i]
        if left_added[i]:
            start += 1
        if right_added[i]:
            end -= 1
        index.append(slice(start, end))
    old_pdf[tuple(index)] = fspn.pdf
    new_pdf = old_pdf * old_weight + new_pdf * new_weight
    new_cdf = multidim_cumsum(new_pdf)
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"

    fspn.breaks = new_breaks_list
    fspn.pdf = new_pdf
    fspn.cdf = new_cdf

def delete_leaf_Multi_Histogram(fspn, ds_context, dataset):
    """
        Insert the new data into the original multi-histogram and update the parameter.
    """
    if fspn.range:
        if len(fspn.scope + fspn.condition) == dataset.shape[1]:
            idx = sorted(fspn.scope + fspn.condition)
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        elif len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]:
            idx = sorted(fspn.scope + list(fspn.range.keys()))
            keep = [idx.index(i) for i in fspn.scope]
            dataset = dataset[:, keep]
        else:
            assert False

    shape = dataset.shape
    new_card = len(dataset)
    if new_card == 0:
        return
    dataset = dataset[~np.isnan(dataset)]
    dataset = dataset.reshape(shape)
    new_card_actual = len(dataset)  # the cardinality without nan.

    old_card = fspn.cardinality
    fspn.cardinality = old_card - new_card
    assert fspn.cardinality >= 0, f"not enough data to delete"
    old_card_actual = old_card * fspn.nan_perc  # the cardinality without nan.
    old_card_nan = old_card * (1 - fspn.nan_perc)
    new_card_nan = new_card - new_card_actual
    fspn.nan_perc = (old_card_nan - new_card_nan) / fspn.cardinality  # update nan_perc

    if new_card_actual == 0:
        return
    delete_weight = new_card_actual / old_card_actual
    remain_weight = 1 - delete_weight

    breaks_list = list(fspn.breaks)
    assert len(breaks_list) == dataset.shape[1], "mismatch number of breaks and data dimension"
    for i in range(len(breaks_list)):
        new_breaks = list(breaks_list[i])
        if np.min(dataset[:, i]) < new_breaks[0] or np.max(dataset[:, i]) > new_breaks[-1]:
            assert False, "deleted value out of bound of original breaks"

    delete_pdf, breaks_list = np.histogramdd(dataset, bins=breaks_list)
    delete_pdf = delete_pdf / np.sum(delete_pdf)
    old_pdf = fspn.pdf
    assert delete_pdf.shape == old_pdf.shape
    new_pdf = (old_pdf - delete_pdf * delete_weight) / remain_weight
    assert np.sum(new_pdf < 0) == 0, f"incorrect pdf, with negative entree {new_pdf[new_pdf < 0]}"
    new_cdf = multidim_cumsum(new_pdf)
    assert np.isclose(np.sum(new_pdf), 1), f"incorrect pdf, with sum {np.sum(new_pdf)}"

    fspn.pdf = new_pdf
    fspn.cdf = new_cdf


def insert_leaf_Merge(fspn, ds_context, dataset):
    """
    Insert the new data into the original merge leave and update the parameter.
    """
    if fspn.range is None:
        assert len(fspn.scope) == dataset.shape[1]
        idx_all = sorted(fspn.scope)
    else:
        assert len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]
        idx_all = sorted(fspn.scope + list(fspn.range.keys()))
    for leaf in fspn.leaves:
        if leaf.range is None:
            idx = [idx_all.index(i) for i in leaf.scope]
        else:
            idx = [idx_all.index(i) for i in sorted(leaf.scope + leaf.condition)]
        if isinstance(fspn, Histogram):
            insert_leaf_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Multi_histogram):
            insert_leaf_Multi_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Merge_leaves):
            insert_leaf_Merge(fspn, ds_context, dataset[:, idx])
        else:
            assert False, "Not implemented yet"


def delete_leaf_Merge(fspn, ds_context, dataset):
    """
    Insert the new data into the original merge leave and update the parameter.
    """
    if fspn.range is None:
        assert len(fspn.scope) == dataset.shape[1]
        idx_all = sorted(fspn.scope)
    else:
        assert len(fspn.scope + list(fspn.range.keys())) == dataset.shape[1]
        idx_all = sorted(fspn.scope + list(fspn.range.keys()))
    for leaf in fspn.leaves:
        if leaf.range is None:
            idx = [idx_all.index(i) for i in leaf.scope]
        else:
            idx = [idx_all.index(i) for i in sorted(leaf.scope + leaf.condition)]
        if isinstance(fspn, Histogram):
            delete_leaf_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Multi_histogram):
            delete_leaf_Multi_Histogram(fspn, ds_context, dataset[:, idx])
        elif isinstance(fspn, Merge_leaves):
            delete_leaf_Merge(fspn, ds_context, dataset[:, idx])
        else:
            assert False, "Not implemented yet"


def split_data_by_range(dataset, rect, scope):
    """
    split the new data by the range specified by a split node
    """
    local_data = copy.deepcopy(dataset)
    attrs = list(rect.keys())
    inds = sorted(scope + attrs)
    for attr in attrs:
        lrange = rect[attr]
        if type(lrange[0]) == tuple:
            left_bound = lrange[0][0]
            right_bound = lrange[0][1]
        elif len(lrange) == 1:
            left_bound = lrange[0]
            right_bound = lrange[0]
        else:
            left_bound = lrange[0]
            right_bound = lrange[1]
        i = inds.index(attr)
        indx = np.where((left_bound <= local_data[:, i]) & (local_data[:, i] <= right_bound))[0]
        local_data = local_data[indx]
    return local_data

def split_data_by_cluster_center(dataset, center, seed=17):
    """
    split the new data based on kmeans center
    """
    k = len(center)
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.cluster_centers_ = np.asarray(center)
    cluster = kmeans.predict(dataset)
    res = []
    for i in np.sort(np.unique(cluster)):
        local_data = dataset[cluster == i, :]
        res.append(local_data)
    return res

'''
notice:
node.node_error[0] contains information of the original node without update, which is used to compare error
node.node_error[1] contains information of currrent node, which is used to update incrementally and describe the current status of the node
'''
def update_QSPN(root, data, workload, data_insert=None, data_delete=None, new_queries=None, rdc_sample_size=50000, rdc_threshold=0.3, wkld_attr_threshold=0.01, wkld_attr_bound=(0.2, 0.5)):
    if data is not None and len(data) > 0:
        sample_data = data.values.astype(int)
    else:
        sample_data = None
    if data_insert is not None and len(data_insert) > 0:
        sample_data_insert = data_insert.values.astype(int)
    else:
        sample_data_insert = None
    #exit(-1)
    #print(root.weights)
    #exit(-1)
    if data_insert is not None and len(data_insert) > 0:
        updated_data = np.concatenate((sample_data, sample_data_insert), axis=0)
    else:
        updated_data = sample_data
    if new_queries is not None and len(new_queries) > 0:
        updated_workload = np.concatenate((workload, new_queries), axis=0)
    else:
        updated_workload = workload
    parametric_types = [Categorical for i in range(len(data.columns))]
    ds_context = Context(parametric_types=parametric_types).add_domains(updated_data)
    new_queries_all_n = len(new_queries) if new_queries is not None else None

    upd_start = perf_counter()
    #Step1: try top_down_update (incremental update), and judge if the tree need to subtree-rebuild (cannot locate which nodes to rebuild)
    need_rebuild_nodes.clear()
    if top_down_update(root, ds_context, sample_data_insert, data_delete, new_queries, new_queries_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound):
        print('need rebuild(estimate)')
        print(len(need_rebuild_nodes), '/', total_nodes_n)
        print([i for i in need_rebuild_nodes])
        need_rebuild_nodes.clear()
        #print(root.cardinality)
        #print(qsum_nodes)
        #for i in qsum_nodes:
        #    print(i.scope)
        #    print(i.node_error)
        #exit(-1)
        #Step2: if need rebuild some subtrees
        #union data and data_insert, union workload and new_queries
        #print(type(updated_data), updated_data.shape)
        #print(updated_data[0])
        #print(updated_data[-1])
        #exit(-1)
        #rebuild subtree by dfs checking
        workload_all_n = len(updated_workload)
        if top_down_adaptive_rebuild(root, ds_context, updated_data, updated_workload, workload_all_n, rdc_sample_size, rdc_threshold, wkld_attr_threshold, wkld_attr_bound):
            #if the whole tree need to rebuild
            print('Rebuild {}...'.format(root))
            need_rebuild_nodes.append(root)
            #print(len(need_rebuild_nodes), '/', total_nodes_n)
            #print(need_rebuild_nodes)
            #exit(-1)
            new_root = learn_FSPN(
                        updated_data,
                        ds_context,
                        workload=updated_workload,
                        queries='kmeans',
                        rdc_sample_size=rdc_sample_size,
                        rdc_strong_connection_threshold=1.1,
                        multivariate_leaf=False,
                        threshold=rdc_threshold,
                        wkld_attr_threshold=wkld_attr_threshold,
                        wkld_attr_bound=wkld_attr_bound,
                        updateQSPN_workload_all_n=workload_all_n
                    )
            root = new_root
    upd_end = perf_counter()
    print(len(need_rebuild_nodes), '/', total_nodes_n)
    print(need_rebuild_nodes)
    print(get_structure_stats(root))
    print("updateQSPN cost: {:.5f} secs".format(upd_end-upd_start))
    #print('Press ENTER to continue...')
    #input()
    #exit(-1)
    return root