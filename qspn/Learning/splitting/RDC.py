import numpy as np
from sklearn.cluster import KMeans

from Learning.splitting.Base import split_data_by_clusters, clusters_by_adjacency_matrix
import logging

import itertools

from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
import scipy.stats

from sklearn.cross_decomposition import CCA
from Structure.StatisticalTypes import MetaType

logger = logging.getLogger(__name__)


CCA_MAX_ITER = 100

L_NO_OHE = True


def ecdf(X):
    """
    Empirical cumulative distribution function
    for data X (one dimensional, if not it is linearized first)
    """

    mv_ids = np.isnan(X)

    N = X.shape[0]
    X = X[~mv_ids]
    R = scipy.stats.rankdata(X, method="max") / len(X)
    X_r = np.zeros(N)
    X_r[~mv_ids] = R
    return X_r


def empirical_copula_transformation(data):
    ones_column = np.ones((data.shape[0], 1))
    data = np.concatenate((np.apply_along_axis(ecdf, 0, data), ones_column), axis=1)
    return data


def make_matrix(data):
    """
    Ensures data to be 2-dimensional
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        assert data.ndim == 2, "Data must be 2 dimensional {}".format(data.shape)

    return data


def ohe_data(data, domain):
    dataenc = np.zeros((data.shape[0], len(domain)))
    #print(f'dataenc={dataenc.shape}')
    #return dataenc
    
    dataenc[data[:, None] == domain[None, :]] = 1
    #return dataenc

    #
    # this control fails when having missing data as nans
    if not np.any(np.isnan(data)):
        assert np.all((np.nansum(dataenc, axis=1) == 1)), "one hot encoding bug {} {} {}".format(
            domain, data, np.nansum(dataenc, axis=1)
        )

    return dataenc


def rdc_transformer(
    local_data,
    meta_types,
    domains,
    k=None,
    s=1.0 / 6.0,
    non_linearity=np.sin,
    return_matrix=False,
    ohe=True,
    rand_gen=None,
):
    """
    Given a data_slice,
    return a transformation of the features data in it according to the rdc
    pipeline:
    1 - empirical copula transformation
    2 - random projection into a k-dimensional gaussian space
    3 - pointwise  non-linear transform
    """

    N, D = local_data.shape
    print(f'N={N}, D={D}')
    #exit(-1)


    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    #
    # precomputing transformations to reduce time complexity
    #

    #
    # FORCING ohe on all discrete features
    features = []
    #print(meta_types)
    #exit(-1)
    for f in range(D):
        #print(f)
        #print(local_data[:, f].reshape(-1, 1))
        #print(ohe_data(local_data[:, f], domains[f]))
        #features.append(local_data[:, f].reshape(-1, 1))
        #continue
        if not L_NO_OHE and meta_types[f] == MetaType.DISCRETE:
            features.append(ohe_data(local_data[:, f], domains[f]))
        else:
            features.append(local_data[:, f].reshape(-1, 1))
    # else:
    #     features = [data_slice.getFeatureData(f) for f in range(D)]
    #exit(-1)
    #
    # NOTE: here we are setting a global k for ALL features
    # to be able to precompute gaussians
    #print(k)
    if k is None:
        feature_shapes = [f.shape[1] if len(f.shape) > 1 else 1 for f in features]
        #print(f'feature_shapes={feature_shapes}')
        k = max(feature_shapes) + 1
    #print(f'k={k}')
    #exit(-1)
    #
    # forcing two columness
    features = [make_matrix(f) for f in features]
    #print([i.shape for i in features])
    #exit(-1)

    #
    # transform through the empirical copula
    features = [empirical_copula_transformation(f) for f in features]
    #print(features)

    #
    # substituting nans with zero (the above step should have taken care of that)
    features = [np.nan_to_num(f) for f in features]

    #
    # random projection through a gaussian
    random_gaussians = [rand_gen.normal(size=(f.shape[1], k)) for f in features]

    rand_proj_features = [s / f.shape[1] * np.dot(f, N) for f, N in zip(features, random_gaussians)]

    nl_rand_proj_features = [non_linearity(f) for f in rand_proj_features]

    #
    # apply non-linearity
    if return_matrix:
        return np.concatenate(nl_rand_proj_features, axis=1)

    else:
        return [np.concatenate((f, np.ones((f.shape[0], 1))), axis=1) for f in nl_rand_proj_features]


def rdc_cca(indexes):
    i, j, rdc_features = indexes
    #print(i, j)
    cca = CCA(n_components=1, max_iter=CCA_MAX_ITER)
    X_cca, Y_cca = cca.fit_transform(rdc_features[i], rdc_features[j])
    #exit(-1)
    rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    #print(i, X_cca, j, Y_cca, rdc)
    #0print()
    # logger.info(i, j, rdc)
    return rdc


def rdc_test(local_data, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-1, rand_gen=None):
    n_features = local_data.shape[1]

    rdc_features = rdc_transformer(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, return_matrix=False, rand_gen=rand_gen
    )

    #print('rdc_features:', [i.shape for i in rdc_features])
    #print(rdc_features)
    #exit(-1)

    pairwise_comparisons = list(itertools.combinations(np.arange(n_features), 2))
    #print(n_features)
    #print(pairwise_comparisons)
    #if n_features < 8:
    #    exit(-1)

    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024, backend="threading")(
        delayed(rdc_cca)((i, j, rdc_features)) for i, j in pairwise_comparisons
    )
    #print('RDC finished')
    #exit(-1)

    rdc_adjacency_matrix = np.zeros((n_features, n_features))
    
    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    #
    # setting diagonal to 1
    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    return rdc_adjacency_matrix


def getIndependentRDCGroups_py(
    local_data, threshold, meta_types, domains, k=None, s=1.0 / 6.0, non_linearity=np.sin, n_jobs=-2, rand_gen=None
):
    rdc_adjacency_matrix = rdc_test(
        local_data, meta_types, domains, k=k, s=s, non_linearity=non_linearity, n_jobs=n_jobs, rand_gen=rand_gen
    )

    #
    # Why is this necessary?
    #
    rdc_adjacency_matrix[np.isnan(rdc_adjacency_matrix)] = 0
    n_features = local_data.shape[1]

    #
    # thresholding
    rdc_adjacency_matrix[rdc_adjacency_matrix < threshold] = 0
    # logger.info("thresholding %s", rdc_adjacency_matrix)

    #
    # getting connected components
    result = np.zeros(n_features)
    for i, c in enumerate(connected_components(from_numpy_matrix(rdc_adjacency_matrix))):
        result[list(c)] = i + 1

    return result


def get_split_cols_RDC_py(threshold=0.3, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None,
                          max_sampling_threshold_cols=10000):
    def split_cols_RDC_py(local_data, ds_context, scope, clusters=None):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        if local_data.shape[0] > max_sampling_threshold_cols:
            local_data_sample = local_data[np.random.randint(local_data.shape[0], size=max_sampling_threshold_cols), :]
            #print(clusters)
            if clusters is None:
                clusters = getIndependentRDCGroups_py(
                    local_data_sample,
                    threshold,
                    meta_types,
                    domains,
                    k=k,
                    s=s,
                    # ohe=True,
                    non_linearity=non_linearity,
                    n_jobs=n_jobs,
                    rand_gen=rand_gen,
                )
            #print(clusters)
            #exit(-1)
            return split_data_by_clusters(local_data, clusters, scope, rows=False)
        else:
            if clusters is None:
                clusters = getIndependentRDCGroups_py(
                    local_data,
                    threshold,
                    meta_types,
                    domains,
                    k=k,
                    s=s,
                    # ohe=True,
                    non_linearity=non_linearity,
                    n_jobs=n_jobs,
                    rand_gen=rand_gen,
                )
            return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_RDC_py


def get_split_rows_RDC_py(n_clusters=2, ohe=True, k=10, s=1 / 6, non_linearity=np.sin, n_jobs=-2, rand_gen=None):
    def split_rows_RDC_py(local_data, ds_context, scope):
        meta_types = ds_context.get_meta_types_by_scope(scope)
        domains = ds_context.get_domains_by_scope(scope)

        rdc_data = rdc_transformer(
            local_data,
            meta_types,
            domains,
            k=k,
            s=s,
            non_linearity=non_linearity,
            return_matrix=True,
            rand_gen=rand_gen,
        )

        clusters = KMeans(n_clusters=n_clusters, random_state=rand_gen, n_jobs=n_jobs).fit_predict(rdc_data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_RDC_py