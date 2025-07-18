import numpy as np
import collections
from collections import deque, OrderedDict
from Learning.utils import convert_to_scope_domain
import logging
from Learning.splitting.Workload import preproc_queries

logger = logging.getLogger(__name__)


class Node(object):
    def __init__(self):
        self.id = 0
        self.scope = []
        #self.scope_idx = []
        self.condition = []
        #self.condition_idx = []
        self.range = dict()
        self.node_error = None

    @property
    def name(self):
        return f"{self.__class__.__name__}Node_{self.id}"

    @property
    def parameters(self):
        raise Exception("Not Implemented")

    def __repr__(self):
        return self.name

    def __rmul__(self, weight):
        assert type(weight) == int or type(weight) == float
        self._tmp_weight = weight
        return self

    def __mul__(self, node):
        assert isinstance(node, Node)
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert len(set(node.scope).intersection(set(self.scope))) == 0, "children's scope is not disjoint"
        assert set(node.condition) == set(self.condition), "condition not matched, should use factorized nodes"
        result = Product()
        result.children.append(self)
        result.children.append(node)
        result.scope.extend(self.scope)
        result.scope.extend(node.scope)
        result.condition.extend(self.condition)
        #_, result.scope_idx, result.condition_idx = convert_to_scope_domain(result.scope, result.condition)
        assign_ids(result)
        return result

    def __add__(self, node):
        assert isinstance(node, Node)
        assert hasattr(node, "_tmp_weight"), "right node has no weight"
        assert hasattr(self, "_tmp_weight"), "left node has no weight"
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert set(node.scope) == (set(self.scope)), "children's scope are not the same"
        assert set(node.condition) == (set(self.condition)), "children's condition are not the same"

        from numpy import isclose

        assert isclose(
            1.0, self._tmp_weight + node._tmp_weight
        ), "unnormalized weights, maybe trying to add many nodes at the same time?"

        result = Sum()
        result.children.append(self)
        result.weights.append(self._tmp_weight)
        result.children.append(node)
        result.weights.append(node._tmp_weight)
        result.scope.extend(self.scope)
        result.condition.extend(self.condition)
        result._tmp_weight = self._tmp_weight + node._tmp_weight
        #_, result.scope_idx, result.condition_idx = convert_to_scope_domain(result.scope, result.condition)
        assign_ids(result)
        return result

    def factor_mul(self, node):
        assert isinstance(node, Node)
        assert len(node.scope) > 0, "right node has no scope"
        assert len(self.scope) > 0, "left node has no scope"
        assert len(set(node.scope).intersection(set(self.scope))) == 0, "children's scope is not disjoint"
        assert set(node.condition) == set(self.scope+self.condition), "scope does not match with others' condition"

        result = Factorize()
        result.children.append(self)
        result.left_child = self
        result.children.append(node)
        result.right_child = self
        result.scope.extend(self.scope)
        result.scope.extend(node.scope)
        result.condition.extend(self.condition)
        #_, result.scope_idx, result.condition_idx = convert_to_scope_domain(result.scope, result.condition)
        assign_ids(result)
        return result


class Sum(Node):
    def __init__(self, weights=None, children=None, cluster_centers=None, cardinality=None):
        Node.__init__(self)
        if weights is None:
            weights = []
        self.weights = weights

        if children is None:
            children = []
        self.children = children

        if cluster_centers is None:
            cluster_centers = []
        self.cluster_centers = cluster_centers

        if cardinality is None:
            cardinality = 0
        self.cardinality = cardinality
        self.cluster_domains = None
        #_, self.scope_idx, self.condition_idx = convert_to_scope_domain(self.scope, self.condition)

    @property
    def parameters(self):
        sorted_children = sorted(self.children, key=lambda c: c.id)
        params = [(n.id, self.weights[i]) for i, n in enumerate(sorted_children)]
        return tuple(params)

class QSum(Sum):
    def __init__(self, weights=None, children=None, cluster_centers=None, cardinality=None):
        Sum.__init__(self, weights, children, cluster_centers, cardinality)
        self.queries_n = None

class Product(Node):
    def __init__(self, children=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children
        self.cardinality = None
        self.typ = None
        #self.qdcorr = None
    
    @property
    def parameters(self):
        return tuple(map(lambda n: n.id, sorted(self.children, key=lambda c: c.id)))


class Factorize(Node):
    def __init__(self, children=None):
        Node.__init__(self)
        if children is None:
            children = []
        self.children = children

    @property
    def parameters(self):
        #return (self.left_child.id, self.right_child.id)
        return tuple(self.children)


class Leaf(Node):
    def __init__(self, scope=None, condition=None, cardinality=0):
        Node.__init__(self)
        self.join_buckets = None
        #self.nonnan_cardinality = None
        if scope is not None:
            if type(scope) == int:
                self.scope.append(scope)
            elif type(scope) == list:
                self.scope.extend(scope)
            else:
                raise Exception("invalid scope type %s " % (type(scope)))

        if condition is not None:
            if type(condition) == int:
                self.condition.append(condition)
            elif type(condition) == list:
                self.condition.extend(condition)
            else:
                raise Exception("invalid condition type %s " % (type(condition)))
        self.cardinality = cardinality

        _, self.scope_idx, self.condition_idx = convert_to_scope_domain(self.scope, self.condition)

    def query(self, query, attr):
        raise NotImplemented

    def likelihood(self, data, attr, log):
        pass



class Context:
    def __init__(self, meta_types=None, domains=None, parametric_types=None, feature_names=None):
        self.meta_types = meta_types
        self.domains = domains
        self.parametric_types = parametric_types
        self.feature_names = feature_names
        self.fanout_attr = []
        self.fanout_attr_positive = []
        self.fanout_attr_inverse = []

        if meta_types is None and parametric_types is not None:
            self.meta_types = []
            for p in parametric_types:
                self.meta_types.append(p.type.meta_type)

    def get_meta_types_by_scope(self, scopes):
        return [self.meta_types[s] for s in scopes]

    def get_domains_by_scope(self, scopes):
        return [self.domains[s] for s in scopes]

    def get_parametric_types_by_scope(self, scopes):
        return [self.parametric_types[s] for s in scopes]

    def add_domains(self, data):
        assert len(data.shape) == 2, "data is not 2D?"
        assert data.shape[1] == len(self.meta_types), "Data columns and metatype size doesn't match"

        from Structure.StatisticalTypes import MetaType

        domain = []

        for col in range(data.shape[1]):
            feature_meta_type = self.meta_types[col]
            min_val = np.nanmin(data[:, col])
            max_val = np.nanmax(data[:, col])
            domain_values = [min_val, max_val]

            if feature_meta_type == MetaType.REAL:
                domain.append(domain_values)
            elif feature_meta_type == MetaType.BINARY:
                domain.append([0, 1])
            elif feature_meta_type == MetaType.DISCRETE:
                domain.append(np.sort(np.unique(data[:, col])))
            else:
                raise Exception("Unkown MetaType " + str(feature_meta_type))

        self.domains = np.asanyarray(domain)

        return self

def get_number_of_edges(node):
    return sum([len(c.children) for c in get_nodes_by_type(node, (Sum, Product))])


def get_number_of_nodes(spn, node_type=Node):
    return len(get_nodes_by_type(spn, node_type))


def get_parents(node, includ_pos=True):
    parents = OrderedDict({node: []})
    for n in get_nodes_by_type(node):
        if not isinstance(n, Leaf):
            for i, c in enumerate(n.children):
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                if includ_pos:
                    parent_list.append((n, i))
                else:
                    parent_list.append(n)
    return parents


def get_depth(node):
    node_depth = {}

    def count_layers(node):
        ndepth = node_depth.setdefault(node, 1)

        if hasattr(node, "children"):
            for c in node.children:
                node_depth.setdefault(c, ndepth + 1)

    bfs(node, count_layers)

    return max(node_depth.values())


def rebuild_scopes_bottom_up(node):
    # this function is not safe (updates in place)

    for n in get_topological_order(node):
        if isinstance(n, Leaf):
            continue

        new_scope = set()
        for c in n.children:
            new_scope.update(c.scope)
        n.scope = list(new_scope)

    return node


def bfs(root, func):
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)

def bfs_print(root, func):
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        #func(node)
        if not isinstance(node, Leaf):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    func(node, c)
                    queue.append(c)

def qspn_bfs(root, query, func):
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        children = func(node, query)
        for c in children:
            if c not in seen:
                seen.add(c)
                queue.append(c)

def get_topological_order(node):
    nodes = get_nodes_by_type(node)

    parents = OrderedDict({node: []})
    in_degree = OrderedDict()
    for n in nodes:
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    S = deque()  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            S.appendleft(u)

    L = []  # Empty list that will contain the sorted elements

    while S:
        n = S.pop()  # remove a node n from S
        L.append(n)  # add n to tail of L

        for m in parents[n]:  # for each node m with an edge e from n to m do
            in_degree_m = in_degree[m] - 1  # remove edge e from the graph
            in_degree[m] = in_degree_m
            if in_degree_m == 0:  # if m has no other incoming edges then
                S.appendleft(m)  # insert m into S

    assert len(L) == len(nodes), "Graph is not DAG, it has at least one cycle"
    return L

def qspn_get_topological_order(node, query):
    nodes = qspn_get_nodes_by_type(node, query)
    #print(nodes)
    #print()

    parents = OrderedDict({node: []})
    in_degree = OrderedDict()
    childrens = OrderedDict()
    for i in nodes:
        n = i[0]
        children = i[1]
        childrens[n] = children
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, Leaf):
            for c in children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    S = deque()  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            S.appendleft(u)

    L = []  # Empty list that will contain the sorted elements

    while S:
        n = S.pop()  # remove a node n from S
        L.append((n, childrens[n]))  # add n to tail of L

        for m in parents[n]:  # for each node m with an edge e from n to m do
            in_degree_m = in_degree[m] - 1  # remove edge e from the graph
            in_degree[m] = in_degree_m
            if in_degree_m == 0:  # if m has no other incoming edges then
                S.appendleft(m)  # insert m into S

    assert len(L) == len(nodes), "Graph is not DAG, it has at least one cycle"
    #print(L)
    return L

def get_topological_order_layers(node):
    nodes = get_nodes_by_type(node)

    parents = OrderedDict({node: []})
    in_degree = OrderedDict()
    for n in nodes:
        in_degree[n] = in_degree.get(n, 0)
        if not isinstance(n, Leaf):
            for c in n.children:
                parent_list = parents.get(c, None)
                if parent_list is None:
                    parents[c] = parent_list = []
                parent_list.append(n)
                in_degree[n] += 1

    layer = []  # Set of all nodes with no incoming edge
    for u in in_degree:
        if in_degree[u] == 0:
            layer.append(u)

    L = [layer]  # add first layer

    added_nodes = len(layer)
    while True:
        layer = []

        for n in L[-1]:
            for m in parents[n]:  # for each node m with an edge e from n to m do
                in_degree_m = in_degree[m] - 1  # remove edge e from the graph
                in_degree[m] = in_degree_m
                if in_degree_m == 0:  # if m has no other incoming edges then
                    layer.append(m)  # insert m into layer

        if len(layer) == 0:
            break

        added_nodes += len(layer)
        L.append(layer)

    assert added_nodes == len(nodes), "Graph is not DAG, it has at least one cycle"
    return L


def get_nodes_by_type(node, ntype=Node):
    assert node is not None

    result = []

    def add_node(node):
        if isinstance(node, ntype):
            result.append(node)

    bfs(node, add_node)

    return result

def print_spn_structure(root, ntype=Node):
    assert root is not None

    result = []

    def add_edge(fa, node):
        if isinstance(node, ntype) and isinstance(fa, ntype):
            result.append('{}{}->{}{}'.format(fa,fa.scope,node,node.scope))

    bfs_print(root, add_edge)

    return result

def qsum_which_child_dis(node, child_i, vec_q):
    return np.linalg.norm(node.cluster_centers[child_i] - vec_q[0])

def qsum_which_child(node, query):
    array_q = np.dstack(query)
    #array_q = array_q.reshape(1, array_q.shape[0], array_q.shape[1])
    vec_q = preproc_queries(array_q, node.scope)
    x = []
    min_dis = None
    for i in range(len(node.children)):
        c = node.children[i]
        dis = qsum_which_child_dis(node, i, vec_q)
        if min_dis is None or dis < min_dis:
            min_dis = dis
            x = [c]
    assert len(x) == 1
    return x

QSPLIT_SCORE_RATIO = 0.0
def qsplit_maxcut_cover_score(node, i, query):
    #query predicates on node.scope
    query_pred = [1] * len(node.scope)
    pred_n = len(query[0])
    assert pred_n == len(query[1])
    query_pred_one_n = 0
    for j, c in enumerate(node.scope):
        if query[0][0, c] == float('-inf') and query[1][0, c] == float('inf'):
            query_pred[j] = 0
        else:
            query_pred_one_n += 1
    #exact: max match_cnt with one of node.cluster_centers[i]
    #fuzzy: match_cnt with union of node.cluster_centers[i]
    exact_score = 0
    fuzzy_match = [0] * len(query_pred)
    for j in node.cluster_centers[i]:
        match_cnt = 0
        for k, c in enumerate(query_pred):
            if c == 1:
                if j[k] == 1:
                    match_cnt += 1
                    fuzzy_match[k] = 1
        exact_score = max(exact_score, match_cnt)
    fuzzy_score = 0
    for j in fuzzy_match:
        fuzzy_score += j
    #print(i, exact_score, fuzzy_score)
    if exact_score == query_pred_one_n:
        return exact_score
    else:
        return QSPLIT_SCORE_RATIO * exact_score + (1 - QSPLIT_SCORE_RATIO) * fuzzy_score

def qsplit_maxcut_which_child(node, query):
    x = []
    opt_score = -1
    for i in range(len(node.children)):
        c = node.children[i]
        score = qsplit_maxcut_cover_score(node, i, query)
        if score > opt_score:
            opt_score = score
            x = [c]
    assert len(x) == 1
    # query_pred = [1] * len(node.scope)
    # pred_n = len(query[0])
    # assert pred_n == len(query[1])
    # for j, c in enumerate(node.scope):
    #     if query[0][0, c] == float('-inf') and query[1][0, c] == float('inf'):
    #         query_pred[j] = 0
    # if len(node.scope) > 5:
    #     if opt_score < sum(query_pred):
    #         print(query_pred, opt_score, 'NOT cover')
    #     else:
    #         print(query_pred, opt_score)
    return x

def qsplit_maxcut_which_childi(node, query):
    x = []
    opt_score = -1
    for i in range(len(node.children)):
        score = qsplit_maxcut_cover_score(node, i, query)
        if score > opt_score:
            opt_score = score
            x = [i]
    assert len(x) == 1
    return x

def qspn_get_nodes_by_type(node, query, ntype=Node):
    assert node is not None

    result = []

    def AddNode_toVisitChildren(node, query):
        children = []
        if not isinstance(node, Leaf):
            if isinstance(node, QSum):
                #print(node.children)
                #print(node.weights)
                #print(node.scope)
                #print(node.cluster_centers)
                #exit(-1)
                assert len(node.cluster_centers) >= 2
                assert len(query[0]) == len(query[1]) == 1
                if type(node.cluster_centers[0]) == list:
                    children = qsplit_maxcut_which_child(node, query)
                else:
                    children = qsum_which_child(node, query)
                #print(children)
                #exit(-1)
            else:
                children = node.children
        if isinstance(node, ntype):
            result.append((node, children))
        return children
    #print(result)
    qspn_bfs(node, query, AddNode_toVisitChildren)
    #print(result)
    #exit(-1)

    return result

def get_node_types(node, ntype=Node):
    assert node is not None

    result = set()

    def add_node(node):
        if isinstance(node, ntype):
            result.add(type(node))

    bfs(node, add_node)

    return result


def assign_ids(node, ids=None):
    if ids is None:
        ids = {}

    def assign_id(node):
        if node not in ids:
            ids[node] = len(ids)

        node.id = ids[node]

    bfs(node, assign_id)
    return node


def eval_spn_bottom_up(node, eval_functions, all_results=None, debug=False, **args):
    """
    Evaluates the spn bottom up
    :param node: spn root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, args**) for leave nodes and (node, [children results], args**)
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node. It is used to store intermediate results so that non-tree graphs can be computed in O(n) size of the network
    :param debug: whether to present progress information on the evaluation
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values through the network
    """

    nodes = get_topological_order(node)

    if debug:
        from tqdm import tqdm

        nodes = tqdm(list(nodes))

    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)
        node_type._is_leaf = issubclass(node_type, Leaf)
    leaf_func = eval_functions.get(Leaf, None)

    tmp_children_list = []
    len_tmp_children_list = 0
    for n in nodes:
        try:
            func = n.__class__._eval_func[-1]
            n_is_leaf = n.__class__._is_leaf
        except:
            if isinstance(n, Leaf) and leaf_func is not None:
                func = leaf_func
                n_is_leaf = True
            else:
                raise AssertionError("No lambda function associated with type: %s" % (n.__class__.__name__))

        if n_is_leaf:
            result = func(n, **args)
        else:
            len_children = len(n.children)
            if len_tmp_children_list < len_children:
                tmp_children_list.extend([None] * len_children)
                len_tmp_children_list = len(tmp_children_list)
            for i in range(len_children):
                ci = n.children[i]
                tmp_children_list[i] = all_results[ci]
            result = func(n, tmp_children_list[0:len_children], **args)

        all_results[n] = result

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results[node]


def eval_spn_top_down(root, eval_functions, all_results=None, parent_result=None, **args):
    """
    evaluates an spn top to down
    :param root: spnt root
    :param eval_functions: is a dictionary that contains k:Class of the node, v:lambda function that receives as parameters (node, [parent_results], args**) and returns {child : intermediate_result}. This intermediate_result will be passed to child as parent_result. If intermediate_result is None, no further propagation occurs
    :param all_results: is a dictionary that contains k:Class of the node, v:result of the evaluation of the lambda function for that node.
    :param parent_result: initial input to the root node
    :param args: free parameters that will be fed to the lambda functions.
    :return: the result of computing and propagating all the values throught the network
    """
    if all_results is None:
        all_results = {}
    else:
        all_results.clear()

    for node_type, func in eval_functions.items():
        if "_eval_func" not in node_type.__dict__:
            node_type._eval_func = []
        node_type._eval_func.append(func)

    all_results[root] = [parent_result]

    for layer in reversed(get_topological_order_layers(root)):
        for n in layer:
            func = n.__class__._eval_func[-1]

            param = all_results[n]
            result = func(n, param, **args)

            if result is not None and not isinstance(n, Leaf):
                assert isinstance(result, dict)

                for child, param in result.items():
                    if child not in all_results:
                        all_results[child] = []
                    all_results[child].append(param)

    for node_type, func in eval_functions.items():
        del node_type._eval_func[-1]
        if len(node_type._eval_func) == 0:
            delattr(node_type, "_eval_func")

    return all_results[root]
