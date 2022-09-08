"""
Algorithms for calculating min/max spanning trees/forests.

"""
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue

import networkx as nx
import numpy as np
from networkx.utils import UnionFind, not_implemented_for, py_random_state

__all__ = [
    "minimum_spanning_edges",
    "maximum_spanning_edges",
    "minimum_spanning_tree",
    "maximum_spanning_tree",
    "random_spanning_tree",
    "partition_spanning_tree",
    "EdgePartition",
    "SpanningTreeIterator",
]


class EdgePartition(Enum):
    """
    An enum to store the state of an edge partition. The enum is written to the
    edges of a graph before being pasted to `kruskal_mst_edges`. Options are:

    - EdgePartition.OPEN
    - EdgePartition.INCLUDED
    - EdgePartition.EXCLUDED
    """

    OPEN = 0
    INCLUDED = 1
    EXCLUDED = 2


@not_implemented_for("multigraph")
def boruvka_mst_edges(
    G, minimum=True, weight="weight", keys=False, data=True, ignore_nan=False
):
    """Iterate over edges of a Borůvka's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The edges of `G` must have distinct weights,
        otherwise the edges may not form a tree.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        This argument is ignored since this function is not
        implemented for multigraphs; it exists only for consistency
        with the other minimum spanning tree functions.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    # Initialize a forest, assuming initially that it is the discrete
    # partition of the nodes of the graph.
    forest = UnionFind(G)

    def best_edge(component):
        """Returns the optimum (minimum or maximum) edge on the edge
        boundary of the given set of nodes.

        A return value of ``None`` indicates an empty boundary.

        """
        sign = 1 if minimum else -1
        minwt = float("inf")
        boundary = None
        for e in nx.edge_boundary(G, component, data=True):
            wt = e[-1].get(weight, 1) * sign
            if isnan(wt):
                if ignore_nan:
                    continue
                msg = f"NaN found as an edge weight. Edge {e}"
                raise ValueError(msg)
            if wt < minwt:
                minwt = wt
                boundary = e
        return boundary

    # Determine the optimum edge in the edge boundary of each component
    # in the forest.
    best_edges = (best_edge(component) for component in forest.to_sets())
    best_edges = [edge for edge in best_edges if edge is not None]
    # If each entry was ``None``, that means the graph was disconnected,
    # so we are done generating the forest.
    while best_edges:
        # Determine the optimum edge in the edge boundary of each
        # component in the forest.
        #
        # This must be a sequence, not an iterator. In this list, the
        # same edge may appear twice, in different orientations (but
        # that's okay, since a union operation will be called on the
        # endpoints the first time it is seen, but not the second time).
        #
        # Any ``None`` indicates that the edge boundary for that
        # component was empty, so that part of the forest has been
        # completed.
        #
        # TODO This can be parallelized, both in the outer loop over
        # each component in the forest and in the computation of the
        # minimum. (Same goes for the identical lines outside the loop.)
        best_edges = (best_edge(component) for component in forest.to_sets())
        best_edges = [edge for edge in best_edges if edge is not None]
        # Join trees in the forest using the best edges, and yield that
        # edge, since it is part of the spanning tree.
        #
        # TODO This loop can be parallelized, to an extent (the union
        # operation must be atomic).
        for u, v, d in best_edges:
            if forest[u] != forest[v]:
                if data:
                    yield u, v, d
                else:
                    yield u, v
                forest.union(u, v)


def kruskal_mst_edges(
    G, minimum, weight="weight", keys=True, data=True, ignore_nan=False, partition=None
):
    """
    Iterate over edge of a Kruskal's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        If `G` is a multigraph, `keys` controls whether edge keys ar yielded.
        Otherwise `keys` is ignored.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    partition : string (default: None)
        The name of the edge attribute holding the partition data, if it exists.
        Partition data is written to the edges using the `EdgePartition` enum.
        If a partition exists, all included edges and none of the excluded edges
        will appear in the final tree. Open edges may or may not be used.

    Yields
    ------
    edge tuple
        The edges as discovered by Kruskal's method. Each edge can
        take the following forms: `(u, v)`, `(u, v, d)` or `(u, v, k, d)`
        depending on the `key` and `data` parameters
    """
    subtrees = UnionFind()
    if G.is_multigraph():
        edges = G.edges(keys=True, data=True)
    else:
        edges = G.edges(data=True)

    """
    Sort the edges of the graph with respect to the partition data. 
    Edges are returned in the following order:

    * Included edges
    * Open edges from smallest to largest weight
    * Excluded edges
    """
    included_edges = []
    open_edges = []
    for e in edges:
        d = e[-1]
        wt = d.get(weight, 1)
        if isnan(wt):
            if ignore_nan:
                continue
            raise ValueError(f"NaN found as an edge weight. Edge {e}")

        edge = (wt,) + e
        if d.get(partition) == EdgePartition.INCLUDED:
            included_edges.append(edge)
        elif d.get(partition) == EdgePartition.EXCLUDED:
            continue
        else:
            open_edges.append(edge)

    if minimum:
        sorted_open_edges = sorted(open_edges, key=itemgetter(0))
    else:
        sorted_open_edges = sorted(open_edges, key=itemgetter(0), reverse=True)

    # Condense the lists into one
    included_edges.extend(sorted_open_edges)
    sorted_edges = included_edges
    del open_edges, sorted_open_edges, included_edges

    # Multigraphs need to handle edge keys in addition to edge data.
    if G.is_multigraph():
        for wt, u, v, k, d in sorted_edges:
            if subtrees[u] != subtrees[v]:
                if keys:
                    if data:
                        yield u, v, k, d
                    else:
                        yield u, v, k
                else:
                    if data:
                        yield u, v, d
                    else:
                        yield u, v
                subtrees.union(u, v)
    else:
        for wt, u, v, d in sorted_edges:
            if subtrees[u] != subtrees[v]:
                if data:
                    yield u, v, d
                else:
                    yield u, v
                subtrees.union(u, v)


def prim_mst_edges(G, minimum, weight="weight", keys=True, data=True, ignore_nan=False):
    """Iterate over edges of Prim's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        If `G` is a multigraph, `keys` controls whether edge keys ar yielded.
        Otherwise `keys` is ignored.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    is_multigraph = G.is_multigraph()
    push = heappush
    pop = heappop

    nodes = set(G)
    c = count()

    sign = 1 if minimum else -1

    while nodes:
        u = nodes.pop()
        frontier = []
        visited = {u}
        if is_multigraph:
            for v, keydict in G.adj[u].items():
                for k, d in keydict.items():
                    wt = d.get(weight, 1) * sign
                    if isnan(wt):
                        if ignore_nan:
                            continue
                        msg = f"NaN found as an edge weight. Edge {(u, v, k, d)}"
                        raise ValueError(msg)
                    push(frontier, (wt, next(c), u, v, k, d))
        else:
            for v, d in G.adj[u].items():
                wt = d.get(weight, 1) * sign
                if isnan(wt):
                    if ignore_nan:
                        continue
                    msg = f"NaN found as an edge weight. Edge {(u, v, d)}"
                    raise ValueError(msg)
                push(frontier, (wt, next(c), u, v, d))
        while nodes and frontier:
            if is_multigraph:
                W, _, u, v, k, d = pop(frontier)
            else:
                W, _, u, v, d = pop(frontier)
            if v in visited or v not in nodes:
                continue
            # Multigraphs need to handle edge keys in addition to edge data.
            if is_multigraph and keys:
                if data:
                    yield u, v, k, d
                else:
                    yield u, v, k
            else:
                if data:
                    yield u, v, d
                else:
                    yield u, v
            # update frontier
            visited.add(v)
            nodes.discard(v)
            if is_multigraph:
                for w, keydict in G.adj[v].items():
                    if w in visited:
                        continue
                    for k2, d2 in keydict.items():
                        new_weight = d2.get(weight, 1) * sign
                        push(frontier, (new_weight, next(c), v, w, k2, d2))
            else:
                for w, d2 in G.adj[v].items():
                    if w in visited:
                        continue
                    new_weight = d2.get(weight, 1) * sign
                    push(frontier, (new_weight, next(c), v, w, d2))


ALGORITHMS = {
    "boruvka": boruvka_mst_edges,
    "borůvka": boruvka_mst_edges,
    "kruskal": kruskal_mst_edges,
    "prim": prim_mst_edges,
}


@not_implemented_for("directed")
def minimum_spanning_edges(
    G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False
):
    """Generate edges in a minimum spanning forest of an undirected
    weighted graph.

    A minimum spanning tree is a subgraph of the graph (a tree)
    with the minimum sum of edge weights.  A spanning forest is a
    union of the spanning trees for each connected component of the graph.

    Parameters
    ----------
    G : undirected Graph
       An undirected graph. If `G` is connected, then the algorithm finds a
       spanning tree. Otherwise, a spanning forest is found.

    algorithm : string
       The algorithm to use when finding a minimum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is 'kruskal'.

    weight : string
       Edge data key to use for weight (default 'weight').

    keys : bool
       Whether to yield edge key in multigraphs in addition to the edge.
       If `G` is not a multigraph, this is ignored.

    data : bool, optional
       If True yield the edge data along with the edge.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    Returns
    -------
    edges : iterator
       An iterator over edges in a maximum spanning tree of `G`.
       Edges connecting nodes `u` and `v` are represented as tuples:
       `(u, v, k, d)` or `(u, v, k)` or `(u, v, d)` or `(u, v)`

       If `G` is a multigraph, `keys` indicates whether the edge key `k` will
       be reported in the third position in the edge tuple. `data` indicates
       whether the edge datadict `d` will appear at the end of the edge tuple.

       If `G` is not a multigraph, the tuples are `(u, v, d)` if `data` is True
       or `(u, v)` if `data` is False.

    Examples
    --------
    >>> from networkx.algorithms import tree

    Find minimum spanning edges by Kruskal's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> mst = tree.minimum_spanning_edges(G, algorithm="kruskal", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [1, 2], [2, 3]]

    Find minimum spanning edges by Prim's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> mst = tree.minimum_spanning_edges(G, algorithm="prim", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [1, 2], [2, 3]]

    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    Modified code from David Eppstein, April 2006
    http://www.ics.uci.edu/~eppstein/PADS/

    """
    try:
        algo = ALGORITHMS[algorithm]
    except KeyError as err:
        msg = f"{algorithm} is not a valid choice for an algorithm."
        raise ValueError(msg) from err

    return algo(
        G, minimum=True, weight=weight, keys=keys, data=data, ignore_nan=ignore_nan
    )



@not_implemented_for("directed")
def maximum_spanning_edges(
    G, algorithm="kruskal", weight="weight", keys=True, data=True, ignore_nan=False
):
    """Generate edges in a maximum spanning forest of an undirected
    weighted graph.

    A maximum spanning tree is a subgraph of the graph (a tree)
    with the maximum possible sum of edge weights.  A spanning forest is a
    union of the spanning trees for each connected component of the graph.

    Parameters
    ----------
    G : undirected Graph
       An undirected graph. If `G` is connected, then the algorithm finds a
       spanning tree. Otherwise, a spanning forest is found.

    algorithm : string
       The algorithm to use when finding a maximum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is 'kruskal'.

    weight : string
       Edge data key to use for weight (default 'weight').

    keys : bool
       Whether to yield edge key in multigraphs in addition to the edge.
       If `G` is not a multigraph, this is ignored.

    data : bool, optional
       If True yield the edge data along with the edge.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    Returns
    -------
    edges : iterator
       An iterator over edges in a maximum spanning tree of `G`.
       Edges connecting nodes `u` and `v` are represented as tuples:
       `(u, v, k, d)` or `(u, v, k)` or `(u, v, d)` or `(u, v)`

       If `G` is a multigraph, `keys` indicates whether the edge key `k` will
       be reported in the third position in the edge tuple. `data` indicates
       whether the edge datadict `d` will appear at the end of the edge tuple.

       If `G` is not a multigraph, the tuples are `(u, v, d)` if `data` is True
       or `(u, v)` if `data` is False.

    Examples
    --------
    >>> from networkx.algorithms import tree

    Find maximum spanning edges by Kruskal's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> mst = tree.maximum_spanning_edges(G, algorithm="kruskal", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [0, 3], [1, 2]]

    Find maximum spanning edges by Prim's algorithm

    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)  # assign weight 2 to edge 0-3
    >>> mst = tree.maximum_spanning_edges(G, algorithm="prim", data=False)
    >>> edgelist = list(mst)
    >>> sorted(sorted(e) for e in edgelist)
    [[0, 1], [0, 3], [2, 3]]

    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    Modified code from David Eppstein, April 2006
    http://www.ics.uci.edu/~eppstein/PADS/
    """
    try:
        algo = ALGORITHMS[algorithm]
    except KeyError as err:
        msg = f"{algorithm} is not a valid choice for an algorithm."
        raise ValueError(msg) from err

    return algo(
        G, minimum=False, weight=weight, keys=keys, data=data, ignore_nan=ignore_nan
    )



def minimum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    """Returns a minimum spanning tree or forest on an undirected graph `G`.

    Parameters
    ----------
    G : undirected graph
        An undirected graph. If `G` is connected, then the algorithm finds a
        spanning tree. Otherwise, a spanning forest is found.

    weight : str
       Data key to use for edge weights.

    algorithm : string
       The algorithm to use when finding a minimum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is
       'kruskal'.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    Returns
    -------
    G : NetworkX Graph
       A minimum spanning tree or forest.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> T = nx.minimum_spanning_tree(G)
    >>> sorted(T.edges(data=True))
    [(0, 1, {}), (1, 2, {}), (2, 3, {})]


    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    There may be more than one tree with the same minimum or maximum weight.
    See :mod:`networkx.tree.recognition` for more detailed definitions.

    Isolated nodes with self-loops are in the tree as edgeless isolated nodes.

    """
    edges = minimum_spanning_edges(
        G, algorithm, weight, keys=True, data=True, ignore_nan=ignore_nan
    )
    T = G.__class__()  # Same graph class as G
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T



def partition_spanning_tree(
    G, minimum=True, weight="weight", partition="partition", ignore_nan=False
):
    """
    Find a spanning tree while respecting a partition of edges.

    Edges can be flagged as either `INLCUDED` which are required to be in the
    returned tree, `EXCLUDED`, which cannot be in the returned tree and `OPEN`.

    This is used in the SpanningTreeIterator to create new partitions following
    the algorithm of Sörensen and Janssens [1]_.

    Parameters
    ----------
    G : undirected graph
        An undirected graph.

    minimum : bool (default: True)
        Determines whether the returned tree is the minimum spanning tree of
        the partition of the maximum one.

    weight : str
        Data key to use for edge weights.

    partition : str
        The key for the edge attribute containing the partition
        data on the graph. Edges can be included, excluded or open using the
        `EdgePartition` enum.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.


    Returns
    -------
    G : NetworkX Graph
        A minimum spanning tree using all of the included edges in the graph and
        none of the excluded edges.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """
    edges = kruskal_mst_edges(
        G,
        minimum,
        weight,
        keys=True,
        data=True,
        ignore_nan=ignore_nan,
        partition=partition,
    )
    T = G.__class__()  # Same graph class as G
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T


def maximum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    """Returns a maximum spanning tree or forest on an undirected graph `G`.

    Parameters
    ----------
    G : undirected graph
        An undirected graph. If `G` is connected, then the algorithm finds a
        spanning tree. Otherwise, a spanning forest is found.

    weight : str
       Data key to use for edge weights.

    algorithm : string
       The algorithm to use when finding a maximum spanning tree. Valid
       choices are 'kruskal', 'prim', or 'boruvka'. The default is
       'kruskal'.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.


    Returns
    -------
    G : NetworkX Graph
       A maximum spanning tree or forest.


    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> G.add_edge(0, 3, weight=2)
    >>> T = nx.maximum_spanning_tree(G)
    >>> sorted(T.edges(data=True))
    [(0, 1, {}), (0, 3, {'weight': 2}), (1, 2, {})]


    Notes
    -----
    For Borůvka's algorithm, each edge must have a weight attribute, and
    each edge weight must be distinct.

    For the other algorithms, if the graph edges do not have a weight
    attribute a default weight of 1 will be used.

    There may be more than one tree with the same minimum or maximum weight.
    See :mod:`networkx.tree.recognition` for more detailed definitions.

    Isolated nodes with self-loops are in the tree as edgeless isolated nodes.

    """
    edges = maximum_spanning_edges(
        G, algorithm, weight, keys=True, data=True, ignore_nan=ignore_nan
    )
    edges = list(edges)
    T = G.__class__()  # Same graph class as G
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T



@py_random_state(3)
def random_spanning_tree(G, weight=None, *, multiplicative=True, seed=None):
    """
    Sample a random spanning tree using the edges weights of `G`.

    This function supports two different methods for determining the
    probability of the graph. If ``multiplicative=True``, the probability
    is based on the product of edge weights, and if ``multiplicative=False``
    it is based on the sum of the edge weight. However, since it is
    easier to determine the total weight of all spanning trees for the
    multiplicative verison, that is significantly faster and should be used if
    possible. Additionally, setting `weight` to `None` will cause a spanning tree
    to be selected with uniform probability.

    The function uses algorithm A8 in [1]_ .

    Parameters
    ----------
    G : nx.Graph
        An undirected version of the original graph.

    weight : string
        The edge key for the edge attribute holding edge weight.

    multiplicative : bool, default=True
        If `True`, the probability of each tree is the product of its edge weight
        over the sum of the product of all the spanning trees in the graph. If
        `False`, the probability is the sum of its edge weight over the sum of
        the sum of weights for all spanning trees in the graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    nx.Graph
        A spanning tree using the distribution defined by the weight of the tree.

    References
    ----------
    .. [1] V. Kulkarni, Generating random combinatorial objects, Journal of
       Algorithms, 11 (1990), pp. 185–207
    """

    def find_node(merged_nodes, node):
        """
        We can think of clusters of contracted nodes as having one
        representative in the graph. Each node which is not in merged_nodes
        is still its own representative. Since a representative can be later
        contracted, we need to recursively search though the dict to find
        the final representative, but once we know it we can use path
        compression to speed up the access of the representative for next time.

        This cannot be replaced by the standard NetworkX union_find since that
        data structure will merge nodes with less representing nodes into the
        one with more representing nodes but this function requires we merge
        them using the order that contract_edges contracts using.

        Parameters
        ----------
        merged_nodes : dict
            The dict storing the mapping from node to representative
        node
            The node whose representative we seek

        Returns
        -------
        The representative of the `node`
        """
        if node not in merged_nodes:
            return node
        else:
            rep = find_node(merged_nodes, merged_nodes[node])
            merged_nodes[node] = rep
            return rep

    def prepare_graph():
        """
        For the graph `G`, remove all edges not in the set `V` and then
        contract all edges in the set `U`.

        Returns
        -------
        A copy of `G` which has had all edges not in `V` removed and all edges
        in `U` contracted.
        """

        # The result is a MultiGraph version of G so that parallel edges are
        # allowed during edge contraction
        result = nx.MultiGraph(incoming_graph_data=G)

        # Remove all edges not in V
        edges_to_remove = set(result.edges()).difference(V)
        result.remove_edges_from(edges_to_remove)

        # Contract all edges in U
        #
        # Imagine that you have two edges to contract and they share an
        # endpoint like this:
        #                        [0] ----- [1] ----- [2]
        # If we contract (0, 1) first, the contraction function will always
        # delete the second node it is passed so the resulting graph would be
        #                             [0] ----- [2]
        # and edge (1, 2) no longer exists but (0, 2) would need to be contracted
        # in its place now. That is why I use the below dict as a merge-find
        # data structure with path compression to track how the nodes are merged.
        merged_nodes = {}

        for u, v in U:
            u_rep = find_node(merged_nodes, u)
            v_rep = find_node(merged_nodes, v)
            # We cannot contract a node with itself
            if u_rep == v_rep:
                continue
            nx.contracted_nodes(result, u_rep, v_rep, self_loops=False, copy=False)
            merged_nodes[v_rep] = u_rep

        return merged_nodes, result

    def spanning_tree_total_weight(G, weight):
        """
        Find the sum of weights of the spanning trees of `G` using the
        approioate `method`.

        This is easy if the choosen method is 'multiplicative', since we can
        use Kirchhoff's Tree Matrix Theorem directly. However, with the
        'additive' method, this process is slightly more complex and less
        computatiionally efficent as we have to find the number of spanning
        trees which contain each possible edge in the graph.

        Parameters
        ----------
        G : NetworkX Graph
            The graph to find the total weight of all spanning trees on.

        weight : string
            The key for the weight edge attribute of the graph.

        Returns
        -------
        float
            The sum of either the multiplicative or additive weight for all
            spanning trees in the graph.
        """
        if multiplicative:
            return nx.total_spanning_tree_weight(G, weight)
        else:
            # There are two cases for the total spanning tree additive weight.
            # 1. There is one edge in the graph. Then the only spanning tree is
            #    that edge itself, which will have a total weight of that edge
            #    itself.
            if G.number_of_edges() == 1:
                return G.edges(data=weight).__iter__().__next__()[2]
            # 2. There are more than two edges in the graph. Then, we can find the
            #    total weight of the spanning trees using the formula in the
            #    reference paper: take the weight of that edge and multiple it by
            #    the number of spanning trees which have to include that edge. This
            #    can be accomplished by contracting the edge and finding the
            #    multiplicative total spanning tree weight if the weight of each edge
            #    is assumed to be 1, which is conviently built into networkx already,
            #    by calling total_spanning_tree_weight with weight=None
            else:
                total = 0
                for u, v, w in G.edges(data=weight):
                    total += w * nx.total_spanning_tree_weight(
                        nx.contracted_edge(G, edge=(u, v), self_loops=False), None
                    )
                return total

    U = set()
    st_cached_value = 0
    V = set(G.edges())
    shuffled_edges = list(G.edges())
    seed.shuffle(shuffled_edges)

    for u, v in shuffled_edges:
        e_weight = G[u][v][weight] if weight is not None else 1
        node_map, prepared_G = prepare_graph()
        G_total_tree_weight = spanning_tree_total_weight(prepared_G, weight)
        # Add the edge to U so that we can compute the total tree weight
        # assuming we include that edge
        # Now, if (u, v) cannot exist in G because it is fully contracted out
        # of existence, then it by definition cannot influence G_e's Kirchhoff
        # value. But, we also cannot pick it.
        rep_edge = (find_node(node_map, u), find_node(node_map, v))
        # Check to see if the 'representative edge' for the current edge is
        # in prepared_G. If so, then we can pick it.
        if rep_edge in prepared_G.edges:
            prepared_G_e = nx.contracted_edge(
                prepared_G, edge=rep_edge, self_loops=False
            )
            G_e_total_tree_weight = spanning_tree_total_weight(prepared_G_e, weight)
            if multiplicative:
                threshold = e_weight * G_e_total_tree_weight / G_total_tree_weight
            else:
                numerator = (
                    st_cached_value + e_weight
                ) * nx.total_spanning_tree_weight(prepared_G_e) + G_e_total_tree_weight
                denominator = (
                    st_cached_value * nx.total_spanning_tree_weight(prepared_G)
                    + G_total_tree_weight
                )
                threshold = numerator / denominator
        else:
            threshold = 0.0
        z = seed.uniform(0.0, 1.0)
        if z > threshold:
            # Remove the edge from V since we did not pick it.
            V.remove((u, v))
        else:
            # Add the edge to U since we picked it.
            st_cached_value += e_weight
            U.add((u, v))
        # If we decide to keep an edge, it may complete the spanning tree.
        if len(U) == G.number_of_nodes() - 1:
            spanning_tree = nx.Graph()
            spanning_tree.add_edges_from(U)
            return spanning_tree
    raise Exception(f"Something went wrong! Only {len(U)} edges in the spanning tree!")



class SpanningTreeIterator:
    """
    Iterate over all spanning trees of a graph in either increasing or
    decreasing cost.

    Notes
    -----
    This iterator uses the partition scheme from [1]_ (included edges,
    excluded edges and open edges) as well as a modified Kruskal's Algorithm
    to generate minimum spanning trees which respect the partition of edges.
    For spanning trees with the same weight, ties are broken arbitrarily.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """

    @dataclass(order=True)
    class Partition:
        """
        This dataclass represents a partition and stores a dict with the edge
        data and the weight of the minimum spanning tree of the partition dict.
        """

        mst_weight: float
        partition_dict: dict = field(compare=False)

        def __copy__(self):
            return SpanningTreeIterator.Partition(
                self.mst_weight, self.partition_dict.copy()
            )

    def __init__(self, G, weight="weight", minimum=True, ignore_nan=False):
        """
        Initialize the iterator

        Parameters
        ----------
        G : nx.Graph
            The directed graph which we need to iterate trees over

        weight : String, default = "weight"
            The edge attribute used to store the weight of the edge

        minimum : bool, default = True
            Return the trees in increasing order while true and decreasing order
            while false.

        ignore_nan : bool, default = False
            If a NaN is found as an edge weight normally an exception is raised.
            If `ignore_nan is True` then that edge is ignored instead.
        """
        self.G = G.copy()
        self.weight = weight
        self.minimum = minimum
        self.ignore_nan = ignore_nan
        # Randomly create a key for an edge attribute to hold the partition data
        self.partition_key = (
            "SpanningTreeIterators super secret partition attribute name"
        )


    def __iter__(self):
        """
        Returns
        -------
        SpanningTreeIterator
            The iterator object for this graph
        """
        self.partition_queue = PriorityQueue()
        self._clear_partition(self.G)
        mst_weight = partition_spanning_tree(
            self.G, self.minimum, self.weight, self.partition_key, self.ignore_nan
        ).size(weight=self.weight)

        self.partition_queue.put(
            self.Partition(mst_weight if self.minimum else -mst_weight, dict())
        )

        return self

    def __next__(self):
        """
        Returns
        -------
        (multi)Graph
            The spanning tree of next greatest weight, which ties broken
            arbitrarily.
        """
        if self.partition_queue.empty():
            del self.G, self.partition_queue
            raise StopIteration

        partition = self.partition_queue.get()
        self._write_partition(partition)
        next_tree = partition_spanning_tree(
            self.G, self.minimum, self.weight, self.partition_key, self.ignore_nan
        )
        self._partition(partition, next_tree)

        self._clear_partition(next_tree)
        return next_tree

    def _partition(self, partition, partition_tree):
        """
        Create new partitions based of the minimum spanning tree of the
        current minimum partition.

        Parameters
        ----------
        partition : Partition
            The Partition instance used to generate the current minimum spanning
            tree.
        partition_tree : nx.Graph
            The minimum spanning tree of the input partition.
        """
        # create two new partitions with the data from the input partition dict
        p1 = self.Partition(0, partition.partition_dict.copy())
        p2 = self.Partition(0, partition.partition_dict.copy())
        for e in partition_tree.edges:
            # determine if the edge was open or included
            if e not in partition.partition_dict:
                # This is an open edge
                p1.partition_dict[e] = EdgePartition.EXCLUDED
                p2.partition_dict[e] = EdgePartition.INCLUDED

                self._write_partition(p1)
                p1_mst = partition_spanning_tree(
                    self.G,
                    self.minimum,
                    self.weight,
                    self.partition_key,
                    self.ignore_nan,
                )
                p1_mst_weight = p1_mst.size(weight=self.weight)
                if nx.is_connected(p1_mst):
                    p1.mst_weight = p1_mst_weight if self.minimum else -p1_mst_weight
                    self.partition_queue.put(p1.__copy__())
                p1.partition_dict = p2.partition_dict.copy()

    def _write_partition(self, partition):
        """
        Writes the desired partition into the graph to calculate the minimum
        spanning tree.

        Parameters
        ----------
        partition : Partition
            A Partition dataclass describing a partition on the edges of the
            graph.
        """
        for u, v, d in self.G.edges(data=True):
            if (u, v) in partition.partition_dict:
                d[self.partition_key] = partition.partition_dict[(u, v)]
            else:
                d[self.partition_key] = EdgePartition.OPEN

    def _clear_partition(self, G):
        """
        Removes partition data from the graph
        """
        for u, v, d in G.edges(data=True):
            if self.partition_key in d:
                del d[self.partition_key]

def func_build_spanning_trees(graph):
    """
    функция для построения всех остовных деревьев неориентированного графа
    :param graph: undirected graph
    :return:
    ВЕЗДЕ ГДЕ НАПИСАЛ -10 был 0
    """
    """
    functions
    """
    def step_A(k, root_T1, root_T2, nodes_in_T2):
        print("В блоке A")
        for node in graph.nodes:
            print(graph.nodes[node])
        # Изменение пометок предшествования
        print("В step A k = ", k)
        # step A:
        if nodes_in_T2 == 1:
            for edge, index in zip(list_edges[k], range(len(list_edges[k]))):
                if root_T2 == edge:
                    if index == 0:
                        graph.nodes[edge]['parent'] = list_edges[k][index + 1]
                        graph.nodes[edge]['root'] = graph.nodes[list_edges[k][index + 1]]['root']
                    else:
                        graph.nodes[edge]['parent'] = list_edges[k][index - 1]
                        graph.nodes[edge]['root'] = graph.nodes[list_edges[k][index - 1]]['root']
            # Изменение корневых пометок
            for node in graph.nodes:
                if graph.nodes[node]['root'] == root_T2:  # Я думал, что не root_T2, а root_T1
                    graph.nodes[node]['root'] = list_edges[k][1]
        elif nodes_in_T2 > 1 and nodes_in_T2 < len(graph.nodes):
            # 1):
            x_j = list_edges[k][1]
            z = graph.nodes[x_j]['parent']
            flag_A = 1
            while flag_A:
                # 2):
                x_i = z
                z = graph.nodes[x_i]['parent']
                # 3):
                graph.nodes[x_i]['parent'] = x_j
                # 4):
                if z != -10:
                    x_j = x_i
                elif z == -10:
                    # 5):
                    flag_A = 0
                    # Изменение корневых пометок
                    graph.nodes[x_i]['root'] = root_T1
        else:
            print("В блоке step_A ошибка: количество корней в дереве T2!")
            print("nodes_in_T2 = ", nodes_in_T2)
    def step_b(k):
        # step Б:
        print("В блоке Б")
        for node in graph.nodes:
            print(graph.nodes[node])
        # определяем корень дерева T1
        root_T1 = graph.nodes[temp_spanning_tree[0][0]]['root']
        counter_true = 0
        for node in list_edges[k]:
            if graph.nodes[node]['parent'] == -10:
                counter_true += 1
        if counter_true == 2:
            root_T1 = graph.nodes[list_edges[k][0]]['root']
        root_T2 = None
        nodes_in_T2 = 1
        marker_b = 1
        if root_T1 == graph.nodes[list_edges[k][0]]['root']:
            root_T2 = graph.nodes[list_edges[k][1]]['root']
        elif root_T1 != graph.nodes[list_edges[k][0]]['root']:
            root_T2 = graph.nodes[list_edges[k][0]]['root']
            marker_b = 0
        elif root_T2 == None:
            print("ОШИБКА вблоке step_b: root_T2 == None")
        # i:
        for node in graph.nodes:
            if graph.nodes[node]['root'] == root_T2 and node != root_T2: # добавил второе условие (после этого всё пошло наперекосяк)
                graph.nodes[node]['root'] = root_T1
                nodes_in_T2 += 1
        # ii:
        step_A(k, root_T1, root_T2, nodes_in_T2)
        print("Перед ii в блоке step Б")
        for node in graph.nodes:
            print(graph.nodes[node])
        if marker_b:
            graph.nodes[list_edges[k][1]]['parent'] = list_edges[k][0]
        elif marker_b == 0:
            graph.nodes[list_edges[k][0]]['parent'] = list_edges[k][1]
        else:
            print("ОШИБКА в блоке step Б: marker_b")
        print("В конце блока step Б")
        for node in graph.nodes:
            print(graph.nodes[node])
    def step_B(removed_edge, temp_spanning_tree):
        temp_tree = copy.deepcopy(temp_spanning_tree)
        visited_nodes = set()
        visited_nodes.add(int(temp_tree[0][0]))
        visited_nodes.add(int(temp_tree[0][1]))
        temp_tree.pop(0)
        indexes_of_T2 = []
        for edge, index in zip(temp_tree, range(len(temp_tree))):
            if edge[0] in visited_nodes or edge[1] in visited_nodes:
                visited_nodes.add(int(edge[0]))
                visited_nodes.add(int(edge[1]))
            else:
                indexes_of_T2.append(index)
        if len(indexes_of_T2) > 0:
            input()
            tree = []
            for index in indexes_of_T2:
                tree.append(temp_tree[index])
            final_node = None
            for node in removed_edge:
                for edge in tree:
                    if int(node) == int(edge[0]) or int(node) == int(edge[1]):
                        final_node = int(node)
                        break
            if final_node == None:
                print("Ошибка в блоке step B: start_node == None!")
            root_tree = int(min(min(tree)))
            print("В блоке step B корень второго дерева root_tree = ", root_tree)
            for edge in tree:
                graph.nodes[edge[0]]['root'] = root_tree
                graph.nodes[edge[1]]['root'] = root_tree
            x_j = root_tree
            z = graph.nodes[x_j]['parent']
            flag_B = 1
            while flag_B:
                x_i = z
                z = graph.nodes[x_i]['parent']
                if x_i == final_node:
                    graph.nodes[x_i]['parent'] = x_j
                    graph.nodes[root_tree]['parent'] = -10
                    flag_B = 0
                else:
                    graph.nodes[x_i]['parent'] = x_j
                    x_j = x_i
        elif len(indexes_of_T2) == 0:
            # step B:
            if graph.nodes[removed_edge[1]]['parent'] == removed_edge[0]:
                # 1):
                graph.nodes[removed_edge[1]]['root'] = removed_edge[1]
                graph.nodes[removed_edge[1]]['parent'] = -10
                # 2):
                for node in graph.nodes:
                    if graph.nodes[node]['parent'] == removed_edge[1]:
                        graph.nodes[node]['root'] = removed_edge[1]
            elif graph.nodes[removed_edge[0]]['parent'] == removed_edge[1]:
                # 1):
                graph.nodes[removed_edge[0]]['root'] = removed_edge[0]
                graph.nodes[removed_edge[0]]['parent'] = -10
                # 2):
                for node in graph.nodes:
                    if graph.nodes[node]['parent'] == removed_edge[0]:
                        graph.nodes[node]['root'] = removed_edge[0]
        print("В конце блока step B:")
        for node in graph.nodes:
            print(graph.nodes[node])
    def step_2(k):
        print("В step 2 k = ", k)
        # step #2:
        while k <= (m - 1):
            # ii):
            if k == (m - 1) and len(temp_spanning_tree) < (n-2):
                step_5(k)
            if graph.nodes[list_edges[k][0]]['root'] != graph.nodes[list_edges[k][1]]['root']:
                temp_spanning_tree.append(list_edges[k])
                for node in list_edges[k]:
                    graph.nodes[node]['visited'] = True
                print("temp_spanning_tree", temp_spanning_tree)
                # step #3:
                step_3(k)
            # i):
            else:
                k += 1
        if k == m:
            step_5(k)
    def step_3(k):
        print("В step 3 k = ", k)
        step_b(k)
        step_4(k)
    def step_4(k):
        print("В step 4 k = ", k)
        if len(temp_spanning_tree) == (n-1):
            tree = copy.deepcopy(temp_spanning_tree)
            list_of_trees.append(tree)
            step_5(k)
        else:
            k += 1
            step_2(k)
    def step_5(k):
        print("В step 5 k = ", k)
        print("Количество найденных остовных деревьев: ", len(list_of_trees))
        for tree, index in zip(list_of_trees, range(len(list_of_trees))):
            print("Остовное дерево №", index + 1, " = ", tree)
        removed_edge = temp_spanning_tree.pop()
        index = 0
        l = 0
        for edge in graph.edges:
            if removed_edge == edge:
                l = index
            index += 1
        if l == 0:
            graph.nodes[0]['visited'] = True
            for node in graph.nodes:
                graph.nodes[node]['root'] = int(node)
                graph.nodes[node]['parent'] = -10
                graph.nodes[node]['visited'] = False
        for node in removed_edge:
            for edge in temp_spanning_tree:
                if node in edge:
                    graph.nodes[node]['visited'] = True
                    break
                graph.nodes[node]['visited'] = False
        print("Вершины графа после удаления ребра: ", removed_edge)
        for node in graph.nodes:
            print(graph.nodes[node])
        step_B(removed_edge, temp_spanning_tree)
        k = l + 1
        step_2(k)

    """
    functions
    """
    # preparing
    list_of_trees = []
    temp_spanning_tree = []
    list_edges = []
    for start, stop in graph.edges:
        list_edges.append((start, stop))
    m = len(graph.edges)
    n = len(graph.nodes)
    d = graph.degree[0]
    # step #1:
    graph.nodes[0]['visited'] = True
    print(graph.nodes[1]['visited'])
    for node in graph.nodes:
        graph.nodes[node]['root'] = int(node)
        graph.nodes[node]['parent'] = -10
        print(graph.nodes[node])
    k = 1
    step_2(k)

def func_networkx_build_spanning_tree(graph):
    """

    :param graph:
    :return:
    """
    iterator = SpanningTreeIterator(graph)
    trees = []
    for i in iterator:
        trees.append(i)
        #print(i.edges())
    return trees

########################################################################################################################
########################################################################################################################
#########################################граф для построения всех остовных деревьев ####################################
########################################################################################################################
########################################################################################################################

exm_edge_0 = (0, 1, 70.1, 630, 0, 701, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
exm_edge_1 = (0, 2, 5.62, 220, 0, 56.2, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
exm_edge_2 = (1, 2, 2.55, 0, 0, 25.5, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
exm_edge_3 = (1, 3, 70, 0, 0, 700, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
exm_edge_4 = (1, 4, 85.89, 0, 0, 858.9, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
exm_edge_5 = (2, 3, 3.69, 0, 0, 36.9, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
exm_edge_6 = (3, 4, 2.33, 0, 0, 23.3, 35, 0, 'Al', 0, 0, 0.89, 0, 0, 0)
exm_edges = np.array([exm_edge_0,
                      exm_edge_1,
                      exm_edge_2,
                      exm_edge_3,
                      exm_edge_4,
                      exm_edge_5,
                      exm_edge_6])
exm_directed_adjacency_list = np.array([(1, 2),
                                        (2, 3, 4),
                                        (3),
                                        (4)])
