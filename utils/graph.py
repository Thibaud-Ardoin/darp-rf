from typing import TypeVar, Iterable, Callable, Set, Union, Tuple, List

N = TypeVar('N')
A = TypeVar('A')
G = TypeVar('G')


def decompose_paths_and_cycles(graph: G,
                               sinks: Set[N],
                               find_start: Callable[[G], Union[N, None]],
                               outgoing_arc: Callable[[G, N], Iterable[Tuple[A, float, N]]],
                               subtract_arc: Callable[[G, A, float], None]
                               ) -> Tuple[List[Tuple[List[A], float]], List[Tuple[List[A], float]]]:
    """
    Decompose a directed graph into paths and cycles.  The graph is assumed to have a set of source (no incoming edges)
    nodes and a set of sink (no outgoing edges) nodes.  The edges are assumed to have a nonnegative weight, which when
    zero, means the edge is not in the graph.  The edge weights are assumed to obey a conservation of flow at each
    non-source/sink node, that is, the sum of weights from incoming edges is equal to the some of weights of outgoing
    edges.  The initial edge weights are assumed to be strictly positive.  The graph will be mutated in-place.

    :param graph: The graph object (user supplied)
    :param sinks: The complete set of sink nodes.
    :param find_start: A function which given the graph, returns a connected source node, or if none exist, an
        arbitrary connected node, or if the graph is empty, returns `None`.
    :param outgoing_arc: A function which given the graph and a node with non-zero weight, finds an arc (edge) with
        nonzero weight and returns a tuple containing the arc, the arc weight and the arc's end node.
    :param subtract_arc: A function which given the graph, an arc and a number, subtracts the number from arc's edge
        weight and removes the arc from the graph if the resulting edge weight is (close to) 0. The function is
        expected to mutate the graph in-place.

    :return: A 2-tuple whose first element is a list of path-weight pairs (each path is a list of arcs), and whose
        second element is a list of cycle-weight pairs (each cycle is a list of arcs).
    """
    paths = []
    cycles = []

    while True:
        node = find_start(graph)
        if node is None:
            break

        visited_nodes = {node: 0}
        node_order = [node]
        edges = []
        while True:
            arc, val, next_node = outgoing_arc(graph, node)
            edges.append((arc, val))

            if next_node in visited_nodes:
                # cycle
                k = visited_nodes[next_node]
                cycle = edges[k:]
                cycle_val = min(v for _, v in cycle)
                cycle = tuple(e for e, _ in cycle)
                cycles.append((cycle, cycle_val))
                for arc in cycle:
                    subtract_arc(graph, arc, cycle_val)

                if k > 0:
                    # backtrack
                    edges = edges[:k]
                    for n in node_order[k + 1:]:
                        del visited_nodes[n]
                    node_order = node_order[:k + 1]
                    node = node_order[-1]
                    continue  # backtrack
                else:
                    break  # Start from scratch
            elif next_node in sinks:
                # path
                path = tuple(e for e, _ in edges)
                path_val = min(v for _, v in edges)
                paths.append((path, path_val,))
                for arc in path:
                    subtract_arc(graph, arc, path_val)
                break

            visited_nodes[next_node] = len(node_order)
            node_order.append(next_node)
            node = next_node

    return paths, cycles
