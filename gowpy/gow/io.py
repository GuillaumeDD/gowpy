import re

from typing import Tuple, List, Sequence, Callable

from gowpy.gow.builder import mk_undirected_edge, mk_directed_edge
from gowpy.gow.builder import GraphOfWords
from gowpy.gow.typing import Edge_label


def gow_to_data(gows: Sequence[GraphOfWords]) -> str:
    """
    Convert a sequence of graph-of-words into a text representation for interoperability with other programs

    Format:
    - "t # N" means the Nth graph,
    - "v M L" means that the Mth vertex in this graph has label L,
    - "e P Q L" means that there is an edge connecting the Pth vertex with the Qth vertex. The edge has label L.

    :param gows:
    :return:
    """
    result_data = []

    for i, gow in enumerate(gows):
        nodes = gow.nodes
        edges = gow.edges

        if len(nodes) > 0:
            result_data.append(u"t # {}\n".format(i))

            node_label_to_id = {}

            for node_label in nodes:
                if not (node_label in node_label_to_id):
                    new_id = len(node_label_to_id)
                    node_label_to_id[node_label] = new_id

                node_id = node_label_to_id[node_label]
                result_data.append(u"v {} {}\n".format(node_id, node_label))

            edge_tuples = []  # TODO implementation with a heap to be more efficient?
            for (node_start_label, node_end_label, edge_label_id) in edges:
                # Computation of the node IDs in this graph given their node labels
                node_start_id = node_label_to_id[node_start_label]
                node_end_id = node_label_to_id[node_end_label]

                edge_tuples.append((node_start_id, node_end_id, edge_label_id))
            edge_tuples.sort()

            for node_start_id, node_end_id, edge_label_id in edge_tuples:
                result_data.append(u"e {} {} {}\n".format(node_start_id,
                                                          node_end_id,
                                                          edge_label_id))

    result_data.append(u"t # {}".format(-1))
    return u"".join(result_data)


r_new_graph_ = re.compile(u't +# +(\d+) +\\* +(\d+)')
r_new_vertex_ = re.compile(u'v +(\d+) +(\d+)')
r_new_edge_ = re.compile(u'e +(\d+) +(\d+) +(\d+)')
r_new_parent_graphs_ = re.compile(u'x: +([\d ]+)')


def load_graphs(input_file_subgraph: str,
                input_file_frequent_nodes: str,
                get_token: Callable[[int], str],
                get_label: Callable[[int], Edge_label],
                is_directed: bool=False) -> Sequence[GraphOfWords]:
    #
    current_id = None
    current_freq = None
    current_vertices = None
    current_edges = None
    current_parent_graph_ids = None

    subgraphs = []

    with open(input_file_subgraph, 'r') as f_input_file:
        for line in f_input_file:
            m_new_graph = r_new_graph_.search(line)
            m_new_vertex = r_new_vertex_.search(line)
            m_new_edge = r_new_edge_.search(line)
            m_new_parent_graphs = r_new_parent_graphs_.search(line)

            if m_new_graph:
                # Saving
                if current_id is not None:
                    subgraphs.append(_to_gow(current_id,
                                             current_freq,
                                             (current_vertices, current_edges),
                                             current_parent_graph_ids,
                                             get_token, get_label,
                                             is_directed))

                # Initialisation of the new graph
                current_id = int(m_new_graph.group(1))
                current_freq = int(m_new_graph.group(2))
                current_vertices = []
                current_edges = []
                current_parent_graph_ids = None

            elif m_new_vertex:
                vertex_id = int(m_new_vertex.group(1))
                vertex_label = int(m_new_vertex.group(2))

                current_vertices.append((vertex_id, vertex_label))

            elif m_new_edge:
                node_start = int(m_new_edge.group(1))
                node_end = int(m_new_edge.group(2))
                edge_label = int(m_new_edge.group(3))

                current_edges.append((node_start, node_end, edge_label))

            elif m_new_parent_graphs:
                current_parent_graph_ids = [int(graph_id) for graph_id in
                                            m_new_parent_graphs.group(1).strip().split(' ')]
                # assert len(current_parent_graph_ids) == current_freq

            else:
                pass  # other lines (probably empty)

        # Last line
        if current_id and current_parent_graph_ids:
            subgraphs.append(
                _to_gow(current_id, current_freq, (current_vertices, current_edges), current_parent_graph_ids,
                        get_token, get_label,
                        is_directed))

    current_id = None
    PADDING_ID = len(subgraphs)
    current_freq = None
    current_vertices = None
    current_edges = None
    current_parent_graph_ids = None

    with open(input_file_frequent_nodes, 'r') as f_input_file:
        for line in f_input_file:
            m_new_graph = r_new_graph_.search(line)
            m_new_vertex = r_new_vertex_.search(line)
            m_new_parent_graphs = r_new_parent_graphs_.search(line)

            if m_new_graph:
                # Saving
                if current_id is not None:
                    subgraphs.append(_to_gow(current_id,
                                             current_freq,
                                             (current_vertices, current_edges),
                                             current_parent_graph_ids,
                                             get_token, get_label,
                                             is_directed))

                # Initialisation of the new graph
                current_id = int(m_new_graph.group(1)) + PADDING_ID
                current_freq = int(m_new_graph.group(2))
                current_vertices = []
                current_edges = []
                current_parent_graph_ids = None

            elif m_new_vertex:
                vertex_id = int(m_new_vertex.group(1))
                vertex_label = int(m_new_vertex.group(2))

                current_vertices.append((vertex_id, vertex_label))

            elif m_new_parent_graphs:
                current_parent_graph_ids = [int(graph_id) for graph_id in
                                            m_new_parent_graphs.group(1).strip().split(' ')]
                # assert len(current_parent_graph_ids) == current_freq

            else:
                pass  # other lines (probably empty)

        # Last line
        if current_id and current_parent_graph_ids:
            subgraphs.append(
                _to_gow(current_id, current_freq, (current_vertices, current_edges), current_parent_graph_ids,
                        get_token, get_label, is_directed))

    return subgraphs

IO_Nodes = List[Tuple[int, int]]  # (node_id, node_code)
IO_Edges = List[Tuple[int, int, int]]  # (node_start_id, node_end_id, edge_code)
IO_Subgraph = Tuple[IO_Nodes, IO_Edges]

def _to_gow(subg_id: int,
            subg_freq: int,
            subgraph: IO_Subgraph,
            subg_current_parent_graph_ids: Sequence[int],
            get_token: Callable[[int], str],
            get_label: Callable[[int], Edge_label],
            is_directed: bool) -> GraphOfWords:
    id_: int = subg_id
    freq: int = subg_freq

    subg_vertices, subg_edges = subgraph

    size = len(subg_vertices)
    parents = subg_current_parent_graph_ids

    # Recomputation of nodes
    # Dealing with nodes:
    # Node = (node id in *this* graph, node code)
    node_id_to_node_code = {}
    nodes = set()
    for node_id, node_code in subg_vertices:
        node_id_to_node_code[node_id] = node_code
        nodes.add(node_code)

    # Dealing with edges
    edges = set()
    for node_start_id, node_end_id, edge_label_code in subg_edges:
        node_start_code = node_id_to_node_code[node_start_id]
        node_end_code = node_id_to_node_code[node_end_id]
        if is_directed:
            edges.add(mk_directed_edge(node_start_code, node_end_code, edge_label_code))
        else:
            edges.add(mk_undirected_edge(node_start_code, node_end_code, edge_label_code))

    return GraphOfWords(nodes=nodes,
                        edges=edges,
                        get_token=get_token,
                        get_label=get_label,
                        freq=freq,
                        directed=is_directed)