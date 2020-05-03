from typing import Sequence, Dict, Optional, Union, List, Callable

from gowpy.gow.typing import Token, Tokenized_document, Tokenizer, \
    Edge, Edge_with_code, Edge_label, Edges, Nodes
import networkx as nx
from gowpy.utils.defaults import default_tokenizer


def mk_undirected_edge(node_start_code: int,
                       node_end_code: int,
                       code: Optional[int] = None) -> Union[Edge, Edge_with_code]:
    """Builds an unambiguous representation of an undirected edge"""
    if node_start_code < node_end_code:
        n1, n2, label = node_start_code, node_end_code, code
    else:
        n1, n2, label = node_end_code, node_start_code, code

    if code is None:
        return n1, n2
    else:
        return n1, n2, label


def mk_directed_edge(node_start_code: int,
                     node_end_code: int,
                     code: Optional[int] = None) -> Union[Edge, Edge_with_code]:
    """Builds an unambiguous representation of a directed edge"""
    if code is None:
        return node_start_code, node_end_code
    else:
        return node_start_code, node_end_code, code


class GraphOfWords(object):
    """
    Represents a graph-of-words

    .. seealso:: gowpy.gow.builder.GoWBuilder
    .. note:: this class should not be used directly, see GoWBuilder
    """
    def __init__(self,
                 nodes: Nodes,
                 edges: Edges,
                 get_token: Callable[[int], str],
                 get_label: Optional[Callable[[int], Edge_label]],
                 freq: int = 1,
                 directed: bool = False):
        self.get_token = get_token
        self.get_label = get_label

        self.nodes = nodes
        self.edges = edges
        self.directed = directed
        self.freq = freq

        self.graph_: Optional[nx.Graph] = None

    def is_edge_labeling(self):
        return self.get_label is not None

    def __str__(self):
        nodes = self.nodes_str()
        edges = self.edges_str()
        return """Graph-of-words\nNodes: {}\nEdges: {}\n""".format(nodes, edges)

    def __repr__(self):
        return self.__str__()

    def nodes_str(self) -> List[str]:
        return [self.get_token(node_code) for node_code in self.nodes]

    def __edges_to_str(self, edge: Edge) -> str:
        start_node, end_node = edge
        return f'{self.get_token(start_node)}__{self.get_token(end_node)}'

    def edges_str(self) -> List[str]:
        if self.is_edge_labeling():
            return [self.__edges_to_str(self.get_label(edge_label_code))
                    for _, _, edge_label_code in self.edges]
        else:
            return [self.__edges_to_str(mk_directed_edge(node_start, node_end) if self.directed else mk_undirected_edge(node_start, node_end))
                    for node_start, node_end in self.edges]

    def to_graph(self) -> nx.Graph:
        """Computes and memoize a NetworkX representation

        This representation is suited for algorithms rather than visualisation.
        """
        if self.graph_ is None:
            g = nx.Graph() if not self.directed else nx.DiGraph()

            [g.add_node(node, label=node) for node in self.nodes]

            if self.is_edge_labeling():
                [g.add_edge(node_start_code, node_end_code, label=edge_code)
                 for node_start_code, node_end_code, edge_code in self.edges]
            else:
                g.add_edges_from(self.edges)

            self.graph_ = g

        return self.graph_

    def to_labeled_graph(self) -> nx.Graph:
        """Computes a NetworkX representation suited for drawing"""
        g = nx.Graph() if not self.directed else nx.DiGraph()

        [g.add_node(self.get_token(node)) for node in self.nodes]

        if self.is_edge_labeling():
            [g.add_edge(self.get_token(node_start_code), self.get_token(node_end_code))
             for node_start_code, node_end_code, _ in self.edges]
        else:
            g.add_edges_from([(self.get_token(node_start_code), self.get_token(node_end_code))
                              for node_start_code, node_end_code in self.edges])

        return g


class GoWBuilder(object):
    """Builder to construct graph-of-words from a single document or a corpus of documents

    Parameters
    ----------
    directed : boolean, False by default
        If True, the graph-of-words is directed, else undirected
    window_size : int, default=4
        Size of the window (in token) to build the graph-of-words.
    edge_labeling : boolean, False by default
        If True, edges are labeled with a unique code, else edges are not labeled.
    tokenizer : callable or None (default)
        Override the string tokenization step.
    """

    def __init__(self,
                 directed: bool = False,
                 window_size: int = 4,
                 tokenizer: Tokenizer = None,
                 edge_labeling: bool = False):
        # Graph parameters
        self.directed: bool = directed
        self.window_size: int = window_size

        self.corpus_size: Optional[int] = None

        self.tokenizer: Tokenizer = tokenizer if tokenizer is not None else default_tokenizer

        self.TOKEN_TO_INT_: Dict[Token, int] = {}
        self.INT_TO_TOKEN_: Dict[int, Token] = {}

        self.edge_labeling = edge_labeling
        if self.edge_labeling:
            self.LABEL_TO_INT_: Dict[Edge_label, int] = {}
            self.INT_TO_LABEL_: Dict[int, Edge_label] = {}

    # TODO generate a real formal python representation
    def __repr__(self):
        return f'''Graph-of-word builder:
        - is_directed: {self.directed}
        - window_size: {self.window_size}
        - edge_labeling: {self.edge_labeling}

        - Number of tokens: {len(self.TOKEN_TO_INT_)}
        - Number of links between tokens: {len(self.LABEL_TO_INT_)}
        '''.lstrip()

    def __str__(self):
        return self.__repr__()

    # Node
    def get_code_(self, token: Token) -> int:
        if token not in self.TOKEN_TO_INT_:
            last_token_id_ = len(self.TOKEN_TO_INT_)
            self.TOKEN_TO_INT_[token] = last_token_id_
            self.INT_TO_TOKEN_[last_token_id_] = token

        return self.TOKEN_TO_INT_[token]

    def get_token_(self, code: int) -> Token:
        return self.INT_TO_TOKEN_[code]

    # Edge
    def get_label_id_(self, label: Edge_label) -> int:
        if label not in self.LABEL_TO_INT_:
            last_label_id_ = len(self.LABEL_TO_INT_)
            self.LABEL_TO_INT_[label] = last_label_id_
            self.INT_TO_LABEL_[last_label_id_] = label

        return self.LABEL_TO_INT_[label]

    def get_label_(self, code: int) -> Edge_label:
        return self.INT_TO_LABEL_[code]

    def get_edge_code_(self, edge: Edge) -> int:
        node_start_code, node_end_code = edge
        # Computation of the edge label and edge label ID
        if self.directed:
            t1, t2 = (node_start_code, node_end_code)
        else:
            if node_start_code < node_end_code:
                t1, t2 = (node_start_code, node_end_code)
            else:
                t1, t2 = (node_end_code, node_start_code)

        edge_label = (t1, t2)

        edge_code = self.get_label_id_(edge_label)

        return edge_code

    def compute_gow_from_corpus(self, raw_documents: Sequence[str]) -> Sequence[GraphOfWords]:
        """Computes a graph-of-words representation for each given documents"""
        result_graph_of_words = []

        for raw_document in raw_documents:
            gow = self.compute_gow_from_document(raw_document)
            result_graph_of_words.append(gow)

        self.corpus_size = len(result_graph_of_words)

        return result_graph_of_words

    def compute_gow_from_tokenized_document(self, tokens: Tokenized_document) -> GraphOfWords:
        nodes = set()
        token_ids = []
        for token in tokens:
            token_id = self.get_code_(token)
            token_ids.append(token_id)
            nodes.add(token_id)

        N = len(tokens)

        edges = set()
        if self.edge_labeling:
            for j in range(N):
                for i in range(max(j - self.window_size + 1, 0), j):
                    # Only keep edges between two *different* tokens
                    if token_ids[i] != token_ids[j]:
                        edge = (token_ids[i], token_ids[j])
                        edge_code = self.get_edge_code_(edge)
                        if self.directed:
                            edges.add(mk_directed_edge(token_ids[i], token_ids[j], edge_code))
                        else:
                            edges.add(mk_undirected_edge(token_ids[i], token_ids[j], edge_code))
        else:
            for j in range(N):
                for i in range(max(j - self.window_size + 1, 0), j):
                    # Only keep edges between two *different* tokens
                    if token_ids[i] != token_ids[j]:
                        if self.directed:
                            edges.add(mk_directed_edge(token_ids[i], token_ids[j]))
                        else:
                            edges.add(mk_undirected_edge(token_ids[i], token_ids[j]))

        return GraphOfWords(nodes=nodes,
                            edges=edges,
                            get_label=self.get_label_ if self.edge_labeling else None,
                            get_token=self.get_token_,
                            directed=self.directed)

    def compute_gow_from_document(self, raw_document: str) -> GraphOfWords:
        """Computes a graph-of-words representation from a document"""
        tokens = self.tokenizer(raw_document)
        return self.compute_gow_from_tokenized_document(tokens)
