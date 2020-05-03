import numpy as np
from typing import Sequence, Optional

import gowpy.gow.io
from gowpy.gow.builder import GraphOfWords
from gowpy.gow.typing import Tokenizer
from gowpy.gow.builder import GoWBuilder


class GoWMiner(GoWBuilder):
    """A miner of frequent subgraphs for a collection of graph-of-words

    Currently, the mining operation is delegated to a C++ program. This class makes it possible to load the
    mined sub-graphs-of-words.

    Parameters
    ----------
    directed : boolean, False by default
        If True, the graph-of-words is directed, else undirected
    window_size : int, default=4
        Size of the window (in token) to build the graph-of-words.
    tokenizer : callable or None (default)
        Override the string tokenization step.
    """

    def __init__(self,
                 directed: bool = False,
                 window_size: int = 4,
                 tokenizer: Tokenizer = None):
        # /!\ Edge labeling is important for IO
        super().__init__(directed, window_size, tokenizer, edge_labeling=True)
        self.frequent_subgraphs: Optional[Sequence[GraphOfWords]] = None

    # TODO generate a real formal python representation
    def __repr__(self):
        if self.frequent_subgraphs is None:
            len_frequent_subgraphs = "not loaded yet"
        else:
            len_frequent_subgraphs = len(self.frequent_subgraphs)
        return f'''Graph-of-word miner:
        - is_directed: {self.directed}
        - window_size: {self.window_size}
        - edge_labeling: {self.edge_labeling}

        - Number of tokens: {len(self.TOKEN_TO_INT_)}
        - Number of links between tokens: {len(self.LABEL_TO_INT_)}

        - Number of loaded subgraph: {len_frequent_subgraphs}
        '''.lstrip()

    def load_graphs(self,
                    input_file_subgraph: str,
                    input_file_frequent_nodes: str) -> None:
        self.frequent_subgraphs = gowpy.gow.io.load_graphs(input_file_subgraph, input_file_frequent_nodes,
                                                           self.get_token_, self.get_label_,
                                                           self.directed)

    def stat_freq_per_pattern(self) -> np.array:
        """Computes the subgraph frequency series"""
        return np.array([pattern.freq for pattern in self.frequent_subgraphs])

    def stat_relative_freq_per_pattern(self) -> np.array:
        """Computes the subgraph normalised frequency series"""
        return np.array([pattern.freq / float(self.corpus_size) for pattern in self.frequent_subgraphs])

    def stat_num_nodes_per_pattern(self) -> np.array:
        """Computes the number of nodes per subgraph series"""
        return np.array([len(pattern.nodes) for pattern in self.frequent_subgraphs])

    def stat_num_edges_per_pattern(self) -> np.array:
        """Computes the number of edges per subgraph series"""
        return np.array([len(pattern.edges) for pattern in self.frequent_subgraphs])
