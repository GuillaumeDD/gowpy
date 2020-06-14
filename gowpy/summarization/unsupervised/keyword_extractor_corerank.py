from typing import Sequence, Tuple, Optional, Union

import numbers
import warnings

from gowpy.gow.builder import GoWBuilder
from gowpy.gow.typing import Tokenizer

from networkx.algorithms.core import core_number as nx_core_number
from gowpy.summarization.unsupervised.algorithms import core_number_weighted

from gowpy.summarization.unsupervised.helpers import elbow


class CoreRankKeywordExtractor(object):
    """Extract keywords from a text document at the node-level of a graph-of-words representation

    Each node/token in the graph-of-words is associated a score, namely the sum of the core numbers
    of its neighbors. Then each node is ranked in decreasing order of score.

    This extractor allows both the selection of an automatically adaptive number of keywords and the
    selection of given number or proportion of keywords.

    Parameters
    ----------
    directed : boolean, False by default
        If True, the graph-of-words is directed, else undirected
    weighted : boolean, False by default
        If True, the edges of the graph-of-words are weighted by their frequency, else they are not weighted
    window_size : int, default=4
        Size of the window (in token) to build the graph-of-words.
    n : int or float in [0.0, 1.0] or None (default)
        If None the number of extracted keywords is automatically computed via the elbow method, else
        extraction the desired number of keywords (int) or extraction of the desired proportion of
        keywords (float)
    tokenizer : callable or None (default)
        Override the string tokenization step.
    """

    def __init__(self,
                 directed: bool = False,
                 weighted: bool = False,
                 window_size: int = 4,
                 n: Optional[Union[int, float]] = None,
                 tokenizer: Tokenizer = None):
        self.builder = GoWBuilder(
            directed=directed,
            weighted=weighted,
            window_size=window_size,
            tokenizer=tokenizer)

        self.n = n

    def extract(self, document: str,
                n: Optional[Union[int, float]] = None) -> Sequence[Tuple[str, float]]:
        """Extracts keywords from the given document

        Parameters
        ----------
        document : string
           The document in which to extract keywords
        n : int or float in [0.0, 1.0] or None (default)
            If None the number of extracted keywords is automatically computed via the elbow method, else
            extraction the desired number of keywords (int) or extraction of the desired proportion of
            keywords (float)
        """
        # Building the graph-of-words
        gow = self.builder.compute_gow_from_document(document)
        graph = gow.to_graph()

        # Computation of the k-cores
        if self.builder.weighted:
            kcore = core_number_weighted(graph)
        else:
            kcore = nx_core_number(graph)

        # Computation of the scores for each node
        v_with_score = []
        for i, v in enumerate(graph.nodes):
            score = 0
            for v_neighbor in graph.neighbors(v):
                score += kcore[v_neighbor]

            v_with_score.append((v, score))

        # Ranking of the nodes
        v_with_score.sort(key=lambda vs: vs[1],
                          reverse=True)

        # Computation of the number of keyword to extract
        if (self.n is None) and (n is None):
            n_stop = elbow([(i, score) for i, (v, score) in enumerate(v_with_score)])
            n_stop = n_stop + 1
        else:
            if (self.n is not None) and (n is not None) and (self.n != n):
                warnings.warn(f"Both instance member n={self.n} and function parameter n={n} are defined. "
                              f"Overriding: using function parameter value n={n}",
                              UserWarning)
            current_n = (n if n is not None
                         else self.n)

            num_requested_keywords = (abs(current_n)
                                      if isinstance(current_n, numbers.Integral)
                                      else round(abs(current_n) * len(v_with_score)))
            n_stop = min(num_requested_keywords,
                         len(v_with_score))

        # Retrieving the keywords
        keywords = []
        for v, score in v_with_score[:n_stop]:
            token_code = graph.nodes[v]['label']
            token = self.builder.get_token_(token_code)
            keywords.append((token, score))

        return keywords
