from typing import Sequence, Tuple

from gowpy.gow.builder import GoWBuilder
from gowpy.gow.typing import Tokenizer

from networkx.algorithms.core import core_number


class GoWKeywordExtractor(object):
    """Extract keywords from a text document based on a graph-of-words representation

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
        # TODO is_weighted
        self.builder = GoWBuilder(
                 directed=directed,
                 window_size=window_size,
                 tokenizer=tokenizer)

    def extract(self, document: str) -> Sequence[Tuple[str, float]]:
        gow = self.builder.compute_gow_from_document(document)
        graph = gow.to_graph()
        kcore = core_number(graph)

        keywords = []
        k_max = 0
        for v, k in kcore.items():
            if k > k_max:
                keywords.clear()
                k_max = k

            if k == k_max:
                token_code = graph.nodes[v]['label']
                token = self.builder.get_token_(token_code)
                keywords.append((token, k))

        return keywords
