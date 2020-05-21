from typing import Sequence, Tuple

from gowpy.gow.builder import GoWBuilder
from gowpy.gow.typing import Tokenizer

from networkx.algorithms.core import k_shell, k_core, core_number as nx_core_number
from networkx.classes.function import density

from gowpy.summarization.unsupervised.algorithms import core_number_weighted

import warnings

from gowpy.summarization.unsupervised.helpers import elbow

GOW_KEYWORD_EXTRACTION_KCORE = "maximum"
GOW_KEYWORD_EXTRACTION_DENSITY = "density"
GOW_KEYWORD_EXTRACTION_INFLEXION = "inflexion"


class KcoreKeywordExtractor(object):
    """Extract keywords from a text document based on the k-core of a graph-of-words representation

    k-core approaches allow the selection of cohesive keywords. Selected keywords correspond to a
    cohesive subgraph. In other words, the granularity of selection is at the level of cohesive
    subgraphs and nodes are selected by entire batch at a time.
    A key property is the selection of an automatically adaptive number of cohesive keywords.

    This extractor implements several ways of exploiting the k-core to extract cohesive keywords.
    The 'maximum' method simply selects the main core (the k-core with maximum k).
    It can be viewed as being too restrictive. Two other selection methods alleviate this limitation.
    On one hand, the 'density' method goes down the hierarchy of k-cores to select the one retaining
    the cohesiveness from the perspective of the density of the k-core. The most appropriate k-core is
    selected via the elbow method.
    On the other hand, the 'inflexion' method exploits the k-shell (which is the part of the k-core
    that does not survive in the (k+1)-core). It consists in going down the hierachy of k-cores as
    long as the shells increase in size, else stopping.

    Parameters
    ----------
    directed : boolean, False by default
        If True, the graph-of-words is directed, else undirected
    weighted : boolean, False by default
        If True, the edges of the graph-of-words are weighted by their frequency, else they are not weighted
    window_size : int, default=4
        Size of the window (in token) to build the graph-of-words.
    selection_method : string {'maximum',  'density', 'inflexion'}
        Method for selecting keywords from the k-core
        'maximum' (default) : keywords associated with the nodes from the main core
        'density' : keywords associated with the k-core selected by density via the elbow method
        'inflexion' : keywords associated with the k-core based on the variation of the size of k-shell
    tokenizer : callable or None (default)
        Override the string tokenization step.
    """
    def __init__(self,
                 directed: bool = False,
                 weighted: bool = False,
                 window_size: int = 4,
                 selection_method: str = GOW_KEYWORD_EXTRACTION_KCORE,
                 tokenizer: Tokenizer = None):
        self.builder = GoWBuilder(
            directed=directed,
            weighted=weighted,
            window_size=window_size,
            tokenizer=tokenizer)

        self.selection_method = selection_method

    def extract(self, document: str) -> Sequence[Tuple[str, float]]:
        """Extracts keywords from the given document"""
        if self.selection_method == GOW_KEYWORD_EXTRACTION_KCORE:
            return self.extract_k_core(document)
        elif self.selection_method == GOW_KEYWORD_EXTRACTION_DENSITY:
            return self.extract_with_density(document)
        elif self.selection_method == GOW_KEYWORD_EXTRACTION_INFLEXION:
            return self.extract_with_inflexion(document)
        else:
            all_values = [GOW_KEYWORD_EXTRACTION_KCORE, GOW_KEYWORD_EXTRACTION_DENSITY, GOW_KEYWORD_EXTRACTION_INFLEXION]
            warnings.warn(f"Unknown value: {self.selection_method}. Using default: {GOW_KEYWORD_EXTRACTION_KCORE}"
                          f"Possible values: {all_values}",
                          UserWarning)
            return self.extract_k_core(document)

    def extract_k_core(self, document: str) -> Sequence[Tuple[str, float]]:
        """Extraction of keywords corresponding to the nodes of the main core"""
        # Building the graph-of-words
        gow = self.builder.compute_gow_from_document(document)
        graph = gow.to_graph()

        # Computation of the k-cores
        if self.builder.weighted:
            kcore_number = core_number_weighted(graph)
        else:
            kcore_number = nx_core_number(graph)

        # Extraction of the keywords associated with the main core
        keywords = []
        k_max = 0
        for v, k in kcore_number.items():
            if k > k_max:
                keywords.clear()
                k_max = k

            if k == k_max:
                token_code = graph.nodes[v]['label']
                token = self.builder.get_token_(token_code)
                keywords.append((token, k))

        return keywords

    def extract_with_density(self, document: str) -> Sequence[Tuple[str, float]]:
        """Extraction of keywords corresponding to the nodes of the k-core satisfying a density criterion

        Density criterion consists in applying the elbow method when going down the k-core
        """
        # Building the graph-of-words
        gow = self.builder.compute_gow_from_document(document)
        if len(gow.nodes) > 0:
            graph = gow.to_graph()

            # Computation of the k-cores
            if self.builder.weighted:
                kcore_number = core_number_weighted(graph)
            else:
                kcore_number = nx_core_number(graph)

            # Sorted sequence of k for each k-core
            ks = sorted({k for _, k in kcore_number.items()})

            # Storage for (i, density)
            densities = []
            # Mapping between i and the k-core value
            i_to_k = {}
            # Storage of k-core graph for each k
            k_graphs = {}

            # Going DOWN the k-core and computation of the k-core densities
            for i, k in enumerate(reversed(ks)):
                g_k = k_core(graph, k=k, core_number=kcore_number)
                k_graphs[k] = g_k
                i_to_k[i] = k
                densities.append((i, density(g_k)))

            # Retrieving the most appropriate density via the elbow method
            i_k_best = elbow(densities)
            # Retrieving the corresponding k
            k_best = i_to_k[i_k_best]

            # Retrieving the keywords for k-core with k=k_best
            keywords = []
            best_graph = k_graphs[k_best]
            for v in best_graph.nodes:
                token_code = best_graph.nodes[v]['label']
                token = self.builder.get_token_(token_code)
                k = kcore_number[v]
                keywords.append((token, k))

            return sorted(keywords, key=lambda p: p[1], reverse=True)
        else:
            return []

    def extract_with_inflexion(self, document: str) -> Sequence[Tuple[str, float]]:
        """Extraction of keywords corresponding to the nodes of the k-core selected from k-shell size differences

        Going down the k-shell while the size of k-shell keeps increasing, stop otherwise
        """
        # Building the graph-of-words
        gow = self.builder.compute_gow_from_document(document)
        if len(gow.nodes) > 0:
            graph = gow.to_graph()

            # Computation of the k-cores
            if self.builder.weighted:
                kcore_number = core_number_weighted(graph)
            else:
                kcore_number = nx_core_number(graph)

            # Sorted sequence of k for each k-core descending
            ks = sorted({k for _, k in kcore_number.items()}, reverse=True)

            # Going down the k-core while k-shell size is increasing
            k_best = None
            previous = None
            for k1, k2 in zip(ks, ks[1:]):
                g_k1 = k_shell(graph, k=k1, core_number=kcore_number)
                g_k2 = k_shell(graph, k=k2, core_number=kcore_number)
                len_k1 = len(g_k1.nodes)
                len_k2 = len(g_k2.nodes)
                current = len_k2 - len_k1

                if previous is not None:
                    if (previous < 0) and (current > 0):
                        k_best = k2
                        break
                previous = current

            if k_best is None:
                k_best = ks[0]

            # Retrieving the keywords for k-core with k=k_best
            keywords = []
            best_graph = k_core(graph, k=k_best, core_number=kcore_number)
            for v in best_graph.nodes:
                token_code = best_graph.nodes[v]['label']
                token = self.builder.get_token_(token_code)
                k = kcore_number[v]
                keywords.append((token, k))

            return sorted(keywords, key=lambda p: p[1], reverse=True)
        else:
            return []
