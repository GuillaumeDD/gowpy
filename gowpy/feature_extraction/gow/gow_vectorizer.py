import networkx.algorithms.isomorphism as iso
from networkx.algorithms import isomorphism

import numbers

from scipy.sparse import csr_matrix

from typing import Sequence, Tuple, Generator

from gowpy.gow.builder import GraphOfWords
from gowpy.gow.typing import Nodes

from sklearn.base import BaseEstimator
from gowpy.gow.miner import GoWMiner

SUBGRAPH_MATCHING_INDUCED = "induced"
SUBGRAPH_MATCHING_PARTIAL = "partial"


class GoWVectorizer(BaseEstimator):
    """Convert a collection of text documents to a matrix of frequent subgraphs matching counts

    Frequent subgraphs have to be mined before using this vectorizer.

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    graph_of_words: GoWMiner
        A graph-of-words miner containing the frequent subgraphs.
    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore frequent subgraphs that have a document frequency strictly
        higher than the given threshold (corpus-specific frequent subgraphs).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
    min_df : float in range [0.0, 1.0] or int, default=0.0
        Ignore frequent subgraphs that have a document frequency strictly
        lower than the given threshold. This value is also called support
        in the literature.
        Note that the smallest value is defined with the support used when
        mining frequent subgraphs.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    indexing : boolean, True by default
        Frequent subgraphs are indexed for faster retrieval when computing
        document features.
    subgraph_matching : string {'induced', 'partial'}
        Frequent subgraph matching approach
        'partial' (default) : subgraph matching corresponding to node and
        edge inclusion.
        'induced' : slower approach, node-induced subgraph matching
    """

    def __init__(self,
                 graph_of_words: GoWMiner,
                 min_df: float = 0.0,
                 max_df: float = 1.0,
                 subgraph_matching: str = SUBGRAPH_MATCHING_PARTIAL,
                 indexing: bool = True):
        self.graph_of_words = graph_of_words

        # Subgraph mining patterns
        self.min_df: float = min_df
        self.max_df: float = max_df
        if self.min_df < 0.0:
            raise ValueError("min_df is smaller than 0%")

        if self.max_df < 0.0:
            raise ValueError("max_df is smaller than 0%")

        self.subgraph_matching: str = subgraph_matching

        if self.graph_of_words is None:
            raise ValueError("No provided graph-of-words miner to compute features (graph_of_words is None)")

        self.indexing = indexing

    def __compute_subpatterns(self) -> Sequence[Tuple[int, GraphOfWords]]:
        # Filtering patterns out by support
        if self.graph_of_words is not None:
            max_doc_count = (self.max_df / float(self.graph_of_words.corpus_size)
                             if isinstance(self.max_df, numbers.Integral)
                             else self.max_df)
            min_doc_count = (self.min_df / float(self.graph_of_words.corpus_size)
                             if isinstance(self.min_df, numbers.Integral)
                             else self.min_df)
            # Selecting subpatterns
            subpatterns = [subgraph for subgraph in self.graph_of_words.frequent_subgraphs
                           if (float(subgraph.freq) / float(self.graph_of_words.corpus_size)) >= min_doc_count
                           if (float(subgraph.freq) / float(self.graph_of_words.corpus_size)) <= max_doc_count
                           ]
        else:
            subpatterns = []

        if self.indexing:
            # Indexing patterns by node codes
            self.node_code_to_feature_i_s_ = {}
            for feature_i, subgraph in enumerate(subpatterns):
                for node_code in subgraph.nodes:
                    if node_code not in self.node_code_to_feature_i_s_:
                        self.node_code_to_feature_i_s_[node_code] = set()

                    self.node_code_to_feature_i_s_[node_code].add(feature_i)

        return [(i, subgraph) for i, subgraph in enumerate(subpatterns)]

    def fit(self, raw_documents: Sequence[str], y=None):
        self.selected_subpatterns_: Sequence[Tuple[int, GraphOfWords]] = self.__compute_subpatterns()
        self.node_matcher_ = iso.categorical_node_match('label', -1)

        return self

    def fit_transform(self, raw_documents: Sequence[str], y=None):
        self.fit(raw_documents, y)
        return self.transform(raw_documents)

    def __get_probable_features_via_nodes(self, document_nodes: Nodes) -> Generator[
        Tuple[int, GraphOfWords], None, None]:
        subpatterns = self.selected_subpatterns_

        feature_i_s = set()
        for node_code in document_nodes:
            if node_code in self.node_code_to_feature_i_s_:
                # Getting the feature indices in which the node code appears
                temp_feature_i_s = self.node_code_to_feature_i_s_[node_code]
                feature_i_s.update(temp_feature_i_s)

        for feature_i in sorted(feature_i_s):
            _, subgraph = subpatterns[feature_i]
            yield (feature_i, subgraph)

    def __iterate_over_features(self, document_nodes: Nodes) -> Generator[Tuple[int, GraphOfWords], None, None]:
        if self.indexing:
            return self.__get_probable_features_via_nodes(document_nodes)
        else:
            subpatterns = self.selected_subpatterns_
            return subpatterns

    def __is_iso_induced(self,
                         feature_gow: GraphOfWords,
                         document_gow: GraphOfWords) -> bool:
        is_iso = False
        document_nodes = document_gow.nodes
        document_edges = document_gow.edges

        # optimisation:
        # checking nodes and edges inclusion in document before running
        # subgraph matching algorithms
        #
        if (feature_gow.nodes.issubset(document_nodes)) and \
                (feature_gow.edges.issubset(document_edges)):
            if len(feature_gow.nodes) <= 2:
                is_iso = True
            else:
                document_graph = document_gow.to_graph()
                feature_graph = feature_gow.to_graph()
                GM = isomorphism.GraphMatcher(document_graph, feature_graph,
                                              node_match=self.node_matcher_)
                is_iso = GM.subgraph_is_isomorphic()

        return is_iso

    @staticmethod
    def __is_iso_partial(feature_gow: GraphOfWords,
                         document_gow: GraphOfWords) -> bool:
        return (feature_gow.nodes.issubset(document_gow.nodes)) and \
               (feature_gow.edges.issubset(document_gow.edges))

    def transform(self, raw_documents: Sequence[str]):
        indptr = [0]
        indices = []
        data = []

        subpatterns = self.selected_subpatterns_
        temp_num_features = len(subpatterns)

        if temp_num_features > 0:
            for document in raw_documents:
                # Document to gowpy
                document_gow = self.graph_of_words.compute_gow_from_document(document)
                if self.subgraph_matching == SUBGRAPH_MATCHING_INDUCED:
                    # Feature computation
                    retained_features = [i_feature
                                         for i_feature, feature_gow in self.__iterate_over_features(document_gow.nodes)
                                         if self.__is_iso_induced(feature_gow, document_gow)
                                         ]
                else:
                    # Feature computation
                    retained_features = [i_feature
                                         for i_feature, feature_gow in self.__iterate_over_features(document_gow.nodes)
                                         if GoWVectorizer.__is_iso_partial(feature_gow, document_gow)
                                         ]

                # Building blocks of the sparse matrix
                for i_feature in retained_features:
                    indices.append(i_feature)
                    data.append(1)
                indptr.append(len(indices))

            resulting_matrix = csr_matrix((data, indices, indptr), dtype=int)
        else:
            resulting_matrix = csr_matrix((len(raw_documents), 0))
        return resulting_matrix

    def get_feature_names(self) -> Sequence[str]:
        feature_names = []

        subpatterns = self.selected_subpatterns_

        for _, subgraph in subpatterns:
            temp = []
            for n in subgraph.nodes_str():
                temp.append(n)
            for e in subgraph.edges_str():
                temp.append(e)

            feature_names.append(' '.join(temp))

        return feature_names

    def _more_tags(self):
        return {'X_types': ['string']}
