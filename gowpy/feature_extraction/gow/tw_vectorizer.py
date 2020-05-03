import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank_numpy
from networkx.algorithms.centrality import degree_centrality, closeness_centrality, betweenness_centrality

from typing import Sequence, Dict

from gowpy.gow.builder import GoWBuilder, Tokenized_document
from gowpy.gow.typing import Tokenizer
from gowpy.utils.defaults import default_tokenizer

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

from operator import itemgetter
import numbers
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


TERM_WEIGHT_DEGREE = "degree"
TERM_WEIGHT_DEGREE_CENTRALITY = "degree_centrality"
TERM_WEIGHT_CLOSENESS_CENTRALITY = "closeness_centrality"
TERM_WEIGHT_BETWEENNESS_CENTRALITY = "betweenness_centrality"
TERM_WEIGHT_PAGERANK = "pagerank"


#
# From: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/feature_extraction/text.py#L820
#
def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


class TwVectorizer(BaseEstimator):
    """Convert a collection of text documents to a matrix of graph-based weight for each token

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore tokens that have a document frequency strictly
        higher than the given threshold (corpus-specific stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
    min_df : float in range [0.0, 1.0] or int, default=1
        Ignore frequent subgraphs that have a document frequency strictly
        lower than the given threshold. This value is also called support
        in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    b : float {0.0, 0.003}, default=0.0
        Slope parameter of the tilting.
    directed : boolean, True by default
        If True, the graph-of-words is directed, else undirected
    window_size : int, default=4
        Size of the window (in token) to build the graph-of-words.
    term_weighting : string {'degree', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'pagerank'}
        Graph-based term weighting approach for the nodes in the graph-of-words
        'degree' (default) : degree (undirected) or indegree (directed) of the nodes.
        'degree_centrality' : normalized degree centrality of the nodes
        'closeness_centrality' : very slow, closeness centrality of the nodes
        'betweenness_centrality' : very slow, the shortest-path betweenness centrality of the nodes
        'pagerank' : slow, the PageRank of the nodes
    tokenizer : callable or None (default)
        Override the string tokenization step.
    """
    def __init__(self,
                 min_df: float = 0.0,
                 max_df: float = 1.0,
                 b: float = 0.0,
                 directed: bool = True,
                 window_size: int = 4,
                 term_weighting: str = TERM_WEIGHT_DEGREE,
                 tokenizer: Tokenizer = None):
        # Subgraph mining patterns
        self.min_df: float = min_df
        self.max_df: float = max_df
        if self.min_df < 0.0:
            raise ValueError("min_df is smaller than 0%")

        if self.max_df < 0.0:
            raise ValueError("max_df is smaller than 0%")

        self.term_weighting = term_weighting

        self.b = b

        self.tokenizer: Tokenizer = tokenizer if tokenizer is not None else default_tokenizer

        self.window_size = window_size
        if self.window_size < 2:
            raise ValueError("window_size < 2")

        self.directed = directed

    def __tw(self, tokens: Tokenized_document) -> Dict[str, int]:
        """Computes the graph-based weight for each token of the document"""
        gow = self.gow_builder_.compute_gow_from_tokenized_document(tokens)
        graph = gow.to_graph()
        tw = {}
        if self.term_weighting == TERM_WEIGHT_DEGREE:
            if graph.is_directed():
                dgraph = nx.DiGraph(graph)
                for (node, degree) in dgraph.in_degree(graph.nodes):
                    token = self.gow_builder_.get_token_(node)
                    tw[token] = degree
            else:
                for (node, degree) in graph.degree(graph.nodes):
                    token = self.gow_builder_.get_token_(node)
                    tw[token] = degree
        else:
            degree_centrality, closeness_centrality, betweenness_centrality
            if self.term_weighting == TERM_WEIGHT_DEGREE_CENTRALITY:
                weighting_fct = degree_centrality
            elif self.term_weighting == TERM_WEIGHT_CLOSENESS_CENTRALITY:
                weighting_fct = closeness_centrality
            elif self.term_weighting == TERM_WEIGHT_BETWEENNESS_CENTRALITY:
                weighting_fct = betweenness_centrality
            elif self.term_weighting == TERM_WEIGHT_PAGERANK:
                weighting_fct = pagerank_numpy
            else:
                weighting_fct = lambda x: 1

            if graph.is_directed():
                dgraph = nx.DiGraph(graph)
                node_to_weight = weighting_fct(dgraph)
                for (node, p) in node_to_weight.items():
                    token = self.gow_builder_.get_token_(node)
                    tw[token] = p
            else:
                node_to_weight = weighting_fct(graph)
                for (node, p) in node_to_weight.items():
                    token = self.gow_builder_.get_token_(node)
                    tw[token] = p
        return tw

    #
    # Largely inspired by the CountVectorizer from scikit-learn
    # See: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/feature_extraction/text.py#L1113
    #
    def __count_vocab(self, tokenized_documents: Sequence[Tokenized_document], fixed_vocab: bool):
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        j_indices = []
        indptr = [0]
        data = []

        for tokens in tokenized_documents:
            feature_counter = {}

            tw = self.__tw(tokens)

            document_length = len(tokens)
            denominator = 1.0 - self.b + self.b * (float(document_length) / self.avdl_)

            for feature in tokens:
                try:
                    feature_idx = vocabulary[feature]

                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = tw[feature] / denominator

                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            data.extend(feature_counter.values())
            indptr.append(len(j_indices))

        # disable defaultdict behaviour
        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        X = sp.csr_matrix((data, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=float)

        X.sort_indices()

        return vocabulary, X

    #
    # Inspired by: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/feature_extraction/text.py#L1058
    #
    def __sort_features(self, X, vocabulary):
        """Sort features by name
        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    #
    # Inspired by: https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/feature_extraction/text.py#L1072
    #
    def __limit_features(self, X, vocabulary, high=None, low=None):
        """Remove too rare or too common features.
        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.
        This does not prune samples with zero features.
        """
        if high is None and low is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], removed_terms

    def fit(self, raw_documents: Sequence[str], y=None):
        self.fit_transform(raw_documents, y)
        return self

    def fit_transform(self, raw_documents: Sequence[str], y=None):
        max_df = self.max_df
        min_df = self.min_df

        self.gow_builder_ = GoWBuilder(window_size=self.window_size,
                                       directed=self.directed,
                                       tokenizer=self.tokenizer)
        N = len(raw_documents)
        self.N_ = N

        avdl = 0.0
        tokenized_documents = []
        for document in raw_documents:
            tok_document = self.tokenizer(document)
            tokenized_documents.append(tok_document)
            avdl += len(tok_document)

        avdl = avdl / float(N)
        self.avdl_ = avdl

        vocabulary, X = self.__count_vocab(tokenized_documents, fixed_vocab=False)
        X = self.__sort_features(X, vocabulary)

        max_doc_count = (max_df
                         if isinstance(max_df, numbers.Integral)
                         else max_df * N)
        min_doc_count = (min_df
                         if isinstance(min_df, numbers.Integral)
                         else min_df * N)

        X, self.stop_words_ = self.__limit_features(X, vocabulary, max_doc_count, min_doc_count)

        self.vocabulary_ = vocabulary

        return X

    def transform(self, raw_documents: Sequence[str]):
        _, X = self.__count_vocab([self.tokenizer(doc) for doc in raw_documents], fixed_vocab=True)

        return X

    def get_feature_names(self) -> Sequence[str]:
        return [t for t, i in sorted(self.vocabulary_.items(), key=itemgetter(1))]

    def _more_tags(self):
        return {'X_types': ['string']}


class TwidfVectorizer(BaseEstimator):
    """Convert a collection of text documents to a TW-IDF matrix

    Equivalent to :class:`TwVectorizer` followed by
    :class:`TfidfTransformer`.

    This implementation produces a sparse representation of the counts using
    scipy.sparse.csr_matrix.

    Parameters
    ----------
    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore tokens that have a document frequency strictly
        higher than the given threshold (corpus-specific stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
    min_df : float in range [0.0, 1.0] or int, default=1
        Ignore frequent subgraphs that have a document frequency strictly
        lower than the given threshold. This value is also called support
        in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    b : float {0.0, 0.003}, default=0.0
        Slope parameter of the tilting.
    directed : boolean, True by default
        If True, the graph-of-words is directed, else undirected
    window_size : int, default=4
        Size of the window (in token) to build the graph-of-words.
    term_weighting : string {'degree', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality', 'pagerank'}
        Graph-based term weighting approach for the nodes in the graph-of-words
        'degree' (default) : degree (undirected) or indegree (directed) of the nodes.
        'degree_centrality' : normalized degree centrality of the nodes
        'closeness_centrality' : very slow, closeness centrality of the nodes
        'betweenness_centrality' : very slow, the shortest-path betweenness centrality of the nodes
        'pagerank' : slow, the PageRank of the nodes
    tokenizer : callable or None (default)
        Override the string tokenization step.
    norm : 'l1', 'l2' or None, optional (default='l2')
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`
    use_idf : boolean (default=True)
        Enable inverse-document-frequency reweighting.
    smooth_idf : boolean (default=True)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    """

    def __init__(self,
                 min_df: float = 0.0,
                 max_df: float = 1.0,
                 b: float = 0.0,
                 directed: bool = True,
                 window_size: int = 4,
                 term_weighting: str = TERM_WEIGHT_DEGREE,
                 tokenizer: Tokenizer = None,
                 #
                 norm='l2',
                 use_idf=True,
                 smooth_idf=True):
        # Subgraph mining patterns
        self.min_df: float = min_df
        self.max_df: float = max_df
        if self.min_df < 0.0:
            raise ValueError("min_df is smaller than 0%")

        if self.max_df < 0.0:
            raise ValueError("max_df is smaller than 0%")

        self.term_weighting = term_weighting

        self.b = b

        self.tokenizer: Tokenizer = tokenizer if tokenizer is not None else default_tokenizer

        self.window_size = window_size
        if self.window_size < 2:
            raise ValueError("window_size < 2")

        self.directed = directed

        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf

    def fit(self, raw_documents: Sequence[str], y=None):
        self.fit_transform(raw_documents, y)
        return self

    def fit_transform(self, raw_documents: Sequence[str], y=None):
        self.tw_vectorizer_ = TwVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            b=self.b,
            directed=self.directed,
            window_size=self.window_size,
            term_weighting=self.term_weighting,
            tokenizer=self.tokenizer)
        self.tfidf_transformer_ = TfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf)

        self.pipeline_ = Pipeline([
            ('tw', self.tw_vectorizer_),
            ('idf', self.tfidf_transformer_)
        ])
        return self.pipeline_.fit_transform(raw_documents, y)

    def transform(self, raw_documents: Sequence[str]):
        return self.pipeline_.transform(raw_documents)

    def get_feature_names(self) -> Sequence[str]:
        return [t for t, i in sorted(self.tw_vectorizer_.vocabulary_.items(), key=itemgetter(1))]

    def _more_tags(self):
        return {'X_types': ['string']}
