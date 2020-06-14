# CHANGELOG

## [0.2.0] - 2020-06-14
This release adds new algorithms for keyword extraction, 
adds some example notebooks and 
fixes some bugs.

### Graph-of-words representation
Addition of edge weighting option in graph-of-words. The weight of an edge is
the co-occurrence count of the tokens.

### Keyword Extraction
#### Implementation of the dense and inflexion k-core selection methods
k-core approaches allow the selection of cohesive keywords. Selected keywords 
correspond to a cohesive subgraph. In other words, the granularity of selection 
is at the level of cohesive subgraphs and nodes are selected by entire batch at 
a time. A key property is the selection of an automatically adaptive number of 
cohesive keywords.

Three selection methods are now available based on the k-core decomposition
of the graph-of-words.
The `'maximum'` method simply selects the main core (the k-core with maximum k).
This is the default method. It can be viewed as being too restrictive. Two other
selection methods alleviate this limitation.
On one hand, the `'density'` method goes down the hierarchy of k-cores to select 
the one retaining the cohesiveness from the perspective of the density of the 
k-core. The most appropriate k-core is selected via the elbow method.
On the other hand, the `'inflexion'` method exploits the k-shell (which is the part 
of the k-core that does not survive in the (k+1)-core). It consists in going down 
the hierachy of k-cores as long as the shells increase in size, else stopping.

Example on `'density'` method:
```python
from gowpy.summarization.unsupervised import KcoreKeywordExtractor

extractor_kw = KcoreKeywordExtractor(directed=False, weighted=True, window_size=4,
                                     # Parameter to set the selection method
                                     selection_method='density')
```

#### Implementation of the CoreRank method
The CoreRank method extracts keywords from a text document at the node-level of 
a graph-of-words representation.
Each node/token in the graph-of-words is associated with a score, namely the sum 
of the core numbers of its neighbors. Then each node is ranked in decreasing 
order of score.

This extractor allows both the selection of an automatically adaptive number of keywords and the
selection of given number or proportion of keywords.

Example usage:
```python
from gowpy.summarization.unsupervised import CoreRankKeywordExtractor

extractor_kw_cr = CoreRankKeywordExtractor(directed=False, weighted=True, window_size=4)

preprocessed_text = "..."  # preprocessed text in which to extract keywords

extractor_kw_cr.extract(preprocessed_text, n=5)
```

#### Graph Algorithm
- Implementation of the generalized core algorithm for weighted graphs
  (i.e. k-core algorithm for weighted graphs) from 
  ["Generalized Cores" V. Batagelj, M. Zaveršnik (2002)](https://arxiv.org/abs/cs/0202039)

### Frequent Subgraphs
- The `GoWMiner` can now be used to incrementally load results of more than one 
  subgraph mining process.
- Fix of a bug in the computation of the sparse matrix in the 
  `GoWVectorizer` vectorizer.

### Misc
- Addition of example notebooks
- Update of the documentation

## [0.1.0] - 2020-05-03
* Public release of the first version of the library
