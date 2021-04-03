#
# Test for the TW vectorizer
#

import pytest

from gowpy.feature_extraction.gow import TwVectorizer
import numpy as np

#
# Example from the following paper:
# Graph-of-word and TW-IDF: New Approach to Ad Hoc IR. Rousseau, François, and Vazirgiannis, Michalis. Proceedings of
# the 22nd ACM international conference on Information & Knowledge Management.(CIKM 2013)
#
_text = """information retrieval is the activity of obtaining information resources relevant to an information need 
from a collection of information resources"""

def test_tw_vectorizer_directed():
    vectorizer_gow = TwVectorizer(
        # Graph-of-words specificities
        directed=True,
        window_size=3,
        # Token frequency filtering
        min_df=0.0,
        max_df=1.0,
        # Graph-based term weighting approach
        term_weighting='degree'
    )

    X = vectorizer_gow.fit_transform([_text])

    # Test: feature computation
    tokens = _text.split()
    gowpy_result = vectorizer_gow.get_feature_names()
    assert all(token in gowpy_result for token in tokens)

    # Test: weighting schema
    expected_result = [('information', 5),
                       ('retrieval', 1),
                       ('is', 2),
                       ('the', 2),
                       ('activity', 2),
                       ('of', 4),
                       ('obtaining', 2),
                       ('resources', 3),
                       ('relevant', 2),
                       ('to', 2),
                       ('an', 2),
                       ('need', 2),
                       ('from', 2),
                       ('a', 2),
                       ('collection', 2), ]

    feature_names = vectorizer_gow.get_feature_names()
    gowpy_result = [(feature, w) for w, feature in zip(X.toarray()[0], feature_names)]
    assert len(expected_result) == len(gowpy_result), \
        "Number of features does not correspond between expectation and what is computed"
    assert all([feature_w in gowpy_result for feature_w in expected_result])


def test_tw_vectorizer_undirected():
    vectorizer_gow = TwVectorizer(
        # Graph-of-words specificities
        directed=False,
        window_size=3,
        # Token frequency filtering
        min_df=0.0,
        max_df=1.0,
        # Graph-based term weighting approach
        term_weighting='degree'
    )

    X = vectorizer_gow.fit_transform([_text])

    # Test: feature computation
    tokens = _text.split()
    gowpy_result = vectorizer_gow.get_feature_names()
    assert all(token in gowpy_result for token in tokens)

    # Test: weighting schema
    expected_result = [('information', 11),
                       ('retrieval', 3),
                       ('is', 4),
                       ('the', 4),
                       ('activity', 4),
                       ('of', 7),
                       ('obtaining', 4),
                       ('resources', 5),
                       ('relevant', 4),
                       ('to', 4),
                       ('an', 4),
                       ('need', 4),
                       ('from', 4),
                       ('a', 4),
                       ('collection', 4), ]

    feature_names = vectorizer_gow.get_feature_names()
    gowpy_result = [(feature, w) for w, feature in zip(X.toarray()[0], feature_names)]
    assert len(expected_result) == len(gowpy_result), \
        "Number of features does not correspond between expectation and what is computed"
    assert all([feature_w in gowpy_result for feature_w in expected_result])
