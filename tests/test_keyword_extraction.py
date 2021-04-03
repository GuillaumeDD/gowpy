#
# Test of keyword Extraction based on Graph-of-Words
#
# Test examples come from the following paper:
# A Graph Degeneracy-based Approach to Keyword Extraction. Tixier, Antoine, Malliaros, Fragkiskos, and
# Vazirgiannis, Michalis. Proceedings of the 2016 Conference on Empirical Methods in Natural Language
# Processing. (EMNLP 2016)
#
# It exemplifies the results of these alternative methods:
#     Batch keyword extraction based on k-core
#         main core
#         k-core + dense selection method
#         k-core + inflexion selection method
#     Word-level keyword extraction
#         CoreRank + elbow method
#         CoreRank + top 33%
#

import pytest

from gowpy.summarization.unsupervised import KcoreKeywordExtractor
from gowpy.summarization.unsupervised import CoreRankKeywordExtractor

# """
# Mathematical aspects of computer-aided share trading. We consider
# problems of statistical analysis of share prices and propose
# probabilistic characteristics to describe the price series.
# We discuss three methods of mathematical modelling of price
# series with given probabilistic characteristics.
# """
_preprocessed_text = """
Mathemat aspect computer-aid  share trade  problem 
statist analysi share price probabilist characterist price  
seri method mathemat model  price   seri probabilist
characterist
""".strip().lower()


def test_keyword_extraction_kcore():
    expected_result = [('mathemat', 11),
                       ('method', 11),
                       ('model', 11),
                       ('probabilist', 11),
                       ('price', 11),
                       ('characterist', 11),
                       ('seri', 11)]

    extractor_kw = KcoreKeywordExtractor(directed=False, weighted=True, window_size=8)
    gowpy_result = extractor_kw.extract(_preprocessed_text)

    assert expected_result == gowpy_result


def test_keyword_extraction_density():
    expected_result = [('mathemat', 11),
                       ('price', 11),
                       ('probabilist', 11),
                       ('characterist', 11),
                       ('seri', 11),
                       ('method', 11),
                       ('model', 11),
                       ('share', 10)]

    extractor_kw = KcoreKeywordExtractor(directed=False, weighted=True, window_size=8,
                                         selection_method='density')
    gowpy_result = extractor_kw.extract(_preprocessed_text)

    assert expected_result == gowpy_result


def test_keyword_extraction_inflexion():
    expected_result = [('mathemat', 11),
                       ('price', 11),
                       ('probabilist', 11),
                       ('characterist', 11),
                       ('seri', 11),
                       ('method', 11),
                       ('model', 11),
                       ('share', 10),
                       ('trade', 9),
                       ('problem', 9),
                       ('statist', 9),
                       ('analysi', 9)]

    extractor_kw = KcoreKeywordExtractor(directed=False, weighted=True, window_size=8,
                                         selection_method='inflexion')
    gowpy_result = extractor_kw.extract(_preprocessed_text)

    assert expected_result == gowpy_result


def test_keyword_extraction_corerank_elbow():
    expected_result = [('mathemat', 128),
                       ('price', 120),
                       ('analysi', 119),
                       ('share', 118),
                       ('probabilist', 112),
                       ('characterist', 112),
                       ('statist', 108),
                       ('trade', 97),
                       ('problem', 97),
                       ('seri', 94)]

    extractor_kw_cr = CoreRankKeywordExtractor(directed=False, weighted=True, window_size=8)
    gowpy_result = extractor_kw_cr.extract(_preprocessed_text)

    assert expected_result == gowpy_result


def test_keyword_extraction_corerank_firstier():
    expected_result = [('mathemat', 128),
                       ('price', 120),
                       ('analysi', 119),
                       ('share', 118),
                       ('probabilist', 112)]

    extractor_kw_cr = CoreRankKeywordExtractor(directed=False, weighted=True, window_size=8, n=0.33)
    gowpy_result = extractor_kw_cr.extract(_preprocessed_text)

    assert expected_result == gowpy_result
