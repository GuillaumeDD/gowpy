#
# Test of the elbow helper
#
import pytest

from gowpy.summarization.unsupervised.helpers import elbow

def test_one_point():
    # Sequence of (x, y) points
    points = [(1, 6)]

    expected_result = 1
    result = elbow(points)

    assert expected_result == result

def test_straight_line():
    # Sequence of (x, y) points
    points = [(42, 6), (43, 6), (44, 6)]

    expected_result = 42
    result = elbow(points)

    assert expected_result == result

def test_decreasing_line():
    # Sequence of (x, y) points
    #   ^
    # 6 | +
    # 5 |
    # 4 |   +
    # 3 |
    # 2 |     +
    # 1 |         +
    # 0 |               +
    #    -------------------->
    #     1 2 3 4 5 6 7 8
    #########################
    points = [(1, 6), (2, 4), (3, 2), (5, 1), (8, 0)]

    expected_result = 3
    result = elbow(points)

    assert expected_result == result
