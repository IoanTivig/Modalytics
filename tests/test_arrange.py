"""Arranging imported images: grid, by name, custom, stacked."""

from __future__ import annotations

import pytest

from modalytics.core import arrange as ar


def test_natural_order_and_fields():
    names = ["sample01-1-10.txt", "sample01-1-2.txt", "sample01-2-1.txt"]
    assert sorted(names, key=ar.natural_key) == ["sample01-1-2.txt", "sample01-1-10.txt", "sample01-2-1.txt"]
    assert ar.name_fields("sample01-1-2.txt") == ["sample01", "1", "2"]
    assert ar.name_fields("z1/3 Phase") == ["z1", "3", "Phase"]


def test_grid_shape():
    assert ar.grid_shape(7) == (3, 3) and ar.grid_shape(7, columns=2) == (2, 4) and ar.grid_shape(7, rows=2) == (4, 2)
    assert ar.grid_shape(0) == (0, 0)


def test_grid_with_gap_of_the_largest_image():
    sizes = [(100, 50), (200, 80), (100, 50), (100, 50)]
    names = ["d", "c", "b", "a"]
    spots = ar.positions(sizes, names, ar.Layout("grid", columns=2, gap_pct=50, order="name"))
    # cells 300 × 120 µm (largest 200 × 80 plus 50 %); name order a, b, c, d row by row
    assert spots[3] == (0, 0) and spots[2] == (300, 0) and spots[1] == (0, 120) and spots[0] == (300, 120)
    by_col = ar.positions(sizes, names, ar.Layout("grid", rows=2, gap_pct=0, fill="columns"))
    assert by_col[3] == (0, 0) and by_col[2] == (0, 80) and by_col[1] == (200, 0)
    assert ar.bounds(sizes, spots) == (400, 200)                    # the 80-µm-high image is in the 2nd row


def test_rows_and_columns_from_the_names():
    names = ["sample01-2-1", "sample01-1-2", "sample01-1-1", "sample01-1-10", "odd one"]
    sizes = [(10, 10)] * 5
    spots = ar.positions(sizes, names, ar.Layout("names", gap_pct=0, row_field=1, column_field=2))
    assert spots[2] == (0, 0) and spots[1] == (10, 0) and spots[3] == (20, 0)       # area 1: images 1, 2, 10
    assert spots[0] == (0, 10)                                                       # area 2
    assert spots[4] == (0, 20)                                                       # no such fields: last row


def test_custom_rows_and_gaps_and_stack():
    sizes = [(100, 40), (50, 40), (100, 40)]
    lay = ar.Layout("custom", gap_pct=25, custom={0: (0, 1, 0), 1: (0, 0, 50), 2: (1, 0, 0)})
    spots = ar.positions(sizes, ["a", "b", "c"], lay)
    # row 0: image 1 first (then a gap of 50 % of the largest width = 50 µm), then image 0; row 1 one height + 25 %
    assert spots[1] == (0, 0) and spots[0] == (100, 0) and spots[2] == (0, pytest.approx(50))
    assert ar.positions(sizes, ["a", "b", "c"], ar.Layout("stack")) == [(0, 0)] * 3
