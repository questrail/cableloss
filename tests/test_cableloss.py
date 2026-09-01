# Copyright (c) 2017-2026 The cableloss developers. All rights reserved.
# Project site: https://github.com/questrail/cableloss
# Use of this source code is governed by a MIT-style license that
# can be found in the LICENSE.txt file for the project.
"""Unit tests for cableloss.py."""

import numpy as np
import pytest

import cableloss

# The published loss per 100 ft, in dB, at a frequency each cable's table
# gives outright. Read off the tables rather than computed from them, so that
# a typo introduced into the data has something independent to fail against.
KNOWN_LOSS_AT_100_FT = [
    ("RG-58", 1.00e6, 0.4),
    ("RG-58", 1.00e9, 21.5),
    ("RG-58/U", 9.00e3, 0.0384),
    ("RG-58/U", 1.00e9, 13.00),
    ("LMR-195", 3.00e5, 0.2000),
    ("LMR-195", 1.50e9, 14.500),
    ("LMR-400", 9.00e3, 0.0112),
    ("LMR-400", 5.80e9, 10.800),
]

CABLE_TYPES = ["RG-58", "RG-58/U", "LMR-195", "LMR-400"]


@pytest.fixture
def my_data_type():
    return np.dtype([("frequency", np.float64), ("amplitude_db", np.float64)])


def test_rg_58_at_100_ft(my_data_type):
    expected_cable_loss = np.array(
        [
            (1.00e6, 0.4),
            (1.00e7, 1.4),
            (5.00e7, 3.3),
            (1.00e8, 4.9),
            (2.00e8, 7.3),
            (4.00e8, 11.2),
            (7.00e8, 16.9),
            (9.00e8, 20.1),
            (1.00e9, 21.5),
        ],
        dtype=my_data_type,
    )
    calculated_cable_loss = cableloss.loss("RG-58", 100)
    np.testing.assert_array_equal(
        calculated_cable_loss["frequency"], expected_cable_loss["frequency"]
    )
    np.testing.assert_array_almost_equal(
        calculated_cable_loss["amplitude_db"], expected_cable_loss["amplitude_db"]
    )


@pytest.mark.parametrize(
    ("cable_type", "frequency", "expected_db"), KNOWN_LOSS_AT_100_FT
)
def test_published_loss_at_100_ft(cable_type, frequency, expected_db):
    cable_loss = cableloss.loss(cable_type, 100)
    row = cable_loss[cable_loss["frequency"] == frequency]
    assert len(row) == 1
    assert row["amplitude_db"][0] == pytest.approx(expected_db)


@pytest.mark.parametrize("cable_type", CABLE_TYPES)
@pytest.mark.parametrize("length", [0, 1, 25, 50, 200, 33.3])
def test_loss_scales_linearly_with_length(cable_type, length):
    """The tables are per 100 ft, and loss in dB is linear in length."""
    at_100_ft = cableloss.loss(cable_type, 100)
    at_length = cableloss.loss(cable_type, length)
    np.testing.assert_array_equal(at_length["frequency"], at_100_ft["frequency"])
    np.testing.assert_allclose(
        at_length["amplitude_db"], at_100_ft["amplitude_db"] * length / 100
    )


@pytest.mark.parametrize("cable_type", CABLE_TYPES)
def test_returned_dtype(cable_type, my_data_type):
    assert cableloss.loss(cable_type, 100).dtype == my_data_type


@pytest.mark.parametrize("cable_type", CABLE_TYPES)
def test_frequencies_ascend(cable_type):
    """Interpolating callers rely on the frequency column being sorted."""
    frequencies = cableloss.loss(cable_type, 100)["frequency"]
    assert np.all(np.diff(frequencies) > 0)


@pytest.mark.parametrize("cable_type", CABLE_TYPES)
def test_repeated_calls_are_independent(cable_type):
    """A call scales its own copy, not the table the next call reads."""
    first = cableloss.loss(cable_type, 50)
    second = cableloss.loss(cable_type, 50)
    np.testing.assert_array_equal(first["amplitude_db"], second["amplitude_db"])


def test_unknown_cable_type():
    with pytest.raises(KeyError):
        cableloss.loss("no-such-cable", 100)
