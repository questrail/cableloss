# -*- coding: utf-8 -*-
# Copyright (c) 2017 The cableloss developers. All rights reserved.
# Project site: https://github.com/questrail/cableloss
# Use of this source code is governed by a MIT-style license that
# can be found in the LICENSE.txt file for the project.
"""Unit tests for cableloss.py.
"""

# Try to future proof code so that it's Python 3.x ready
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

# Standard module imports
import unittest

import numpy as np

import cableloss


class TestCalculatingCableLoss(unittest.TestCase):

    def setUp(self):
        # Setup the known data arrays

        self.my_data_type = np.dtype([
            (b'frequency', b'f8'), (b'amplitude_db', b'f8')])

        #  self.my_data_type = {'names': ('frequency', 'amplitude_db'),
        #  'formats': ('f8', 'f8')}

    def test_rg_58_at_100_ft(self):
        expected_cable_loss = np.array([
            (1.00e6, 0.4),
            (1.00e7, 1.4),
            (5.00e7, 3.3),
            (1.00e8, 4.9),
            (2.00e8, 7.3),
            (4.00e8, 11.2),
            (7.00e8, 16.9),
            (9.00e8, 20.1),
            (1.00e9, 21.5)], dtype=self.my_data_type)
        calculated_cable_loss = cableloss.loss('RG-58', 100)
        np.testing.assert_array_equal(
            calculated_cable_loss['frequency'],
            expected_cable_loss['frequency'])
        np.testing.assert_array_almost_equal(
            calculated_cable_loss['amplitude_db'],
            expected_cable_loss['amplitude_db'])


if __name__ == '__main__':
    unittest.main()
