# -*- coding: utf-8 -*-
# Copyright (c) 2017-2022 The cableloss developers. All rights reserved.
# Project site: https://github.com/questrail/cableloss
# Use of this source code is governed by a MIT-style license that
# can be found in the LICENSE.txt file for the project.
"""Provide the cable loss for the specified cable type and length.

The length should be specified in feet.
"""

# Standard module imports

# Data analysis related imports
import numpy as np

__version__ = '0.2.0'


def loss(cable_type, length):
    # type: (str, Union[int, float]) -> np.array # noqa
    """Return the cable loss based on the cable type and length in feet

    args:
        cable_type: A string listing the cable type
        length: A number in feet

    returns:
        A numpy array listing the frequency in Hz and amplitude in dB
    """

    data_type = [('frequency', np.float64), ('amplitude_db', np.float64)]

    # The following cable losses are for 100 ft of cable.
    cable_loss_100_ft = {
        'RG-58': np.array([
            (1.00e6, 0.4),
            (1.00e7, 1.4),
            (5.00e7, 3.3),
            (1.00e8, 4.9),
            (2.00e8, 7.3),
            (4.00e8, 11.2),
            (7.00e8, 16.9),
            (9.00e8, 20.1),
            (1.00e9, 21.5)], dtype=data_type),
        'RG-58/U': np.array([
            (9.00e3, 0.0384),
            (1.00e4, 0.0404),
            (2.50e4, 0.0640),
            (5.00e4, 0.0906),
            (1.00e5, 0.1283),
            (5.00e5, 0.2873),
            (1.00e6, 0.4067),
            (5.00e6, 0.9111),
            (1.00e7, 1.2896),
            (5.00e7, 3.0000),
            (7.50e7, 3.5403),
            (1.00e8, 4.4),
            (1.50e8, 5.0109),
            (2.00e8, 6.00),
            (3.00e8, 7.0924),
            (4.00e8, 8.5),
            (5.00e8, 9.1619),
            (6.00e8, 10.0385),
            (7.00e8, 10.8448),
            (8.00e8, 11.5955),
            (9.00e8, 13.00),
            (1.00e9, 13.00)], dtype=data_type),
        'LMR-195': np.array([
            (3.00e5, 0.2000),
            (3.00e6, 0.6000),
            (3.00e7, 2.0000),
            (5.00e7, 2.5000),
            (1.50e8, 4.4000),
            (2.20e8, 5.4000),
            (4.50e8, 7.8000),
            (9.00e8, 11.100),
            (1.50e9, 14.500)], dtype=data_type),
        'LMR-400': np.array([
            (9.00e3, 0.0112),
            (1.00e4, 0.0118),
            (2.50e4, 0.0188),
            (5.00e4, 0.0267),
            (1.00e5, 0.0380),
            (5.00e5, 0.0860),
            (1.00e6, 0.1223),
            (5.00e6, 0.2771),
            (1.00e7, 0.3940),
            (3.00e7, 0.7000),
            (5.00e7, 0.9000),
            (7.50e7, 1.0968),
            (1.00e8, 1.2695),
            (1.50e8, 1.5000),
            (2.00e8, 1.8054),
            (2.20e8, 1.9000),
            (3.00e8, 2.2185),
            (4.00e8, 2.5767),
            (4.50e8, 2.7000),
            (5.00e8, 2.8759),
            (6.00e8, 3.1550),
            (7.00e8, 3.4121),
            (8.00e8, 3.6516),
            (9.00e8, 3.9000),
            (1.00e9, 4.0900),
            (1.50e9, 5.1000),
            (1.80e9, 5.7000),
            (2.00e9, 6.0000),
            (2.50e9, 6.8000),
            (5.80e9, 10.800)], dtype=data_type),
    }

    loss = cable_loss_100_ft[cable_type]
    loss['amplitude_db'] = loss['amplitude_db'] * length / 100

    return loss
