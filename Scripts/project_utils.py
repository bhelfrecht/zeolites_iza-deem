#!/usr/bin/env python

import os
import sys
import numpy as np
import h5py

def load_structures_from_hdf5(filename, datasets=None, concatenate=False):
    """
        Load structure-based data from an HDF5 file
    """

    structure_values = []

    f = h5py.File(filename, 'r')

    if datasets is not None:
        for dataset_name in datasets:
            structure_values.append(f[dataset_name][:])
    else:
        for structure_value in f.values():
            structure_values.append(structure_value)

    f.close()

    if concatenate:
        structure_values = np.vstack(structure_values)

    return structure_values

def build_representatives_from_hdf5(filename, representative_idxs, split_idxs):
    """
        Build a set of representative environments
    """

    representatives = []

    f = h5py.File(filename, 'r')

    for i in representative_idxs:
        structure_number = np.nonzero((split_idxs-i) > 0)[0][0]
        n = i - split_idxs
        environment_number = n[structure_number-1]
        representatives.append(f[f'{structure_number}'][environment_number, :])

    f.close()

    representatives = np.asarray(representatives)

    return representatives
