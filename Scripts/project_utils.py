#!/usr/bin/env python

import os
import sys
import numpy as np
import h5py
from sklearn.utils.multiclass import _ovr_decision_function

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
            structure_values.append(structure_value[:])

    f.close()

    if concatenate:
        structure_values = np.vstack(structure_values)

    return structure_values

def df_to_class(df, df_type, n_classes, use_df_sums=True):
    """
        Make class predictions based on a decision function.
        Uses the sci-kit learn conversion from OVO to OVR

        ---Arguments---
        df: decision function on which to make class predictions
        df_type: decision function type, 'ovo' or 'ovr'
        n_classes: number of integer classes

        ---Returns---
        predicted_class: predicted integer class
    """

    # Approximation to the number of classes, should be valid up to at least 1M
    #n_classes = int(np.sqrt(2*df.shape[-1])) + 1

    if n_classes > 2:
        if df_type == 'ovo':

            # Convert OVO decision function to OVR to determine class
            df_ovr = _ovr_decision_function(df < 0.0, -df, n_classes)

        elif df_type == 'ovr':
            df_ovr = df

        else:
            print("Error: invalid decision function. Use 'ovo' or 'ovr'")
            return

        # Predicted class determined by the largest value of the OVR decision function
        predicted_class = np.argmax(df_ovr, axis=1) + 1

    else:
        predicted_class = np.zeros(df.shape[0], dtype=int)

        # This appears to be the convention, which is "opposite" of that above
        # Default exactly zero decision function value to the "positive" class
        predicted_class[df >= 0] = 2
        predicted_class[df < 0] = 1

    return predicted_class
