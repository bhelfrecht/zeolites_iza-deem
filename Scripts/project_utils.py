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

def load_soaps(deem_file, iza_file,
               idxs_deem_train, idxs_deem_test,
               idxs_iza_train, idxs_iza_test,
               idxs_deem_delete=[], idxs_iza_delete=[]):

    # Load SOAPs
    soaps_deem = load_structures_from_hdf5(deem_file, datasets=None, concatenate=False)
    for i in sorted(idxs_deem_delete, reverse=True):
        soaps_deem.pop(i)

    soaps_iza = load_structures_from_hdf5(iza_file, datasets=None, concatenate=False)
    for i in sorted(idxs_iza_delete, reverse=True):
        soaps_iza.pop(i)

    # Build the train and test sets
    deem_train = [soaps_deem[i] for i in idxs_deem_train]
    deem_test = [soaps_deem[i] for i in idxs_deem_test]
    iza_train = [soaps_iza[i] for i in idxs_iza_train]
    iza_test = [soap_iza[i] for i in idxs_iza_test]

    soaps_train = iza_train + deem_train
    soaps_test = iza_test + deem_test

    return soaps_train, soaps_test

def preprocess_soaps(soaps_train, soaps_test):

    # Can also do other scaling/centering here --
    # this is mostly just to get the SOAPs to a 'usable' magnitude
    soaps_scale = np.std(soaps_train)
    soaps_train_scaled = soaps_train / soaps_scale
    soaps_test_scaled = soaps_test / soaps_scale

    return soaps_train, soaps_test

# TODO: rename these functions
def load_decision_functions(deem_file, iza_file,
                            idxs_deem_train, idxs_deem_test,
                            idxs_iza_train, idxs_iza_test):

    deem_data = np.loadtxt(deem_file)
    iza_data = np.loadtxt(iza_file)

    deem_data_train = deem_data[idxs_deem_train]
    deem_data_test = deem_data[idxs_deem_test]

    iza_data_train = iza_data[idxs_iza_train]
    iza_data_test = iza_data[idxs_iza_test]

    train_data = np.concatenate((iza_data_train, deem_data_train))
    test_data = np.concatenate((iza_data_test, deem_data_test))

    return train_data, test_data

def load_kernels(kernel_file):

    # Load the kernels
    f = h5py.File(kernel_file, 'r')

    K_train = f['K_train'][:]
    K_test = f['K_test'][:]
    K_test_test = f['K_test_test'][:]

    f.close()

    return K_train, K_test, K_test_test

def compute_kernels(soaps_train, soaps_test, kernel_file=None, **kwargs):


    # Build kernel between all DEEM and all IZA
    K_train = build_kernel(soaps_train, soaps_train, **kwargs)
    K_test = build_kernel(soaps_test, soaps_train, **kwargs)
    K_test_test = build_kernel(soaps_test, soaps_test, **kwargs)

    if kernel_file is not None:

        # Save kernels for later
        g = h5py.File(kernel_file, 'w')

        g.create_dataset('K_train', data=K_train)
        g.create_dataset('K_test', data=K_test)
        g.create_dataset('K_test_test', data=K_test_test)

        for k, v in kernel_parameters.items():
            g.attrs[k] = v

        g.close()

    return K_train, K_test, K_test_test

def preprocess_kernels(K_train, K_test=[], K_test_test=[]):

    K_test_test = [center_kernel(k, K_ref=K_train) for k in K_test_test]
    K_test = [center_kernel(k, K_ref=K_train) for k in K_test]
    K_train = center_kernel(K_train)

    K_scale = np.trace(K_train) / K_train.shape[0]

    K_test_test = [k / K_scale for k in K_test_test]
    K_test = [k / K_scale for k in K_test]
    K_train /= K_scale

    return K_train, K_test, K_test_test

def do_svc(train_data, test_data, train_classes, test_classes,
           svc_type='linear', outputs=['decision_functions', 'predictions', 'scores'], **kwargs):

    if svc_type == 'kernel':
        svc = SVC(**kwargs)

    elif svc_type == 'linear':
        svc = LinearSVC(**kwargs)

    else:
        print("Error: invalid svc_type; valid choices are 'kernel' and 'linear'")
        return

    svc.fit(train_data, train_classes)

    output_list = []

    # Structure in this way to return in the same order as given in the outputs list
    for out in outputs:
        if out == 'decision_functions':
            df_train = svc.decision_function(train_data)
            df_test = svc.decision_function(test_data)
            output_list.extend((df_train, df_test))

        elif out == 'predictions':
            predicted_train_classes = svc.predict(train_data)
            predicted_test_classes = svc.predict(test_data)
            output_list.extend((predicted_train_classes, predicted_test_classes))

            print(classification_report(test_classes, predicted_test_classes))
            print(confusion_matrix(test_classes, predicted_test_classes))

        elif out == 'scores':
            train_score = svc.score(train_data, train_classes)
            test_score = svc.score(test_data, test_classes)
            output_list.extend((train_scores, test_scores))

            print(train_score)
            print(test_score)

    return output_list

def regression_check(train_data, test_data,
                     train_target, test_target,
                     regression_type='linear'):

    if regression_type == 'linear':
        regression_func = LR

    elif regression_type == 'kernel':
        regression_func = KRR

    else:
        print("Error: invalid regression type; use 'linear' or 'kernel'")
        return

    # Test KRR on decision functions
    # NOTE: KRR can't predict the test set
    # decision function very well -- why? <-- TODO: is this only for LinearSVC or also SVC?

#     regressor = KernelRidge(alpha=1.0E-12, kernel='precomputed')
#     regressor.fit(train_data, train_target)
#     predicted_train_target = regressor.predict(train_data)
#     predicted_test_target = regressor.predict(test_data)

    regressor = regression_func(regularization=1.0E-12)
    regressor.fit(train_data, train_target)
    predicted_train_target = regressor.transform(train_data)
    predicted_test_target = regressor.transform(test_data)

    print(np.mean(np.abs(predicted_train_target - train_target), axis=0))
    print(np.mean(np.abs(predicted_test_target - test_target), axis=0))

def preprocess_decision_functions(df_train, df_test):
    df_center = np.mean(df_train, axis=0)

    df_train -= df_center
    df_test -= df_center

    if df_train.ndim == 1:
        df_scale = np.linalg.norm(df_train) / np.sqrt(df_train.size)
    else:
        df_scale = np.linalg.norm(df_train, axis=0) / np.sqrt(df_train.shape[0] / df_train.shape[1])

    df_train /= df_scale
    df_test /= df_scale

    return df_train, df_test, df_center, df_scale

def postprocess_decision_functions(df_train, df_test, df_center, df_scale):

    # Rescale to raw decision function
    dfp_train = dfp_train * df_scale + df_center
    dfp_test = dfp_test * df_scale + df_center

    # Predict classes based on KPCovRized decision functions
    predicted_cantons_train = df_to_class(dfp_train, df_type, n_classes)
    predicted_cantons_test = df_to_class(dfp_test, df_type, n_classes)

def split_and_save(train_data, test_data,
                   train_idxs, test_idxs,
                   train_slice, test_slice,
                   hdf5_attrs=None,
                   output, output_format='%f'):

    # Save KPCovR class predictions
    n_samples = len(train_data) + len(test_data)
    data = np.zeros((n_samples, train_data.shape[1]))
    data[train_idxs] = train_data[train_slice]
    data[test_idxs] = test_data[test_slice]

    if hdf5_attrs is not None:
        n_digits = len(str(n_samples - 1))
        g = h5py.File(output, 'w')
        for ddx, d in enumerate(data):
            g.create_dataset(str(ddx).zfill(n_digits), data=d)

        for k, v in hdf5_attrs.items():
            g.attrs[k] = v

        g.close()
    else:
        np.savetxt(output, data, fmt=output_format)

def do_covr(train_data, test_data,
            train_targets, test_targets,
            covr_type='linear', **covr_parameters):

    if covr_type == 'linear':
        covr_func = PCovR
    elif covr_type == 'kernel':
        covr_func = KPCovR
    else:
        print("Error: invalid CovR type; use 'linear' or 'gaussian'")

    covr = covr_func(**covr_parameters)

    covr.fit(train_data, train_targets)

    T_train = covr.transform_K(train_data)
    predicted_train_target = covr.transform_Y(train_data)
    T_test = covr.transform_K(test_data)
    predicted_test_target = covr.transform_Y(test_data)

    predicted_train_target = np.squeeze(dfp_train) # TODO: move the squeezing to the KPCovR function
    predicted_test_target = np.squeeze(dfp_test)

    return T_train, T_test, predicted_train_target, predicted_test_target
