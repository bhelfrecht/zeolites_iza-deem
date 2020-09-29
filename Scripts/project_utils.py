#!/usr/bin/env python

import os
import sys
import numpy as np
import h5py
from sklearn.utils.multiclass import _ovr_decision_function
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append('/home/helfrech/Tools/Toolbox/utils')
from kernels import build_kernel
from kernels import center_kernel_fast, center_kernel_oos_fast
from regression import LR, KRR
from regression import PCovR, KPCovR
from tools import save_json

# TODO: move to utils/tools.py
def save_hdf5(filename, data):
    """
        Save an array or list of arrays to an HDF5 file

        ---Arguments---
        filename: name of the file in which to save the data
        data: data to save. If a list of arrays, saves each
            array as a separate dataset in a top-level group.
            Otherwise just saves the array as a dataset
    """

    f = h5py.File(filename, 'w')

    if isinstance(data, list):
        n_digits = len(str(len(data) - 1))
        for ddx, d in enumerate(data):

            # Don't need `track_order=True` because
            # we create datasets with names in alphanumeric order
            f.create_dataset(str(ddx).zfill(n_digits), data=d)
    else:
        f.create_dataset('0', data=data)

    f.close()

# TODO: rename this or make a more general HDF5 loading function
# TODO: move to utils/tools.py
def load_structures_from_hdf5(filename, datasets=None, concatenate=False):
    """
        Load structure-based data from an HDF5 file

        ---Arguments---
        filename: name of the HDF5 file to load from
        datasets: list of dataset names to load data from
        concatenate: whether to concatenate the loaded datasets
            into a single array

        ---Returns---
        structure_values: data loaded from the HDF5 file
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

def df_to_class(df, df_type, n_classes):
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
        idxs_deem_delete=[], idxs_iza_delete=[],
        train_test_concatenate=False):
    """
        Load IZA and DEEM SOAP vectors and assemble
        into arrays appropriate for the SVM-PCovR models

        ---Arguments---
        deem_file: file containing the DEEM SOAPs
        iza_file: file containing the IZA SOAPs
        idxs_deem_train: indices for the DEEM train set
        idxs_deem_test: indices for the DEEM test set
        idxs_iza_train: indices for the IZA train set
        idxs_iza_test: indices for the IZA test set
        idxs_deem_delete: indices of DEEM structures to
            omit from the returned SOAP vectors
        idxs_iza_delete: indices of IZA structures to
            omit from the returned SOAP vectors
        train_test_concatenate: whether to concatenate
            the SOAP vectors by structure and return a single
            array instead of a list of arrays where each element
            of the list contains the SOAP vectors for the environments
            in a single structure

        ---Returns---
        soaps_train: SOAP vectors in the train set
        soaps_test: SOAP vectors in the test set
    """

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
    iza_test = [soaps_iza[i] for i in idxs_iza_test]

    soaps_train = iza_train + deem_train
    soaps_test = iza_test + deem_test

    if train_test_concatenate:
        soaps_train = np.vstack(soaps_train)
        soaps_test = np.vstack(soaps_test)

    return soaps_train, soaps_test

def preprocess_soaps(soaps_train, soaps_test, return_scale=False):
    """
        Scale SOAP vectors globally by the standard deviation
        of all of the SOAP elements; useful for getting
        the SOAP vector elements of a 'usable' magnitude,
        as they are often quite small

        ---Arguments---
        soaps_train: SOAP vectors in the train set
        soaps_test: SOAP vectors in the test set
        return_scale: if True, return the scale factor
            used to scale the train and test SOAP vectors

        ---Returns---
        soaps_train: scaled SOAP vectors in the train set
        soaps_test: scaled SOAP vectors in the test set
        soaps_scale: scale factor
    """

    # Can also do other scaling/centering here --
    # this is mostly just to get the SOAPs to a 'usable' magnitude.
    # The std. dev. on the whole matrix of SOAPs has no real meaning
    # since it is computed on the flattened array
    soaps_scale = np.std(soaps_train)
    soaps_train = soaps_train / soaps_scale
    soaps_test = soaps_test / soaps_scale

    if return_scale:
        return soaps_train, soaps_test, soaps_scale
    else:
        return soaps_train, soaps_test

def load_data(deem_file, iza_file,
        idxs_deem_train, idxs_deem_test,
        idxs_iza_train, idxs_iza_test):
    """
        Load IZA and DEEM data (e.g., decision functions)
        and concatenate into test and train sets; useful
        for the SVM-PCovR models

        ---Arguments---
        deem_file: filename of file containing the DEEM data
        iza_file: filename of the file containing the IZA data
        idxs_deem_train: indices for the DEEM train set
        idxs_deem_test: indices for the DEEM test set
        idxs_iza_train: indices for the IZA train set
        idxs_iza_test: indices for the IZA test set

        ---Returns---
        train_data: data for the train set
        test_data: data for the test set
    """

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
    """
        Load train, test, and test-test kernels stored in an HDF5 file

        ---Arguments---
        kernel_file: filename of the HDF5 file containing the kernels

        ---Returns---
        K_train: kernel between the training points
        K_test: kernel between the test points and the training points
        K_test_test: kernel between the test points
    """

    f = h5py.File(kernel_file, 'r')

    K_train = f['K_train'][:]
    K_test = f['K_test'][:]
    K_test_test = f['K_test_test'][:]

    f.close()

    return K_train, K_test, K_test_test

def compute_kernels(soaps_train, soaps_test, kernel_file, **kwargs):
    """
        Compute and save train, test, and test-test kernels in an HDF5 file

        ---Arguments---
        soaps_train: train set SOAP vectors
        soaps_test: test set SOAP vectors
        kernel_file: filename of the HDF5 file in which to save the kernels
        **kwargs: keyword arguments for the kernel building function
    """

    K_train = build_kernel(soaps_train, soaps_train, **kwargs)
    K_test = build_kernel(soaps_test, soaps_train, **kwargs)
    K_test_test = build_kernel(soaps_test, soaps_test, **kwargs)

    g = h5py.File(kernel_file, 'w')

    g.create_dataset('K_train', data=K_train)
    g.create_dataset('K_test', data=K_test)
    g.create_dataset('K_test_test', data=K_test_test)

    for k, v in kwargs.items():
        g.attrs[k] = v

    g.close()

def preprocess_kernels(K_train, K_test=None, K_test_test=None, K_bridge=None):
    """
        Center and scale train, test, and test-test kernels.
        K_test and K_test_test can be provided as single matrices
        or as lists. In the latter case, each kernel in the list
        is centered relative to K_train (and K_bridge for test-test kernels)

        ---Arguments---
        K_train: kernel between the training points
        K_test: kernel between the test points and the training points
        K_test_test: kernel between the test points
        K_bridge: kernel that 'bridges' the test-test kernel
            and the train-train kernel; this is typically the
            kernel between the test points and the training points

        ---Returns---
        K_train: centered and scaled kernel between the training points
        K_test: (list of) kernel(s) between the test points and the
            training points
        K_test_test: (list of) kernel(s) between the test points
    """

    if K_test_test is not None and K_bridge is None:
        print("Error: must supply K_bridge to center OOS kernels")
        return

    if isinstance(K_test_test, list):
        K_test_test = [center_kernel_oos_fast(k, 
            K_bridge=K_bridge, K_ref=K_train) for k in K_test_test]
    elif K_test_test is not None:
        K_test_test = center_kernel_oos_fast(K_test_test, 
                K_bridge=K_bridge, K_ref=K_train)

    if isinstance(K_test, list):
        K_test = [center_kernel_fast(k, K_ref=K_train) for k in K_test]
    elif K_test is not None:
        K_test = center_kernel_fast(K_test, K_ref=K_train)

    K_train = center_kernel_fast(K_train)

    K_scale = np.trace(K_train) / K_train.shape[0]

    if isinstance(K_test_test, list):
        K_test_test = [k / K_scale for k in K_test_test]
    elif K_test_test is not None:
        K_test_test /= K_scale

    if isinstance(K_test, list):
        K_test = [k / K_scale for k in K_test]
    elif K_test is not None:
        K_test /= K_scale

    K_train /= K_scale

    output_list = [K_train]

    if K_test is not None:
        output_list.append(K_test)

    if K_test_test is not None:
        output_list.append(K_test_test)

    return output_list

def do_svc(train_data, test_data, train_classes, test_classes,
        svc_type='linear', 
        outputs=['decision_functions', 'predictions', 'weights', 'model'], **kwargs):
    """
        Wrapper function for performing KSVC/LSVC and computing
        decision functions, class predictions, and primal weights

        ---Arguments---
        train_data: training data
        test_data: test data
        train_classes: class labels for the train set
        test_classes: class labels for the test set
        svc_type: whether to run a kernel SVC ('kernel')
            or a linear SVC ('linear')
        outputs: which quantities to compute: 'decision_functions',
            'predictions', or 'weights', provided as a list.
            If 'model' is added to the output list,
            the SVC object is also returned.
            The desired quantities will be returned in the same
            order in which they are provided
        **kwargs: keyword arguments for the scikit-lear SVC

        ---Returns---
        output_list: a list of unpacked (train, test) tuples of
            the desired output quantities, returned in the same
            order as specified in the 'outputs' argument
        """

    # TODO: eliminate predictions so that they the predictions must
    # come from df_to_class and therefore be totally consistent with
    # the KPCovR-based class predictions?

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
            
            # If the test data is a list of arrays, then compute decision functions
            # for each list element. This is a bit tricky, since sklearn will
            # accept lists for fitting and transforming, so this isn't the
            # greatest practice b/c it is a bit ambiguous, but since we always
            # work with the data in numpy arrays for this project, it should work fine
            # and it is consistent with the behavior of the kernel preprocessing utility
            # defined above
            if isinstance(test_data, list) and \
                    all([isinstance(td, np.ndarray) for td in test_data]):
                df_test = [svc.decision_function(td) for td in test_data]
            else:
                df_test = svc.decision_function(test_data)

            output_list.extend((df_train, df_test))

        elif out == 'predictions':
            predicted_train_classes = svc.predict(train_data)

            # If the test data is a list of arrays, compute the predictions
            # for each list element. See note above.
            if isinstance(test_data, list) and \
                    all([isinstance(td, np.ndarray) for td in test_data]):
                predicted_test_classes = [svc.predict(td) for td in test_data]
            else:
                predicted_test_classes = svc.predict(test_data)

            output_list.extend((predicted_train_classes, predicted_test_classes))

        elif out == 'weights':
            output_list.append(svc.coef_)

        elif out == 'model':
            output_list.append(svc)

    return output_list

def regression_check(train_data, test_data,
        train_target, test_target,
        regression_type='linear'):
    """
        Wrapper function to perform LR or KRR

        ---Arguments---
        train_data: training data for the predictor variable
        test_data: test data for the predictor variable
        train_target: training data for the response variable
        test_target: test data for the response variable
        regression_type: whether to perform linear regression ('linear')
            or kernel regression ('kernel')

        ---Returns---
        predicted_train_target: predicted response variable for the train set
        predicted_test_target: predicted response variable for the test set
    """

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
    predicted_train_target = regressor.predict(train_data)
    predicted_test_target = regressor.predict(test_data)
    
    return predicted_train_target, predicted_test_target

def preprocess_data(train_data, test_data):
    """
        Center and scale data so that the column means
        are zero and the column variances are 1/n_features

        ---Arguments---
        train_data: data for the training set
        test_data: data for the test set

        ---Returns---
        train_data: centered and scaled training data
        test_data: centered and scaled test data
        train_center: column means of the train set
        train_scale: scale factor
    """

    train_center = np.mean(train_data, axis=0)

    train_data -= train_center
    test_data -= train_center

    if train_data.ndim == 1:
        train_scale = np.linalg.norm(train_data) / np.sqrt(train_data.size)
    else:
        train_scale = np.linalg.norm(train_data, axis=0) / np.sqrt(train_data.shape[0] / train_data.shape[1])

    train_data /= train_scale
    test_data /= train_scale

    return train_data, test_data, train_center, train_scale

def postprocess_decision_functions(df_train, df_test, 
        df_center, df_scale, df_type, n_classes):
    """
        Un-center and un-scale decision functions learned
        through a regressor and transform them into class predictions

        ---Arguments---
        df_train: decision functions for the train set
        df_test: decision functions for the test set
        df_center: centering parameter for the decision functions
            (column means of the train set decision functions)
        df_scale: scale parameter for the decision functions
        df_type: whether decision functions are OVO ('ovo')
            or OVR ('ovr')
        n_classes: number of true class labels

        ---Returns---
        predicted_cantons_train: predicted class labels for the train set
        precited_cantons_test: predicted class labels for the test set
    """

    # Rescale to raw decision function
    df_train = df_train * df_scale + df_center
    df_test = df_test * df_scale + df_center

    # Predict classes based on KPCovRized decision functions
    predicted_cantons_train = df_to_class(df_train, df_type, n_classes)
    predicted_cantons_test = df_to_class(df_test, df_type, n_classes)

    return predicted_cantons_train, predicted_cantons_test

def split_and_save(train_data, test_data, train_idxs, test_idxs,
        train_slice, test_slice, output, output_format='%f', hdf5_attrs=None):
    """
        Reorganize the train and test data to match the order of the
        original data; useful for e.g., taking train and test sets comprising
        both IZA and DEEM structures and extracting just the DEEM
        structures, saving them in the same order as the original DEEM data

        ---Arguments---
        train_data: data for the combined train set
        test_data: data for the combined test set
        train_idxs: training indices for the data subcategory (e.g., DEEM)
            relative to the order of the original subcategory data
        test_idxs: test indices for the data subcategory (e.g., DEEM)
            relative to the order of the original subcategory data
        train_slice: slice of the `train_data` that corresponds
            to the subcategory of interest (e.g., DEEM)
        test_slice: slice of the `test_data` that corresponds
            to the subcategory of interest (e.g., DEEM)
        output: name of file in which to store the outputs
        output_format: numpy savetxt format specifier for
            data to be saved as text
        hdf5_attrs: dictionary of attributes to save as attributes
            in the HDF5 file. If None, data will be saved as text.
            If an empty dict, data will be saved as HDF5 without attributes
        """

    # Save KPCovR class predictions
    #n_samples = len(train_data) + len(test_data)
    n_samples = len(train_idxs) + len(test_idxs)
    if train_data.ndim == 1:
        data = np.zeros(n_samples)
    else:
        data = np.zeros((n_samples, train_data.shape[1]))
    data[train_idxs] = train_data[train_slice]
    data[test_idxs] = test_data[test_slice]

    if hdf5_attrs is not None:
        n_digits = len(str(n_samples - 1))
        g = h5py.File(output, 'w')
        for ddx, d in enumerate(data):
            g.create_dataset(str(ddx).zfill(n_digits), data=d)

        for k, v in hdf5_attrs.items():
            if v is None:
                g.attrs[k] = 'None'
            else:
                g.attrs[k] = v

        g.close()
    else:
        np.savetxt(output, data, fmt=output_format)

def do_pcovr(train_data, test_data, train_targets, test_targets,
        pcovr_type='linear', compute_xr=False, **pcovr_parameters):
    """
        Wrapper function for PCovR and KPCovR

        ---Arguments---
        train_data: training data for the predictor variable
        test_data: test data for the predictor variable
        train_targers: training data for the response variable
        test_targets: test data for the response variable
        pcovr_type: whether to perform PCovR ('pcovr')
            or KPCovR ('KPCovR')
        compute_xr: whether to compute a reconstruction
            of the train_data (PCovR only)
        **pcovr_parameters: keyword arguments for the
            PCovR/KPCovR functions

        ---Returns---
        T_train: (K)PCovR latent space projection of the train set
        T_test: (K)PCovR latent space projection of the test set
        predicted_train_target: predicted response variable for the train set
        predicted_test_target: predicted response variable for the test set
        (xr_train): Reconstruction of the train predictor variable
            if compute_xr is True
        (xr_test): Reconstruction of the test predictor variable
            if compute_xr is True
    """

    if pcovr_type == 'linear':
        pcovr_func = PCovR
    elif pcovr_type == 'kernel':
        pcovr_func = KPCovR
    else:
        print("Error: invalid CovR type; use 'linear' or 'gaussian'")

    pcovr = pcovr_func(**pcovr_parameters)

    pcovr.fit(train_data, train_targets)

    T_train = pcovr.transform(train_data)
    predicted_train_target = pcovr.predict(train_data)
    T_test = pcovr.transform(test_data)
    predicted_test_target = pcovr.predict(test_data)

    # TODO: move the squeezing to the KPCovR function?
    predicted_train_target = np.squeeze(predicted_train_target)
    predicted_test_target = np.squeeze(predicted_test_target)

    if compute_xr and pcovr_type == 'linear':
        xr_train = pcovr.inverse_transform(train_data)
        xr_test = pcovr.inverse_transform(test_data)
        return T_train, T_test, predicted_train_target, predicted_test_target, xr_train, xr_test
    else:
        return T_train, T_test, predicted_train_target, predicted_test_target

def generate_reports(cantons_train, cantons_test,
        predicted_cantons_train, predicted_cantons_test,
        class_names=None):
    """
        Generate classification reports and confusion matrices

        ---Arguments---
        cantons_train: true class labels for the train set
        cantons_test: true class labels for the test set
        predicted_cantons_train: predicted class labels for the train set
        predicted_cantons_test: predicted class labels for the test set
        class_names: alternative names for the class labels, in order

        ---Returns---
        train_report: classification report for the train set
        test_report: classification report for the test set
        train_matrix: confusion matrix for the train set
        test_matrix: confusion matrix for the test set
    """

    train_report = classification_report(cantons_train, predicted_cantons_train, 
            output_dict=True, target_names=class_names, zero_division=0) 
    test_report = classification_report(cantons_test, predicted_cantons_test,
            output_dict=True, target_names=class_names, zero_division=0)
                            
    train_matrix = confusion_matrix(cantons_train, predicted_cantons_train)
    test_matrix = confusion_matrix(cantons_test, predicted_cantons_test)

    return train_report, test_report, train_matrix, test_matrix
                                                    
def save_reports(train_report, test_report, train_matrix, test_matrix, 
        train_report_file, test_report_file,
        train_matrix_file, test_matrix_file):
    """
        Save classification reports and confusion matrices

        ---Arguments---
        train_report: classification report for the train set
        test_report: classificaion report for the test set
        train_matrix: confusion matrix for the train set
        test_matrix: confusion matrix for the test set
        train_report_file: filename of the file in which
            to save the classification report for the train set
        test_report_file: filename of the file in which
            to save the classfication report for the test set
        train_matrix_file: filename of the file in which
            to save the confusion matrix for the train set
        test_matrix_file: filename of the file in which
            to save the confusion matrix for the test set
    """

    save_json(train_report, train_report_file)
    save_json(test_report, test_report_file)
                                                                                        
    np.savetxt(train_matrix_file, train_matrix)
    np.savetxt(test_matrix_file, test_matrix)

def print_report(report, n_digits=2):
    """
        Print a classificaion report

        ---Arguments---
        report: classification report to print
            (as a dictionary)
        n_digits: number of digits to round printed values to
    """

    headers = ['precision', 'recall', 'f1-score', 'support']
    
    # Determine required widths for headers, class labels, and data
    max_header_width = np.amax([len(header) for header in headers])
    max_data_width = n_digits + 2
    max_width = np.maximum(max_header_width, max_data_width)
    max_label_width = np.amax([len(label) for label in report.keys()])
    
    # Set string formatting for the headers and data
    header_format = f'{{:>{max_label_width}s}} '
    header_format += f' {{:>{max_width}s}}' * len(headers)

    data_format = f'{{:>{max_label_width}s}} '
    data_format += f' {{:>{max_width}.{n_digits}f}}' * (len(headers) - 1)
    data_format += f' {{:>{max_width}d}}'

    # Formatting of the printed table of results
    report_string = header_format.format('', *headers)
    hline = '-' * len(report_string) + '\n'
    report_string = hline + report_string

    # Accuracy is to be displayed separately,
    # as it is a single value
    accuracy = report.pop('accuracy')

    # Print precision, recall, f1 score, and number of samples
    for row_label, data in report.items():
        report_string += '\n' + data_format.format(row_label, data['precision'],
                data['recall'], data['f1-score'], data['support'])

    # Print accuracy
    report_string += '\n' + hline
    report_string += f'\n{{:>{max_label_width}s}} '.format('accuracy')
    report_string += f' {{:>{max_width}.{n_digits}f}}'.format(accuracy)
    report_string += '\n' + hline

    print(report_string)
    
