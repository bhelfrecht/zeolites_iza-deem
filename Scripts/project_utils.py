#!/usr/bin/env python

import os
import sys
import numpy as np
import h5py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.multiclass import _ovr_decision_function
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append('/home/helfrech/Tools/Toolbox/utils')
from kernels import build_kernel
from kernels import center_kernel_fast, center_kernel_oos_fast
from kernels import gaussian_kernel, linear_kernel
from kernels import sqeuclidean_distances
from regression import LR, KRR
from regression import PCovR, KPCovR
from tools import save_json
from copy import deepcopy

def get_basename(path):
    """
        Shorthand for getting the file prefix
        from a given file path

        ---Arguments---
        path: path to the file

        ---Returns---
        file_prefix: filename without base
            directory path or extension
    """
    return os.path.splitext(os.path.basename(path))[0]

def removesuffix(string, suffix):
    """
        Remove the suffix from a string.
        Built-in string method in Python 3.9;
        we reproduce it here
    
        ---Arguments---
        string: string from which to remove the suffix
        suffix: substring to remove

        ---Returns---
        string: string with suffix removed
    """
    if suffix in string and len(suffix) > 0:
        string = string[0:-len(suffix)]
    return string

def removeprefix(string, prefix):
    """
        Remove the prefix from a string.
        Built-in string method in Python 3.9;
        we reproduce it here

        ---Arguments---
        string: string from which to remove the prefix
        prefix: substring to remove

        ---Returns---
        string: string with prefix removed
    """
    if prefix in string and len(prefix) > 0:
        string = string[len(prefix):]
    return string

class NormScaler(BaseEstimator, TransformerMixin):
    """
        A scaler than can scale by mean and by
        columnwise or global norm

        ---Attributes---
        with_mean: whether to center the data
        with_norm: whether to scale the data by a norm
        featurewise: scale by columnwise norm

        ---Methods---
        fit: fit the scaler (with training data)
        transform: apply centering and scaling
        inverse_transform: undo centering and scaling
    """
    def __init__(self, with_mean=True, with_norm=True, featurewise=False):
        self.with_mean = with_mean
        self.with_norm = with_norm
        self.featurewise = featurewise

    def fit(self, X, y=None):
        """
            Fit the scaler

            ---Arguments---
            X: data from which we calculate the mean and/or norm
        """

        if self.featurewise:
            axis = 0
            n_cols = X.shape[1]
        else:
            axis = None
            n_cols = 1

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
            Xc = X - self.mean_
        else:
            self.mean_ = None
            Xc = X

        if self.with_norm:
            self.norm_ = np.linalg.norm(Xc, axis=axis) / np.sqrt(len(Xc) / n_cols)
        else:
            self.norm_ = None

        return self

    def transform(self, X):
        """
            Apply centering and/or scaling

            ---Arguments---
            X: data to center and/or scale
        
            ---Returns---:
            X: centered and/or scaled X
        """
        if self.mean_ is not None:
            X = X - self.mean_

        if self.norm_ is not None:
            X = X / self.norm_

        return X

    def inverse_transform(self, X):
        """
            Undo centering and scaling

            ---Arguments---
            X: data from which to remove centering and/or scaling

            ---Returns---
            X: unscaled and uncentered X
        """
        if self.norm_ is not None:
            X = X * self.norm_

        if self.mean_ is not None:
            X = X + self.mean_

        return X

class KernelNormScaler(BaseEstimator, TransformerMixin):
    """
        Centerer and scaler for kernels

        ---Attributes---
        with_mean: whether to center the kernel
        with_norm: whether to scale the kernel

        ---Methods---
        fit: fit the scaler (with training data)
        transform: apply centering and scaling
        inverse_transform: undo centering and scaling
    """
    def __init__(self, with_mean=True, with_norm=True):
        self.with_mean = with_mean
        self.with_norm = with_norm

    def fit(self, K, y=None):
        """
            Compute the kernel centering and scaling

            ---Arguments---
            K: kernel with which to compute the centering and/or scaling
        """
        if self.with_mean:
            self.centerer_ = KernelCenterer().fit(K)
            Kc = self.centerer.transform(K)
        else:
            self.centerer_ = None
            Kc = K

        if self.with_norm:
            self.norm_ = np.trace(Kc) / Kc.shape[0]
        else:
            self.norm_ = None

        return self

    def transform(self, K):
        """
            Apply centering and scaling to a kernel

            ---Arguments---
            K: kernel on which to apply the centering and/or scaling

            ---Returns---
            K: centered and/or scaled kernel
        """
        if self.centerer_ is not None:
            K = self.centerer_.transform(K)

        if self.norm_ is not None:
            K = K / self.norm_

        return K

    def inverse_transform(self, K):
        """
            Undo centering and scaling of a kernel

            ---Arguments---
            K: kernel on which to undo the centering and/or scaling

            ---Returns---
            K: uncentered and unscaled kernel
        """
        if self.norm_ is not None:
            K = K * self.norm_

        if self.centerer_ is not None:
            K_fit_rows = self.centerer_.K_fit_rows_
            K_fit_all = self.centerer_.K_fit_all_
            K_pred_cols = np.mean(K, axis=1)
            K = K - K_fit_all + K_pred_cols + K_fit_rows

        return K

# TODO: linear kernel option as well?
#class KernelConstructor(BaseEstimator, TransformerMixin):
#    def __init__(self, gamma=1.0):
#        self.gamma = gamma
#
#    def fit(self, X, y=None):
#        self.X_train = X
#        return self
#
#    def transform(self, X):
#        return gaussian_kernel(X, self.X_train, gamma=self.gamma)

class KernelLoader(BaseEstimator, TransformerMixin):
    """
        Class to load kernels from a file for use in an sklearn pipeline

        ---Attributes---
        filename: filename from which the kernel will be loaded
        load_args: keyword arguments passed to the function
            for loading the kernel (currently assumes load_hdf5)
        idxs_train: train indices to extract from the kernel
        K: the kernel matrix

        ---Methods---
        fit: load the kernel and set the training indices
        transform: slice the kernel and return the corresponding
            values as a (subkernel) matrix
    """
    def __init__(self, filename=None, **load_args):
                
        # Filename is actually required, but give it
        # a default of None to be compatible with sklearn
        self.filename = filename
        self.load_args = load_args

    def fit(self, idxs, y=None):
        """
            Load the kernel and set the train indices

            ---Arguments---
            idxs: train indices (column indices to select)
            y: ignored; for consistency with other sklearn fit methods

            ---Returns---
            self: fitted kernel loader
        """
        self.K = load_hdf5(self.filename, **self.load_args)
        self.idxs_train = idxs.flatten()
        return self

    def transform(self, idxs):
        """
            Slice the kernel and return the values

            ---Arguments---
            idxs: row indices to select

            ---Returns---
            K: sliced kernel (the subkernel defined by
                the indices (idxs, self.idxs_train),
                where self.idxs_train are those indices
                passed during the fit
        """

        return self.K[idxs.flatten(), :][:, self.idxs_train]     

#class MalleableGaussianKernel(object):
#    def __init__(self, delta=1.0E-12, max_terms=15):
#        self.delta = delta
#        self.max_terms = max_terms
#        self.n_factorial = np.insert(np.cumprod(np.arange(1, max_terms)), 0, 1)
#
#    def _fit(self, XA, XB):
#        D = sqeuclidean_distances(XA, XB)
#        powers = np.zeros((len(XA), len(XB), self.max_terms))
#        for n in range(0, self.max_terms):
#            powers[:, :, n] = D ** n
#
#        return powers
#
#    def fit(self, XA, XB):
#        self.powers = np.zeros((len(XA), len(XB), self.max_terms))
#        if isinstance(XA, list) and isinstance(XB, list):
#            for adx, a in enumerate(XA):
#                for bdx, b in enumerate(XB):
#                    self.powers[adx, bdx, :] = np.mean(
#                        self._fit(a, b), axis=(0, 1)
#                    )
#
#        elif isinstance(XA, list):
#            for adx, a in enumerate(XA):
#                self.powers[adx, :, :] = np.mean(
#                    self._fit(a, XB), axis=0
#                )
#
#        elif isinstance(XB, list):
#            for bdx, b in enumerate(XB):
#                self.powers[:, bdx, :] = np.mean(
#                    self._fit(XA, b), axis=1
#                )
#
#        else:
#            self.powers = self._fit(XA, XB)
#
#        return self
#
#    def transform(self, gamma=1.0):
#        K = np.zeros(self.powers.shape[0:-1])
#        for n in range(0, self.max_terms):
#            k = self.powers[:, :, n] * (-gamma) ** n / self.n_factorial[n]
#            K += k
#            # TODO: how to handle breaking and max_terms?
#            #if np.linalg.norm(k) / np.sqrt(k.shape[0]) <= delta:
#            #    break
#
#        print('Warning: reached maximum number of expansion terms')
#        return K

class SampleSelector(BaseEstimator, TransformerMixin):
    """
        Wrapper class for selecting samples by index
        and passing them to another transformer

        ---Attributes---
        model: model that will be (deep) copied and used
            to fit/transform the selected samples, if given
        model_: fitted model
        X: the dataset from which samples will be selected

        ---Methods---
        fit: copy the model and fit with samples from self.X
        transform: transform samples from self.X
        inverse_transform: inverse_transform samples
    """

    def __init__(self, X=None, model=None):

        # X is actually required to get things to work,
        # but has to be a keyword argument for sklearn.
        # model is optional, in which case transforms
        # will just return the selected samples from X
        self.X = X
        self.model = model
        self.model_ = None

    def fit(self, idxs, **fit_params):
        """
            Copy and fit the stored model

            ---Arguments---
            idxs: sample indices to use to fit the model
            fit_params: additional parameters to pass to the model fit

            ---Returns---
            self: selector with fitted model, if provided
                upon initialization
        """

        if self.model is not None:
            self.model_ = deepcopy(self.model)
            self.model_.fit(self.X[idxs.flatten()], **fit_params)

        return self

    def transform(self, idxs):
        """
            Transform according to the stored model

            ---Arguments---
            idxs: sample indices to transform

            ---Returns---
            T: if a model is given in initialization, the transformed
                data at the input indices (idxs). If a model
                is not given at initialization, the (untransformed)
                data at the input indices is returned
        """

        if self.model_ is not None:
            T = self.model_.transform(self.X[idxs.flatten()])
        else:
            T = self.X[idxs.flatten()]

        return T

    def inverse_transform(self, X):
        """
            Perform an inverse transform according to the stored model

            ---Arguments---
            X: data on which to perform the inverse transform

            ---Returns---
            iT: if a model is given at initialization,
                data that has been inverse-transformed. If a model
                is not given at initialization, the (uninversetransformed)
                data is returned
        """
        if self.model_ is not None:
            iT = self.model_.inverse_transform(X)
        else:
            iT = X

        return iT

def score_by_index(idxs, y_pred, y=None, scorer=mean_absolute_error, **kwargs):
    """
        Wrapper function to use scorers that select by index
        in sklearn pipelines and cross-validation

        ---Arguments---
        idxs: the indices on which the score will be computed
        y_pred: the predictions to score
        y: the reference data; while a keyword argument,
            this argument is required for the scoring to work
            properly. It is formally a keyword argument
            for compatibility with the sklearn scorer infrastructure
            (e.g., for cross-validation)
        scorer: the scoring function to use, that has the call signature
            (y_true, y_pred, **kwargs)
        kwargs: additional keyword arguments to pass to the scorer

        ---Returns---
        score: score computed on the indices idxs according
            to the scorer
    """

    # In a 'select-by-index' pipeline, y_pred will already
    # be the correct properties (instead of a set of indices),
    # so we only need to extract the reference data y
    # at the appropriate indices
    if y is not None:
        score = scorer(y[idxs.flatten()], y_pred, **kwargs)

    # Functions as a normal scorer if y is not provided
    else:
        score = scorer(idxs, y_pred, **kwargs)

    return score

def cv_generator(cv_idxs):
    """
        A generator yielding (train, test) tuples
        from a precomputed set of folds, in the shape
        (n_samples_in_fold, n_folds)

        ---Arguments---
        cv_idxs: a 2D matrix of folds, with shape
            (n_samples_in_fold, n_folds)

        ---Yields---
        train_idxs: indices for training, from k-1 folds
        test_idxs: indices for validation, a single fold
    """

    k = cv_idxs.shape[1]
    for kdx in range(0, k):
        k_list = list(range(0, k))
        k_list.pop(kdx)
        test_idxs = cv_idxs[:, kdx]
        train_idxs = np.concatenate(cv_idxs[:, k_list])
        yield train_idxs, test_idxs

def get_optimal_parameters(cv_results, scoring, **base_parameters):
    """
        Extract optimal hyperparameters from an sklearn CV search

        ---Arguments---
        cv_results: the sklearn CV results
        scoring: name of the scoring entry in the CV results
        base_parameters: dictionary of base (i.e., unvarying)
            parameters for the model of interest.
            The optimal hyperparameters will be 
            added to this dictionary.

        ---Returns---
        opt_parameters: hyperparameters for the model
            with the lowest error
    """

    idx = np.argmin(cv_results[f'rank_test_{scoring}'])
    opt_parameters = base_parameters.copy()

    # Find the parameter name; ignore the 'double underscore'
    # notation telling us to which model the parameter belongs to,
    # as this will be done manually
    for key, value in cv_results['params'][idx].items():
        split_key = key.split('__')[-1]
        opt_parameters[split_key] = value

    return opt_parameters

# TODO: move to utils/tools.py
def save_hdf5(filename, data, chunks=None, attrs={}):
    """
        Save an array or list of arrays to an HDF5 file

        ---Arguments---
        filename: name of the file in which to save the data
        data: data to save. If a list of arrays, saves each
            array as a separate dataset in a top-level group.
            Otherwise just saves the array as a dataset
        attrs: dictionary of attributes to add to the HDF5 file
    """

    f = h5py.File(filename, 'w')

    if isinstance(data, list):
        n_digits = len(str(len(data) - 1))
        for ddx, d in enumerate(data):

            # Don't need `track_order=True` because
            # we create datasets with names in alphanumeric order
            f.create_dataset(str(ddx).zfill(n_digits), data=d)
    else:
        f.create_dataset('0', data=data, chunks=chunks)

    # Add attributes
    for k, v in attrs.items():
        if v is None:
            f.attrs[k] = 'None'
        else:
            f.attrs[k] = v

    f.close()

# TODO: move to utils/tools.py
def load_hdf5(filename, datasets=None, indices=None, concatenate=False):
    """
        Load data from an HDF5 file

        ---Arguments---
        filename: name of the HDF5 file to load from
        datasets: list of dataset names to load data from.
            If None, loads all datasets.
        indices: list of (arrays of) indices to load from each dataset.
            If None, loads all data from the selected datasets.
            Can be provided as a tuple for multidimensional
            numpy indexing.
        concatenate: whether to concatenate the loaded datasets
            into a single array. If only one dataset is present,
            use `concatenate=True` to return the array
            instead of a one-element list
        ---Returns---
        dataset_values: data loaded from the HDF5 file
    """

    dataset_values = []

    f = h5py.File(filename, 'r')

    if datasets is None:

        # f.keys() returns a view, so to get the actual
        # names we need list comprehension
        datasets = [key for key in f.keys()]

    if indices is None:
        indices = [slice(None)] * len(datasets)
    elif isinstance(indices, np.ndarray):
        indices = [indices]

    for dataset_name, idxs in zip(datasets, indices):
        dataset_values.append(f[dataset_name][idxs])

    f.close()

    if concatenate:
        dataset_values = np.vstack(dataset_values)
    elif len(dataset_values) == 1:
        dataset_values = dataset_values[0]

    return dataset_values

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

##### DELETE BEGIN #####
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
    soaps_deem = load_hdf5(deem_file, datasets=None, concatenate=False)
    for i in sorted(idxs_deem_delete, reverse=True):
        soaps_deem.pop(i)

    soaps_iza = load_hdf5(iza_file, datasets=None, concatenate=False)
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
        outputs=['decision_functions', 'predictions', 'weights'], 
        save_model=None, **kwargs):
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
            The desired quantities will be returned in the same
            order in which they are provided
        save_model: save the SVC model in JSON at the given location
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

    if save_model is not None:
        save_json(svc.__dict__, save_model, array_convert=True)

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

def preprocess_data(train_data, test_data, scale='feature'):
    """
        Center and scale data so that the column means
        are zero and the column variances are 1/n_features

        ---Arguments---
        train_data: data for the training set
        test_data: data for the test set
        scale: type of scaling: by feature ('feature')
            or global scaling ('global')

        ---Returns---
        train_data: centered and scaled training data
        test_data: centered and scaled test data
        train_center: column means of the train set
        train_scale: scale factor
    """

    # TODO: be careful here since data is modified in place
    train_center = np.mean(train_data, axis=0)

    train_data -= train_center
    test_data -= train_center

    # TODO: clean this up so error is raised if scaling keyword isn't recognized
    # TODO: could we also just use the global scaling for the SVM SOAPs for consistency?
    if train_data.ndim == 1:
        train_scale = np.linalg.norm(train_data) / np.sqrt(train_data.size)
    elif scale == 'global':
        train_scale = np.linalg.norm(train_data) / np.sqrt(train_data.shape[0])
    elif scale == 'feature':
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
        pcovr_type='linear', compute_xr=False, save_model=None, **pcovr_parameters):
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
        save_model: save the PCovR model in JSON at the specified location
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

    outputs = [T_train, T_test, predicted_train_target, predicted_test_target]

    if compute_xr and pcovr_type == 'linear':
        xr_train = pcovr.inverse_transform(train_data)
        xr_test = pcovr.inverse_transform(test_data)
        outputs.extend([xr_train, xr_test])

    if save_model is not None:
        save_json(pcovr.__dict__, save_model, array_convert=True)
    
    return outputs
##### DELETE END #####

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
    
