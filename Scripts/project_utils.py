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

class StandardNormScaler(BaseEstimator, TransformerMixin):
    """
        A scaler than can scale by mean and by
        columnwise or global norm or variance

        ---Attributes---
        with_mean: whether to center the data
        with_scale: whether to scale the data
        scale_type: how to normalize the data: either by the 
            trace of the (biased) covariance ('std')
            or by the L2 norm divided by the square root
            of the number of samples ('norm')
        featurewise: scale columnwise
        sample_weight_: sample weights from the fitting procedure

        ---Methods---
        fit: fit the scaler (with training data)
        transform: apply centering and scaling
        inverse_transform: undo centering and scaling
    """
    def __init__(
        self,
        with_mean=True, 
        with_scale=True, 
        scale_type='norm',
        featurewise=False, 
    ):
        self.with_mean = with_mean
        self.with_scale = with_scale
        self.scale_type = scale_type
        self.featurewise = featurewise

    def fit(self, X, y=None, sample_weight=None):
        """
            Fit the scaler

            ---Arguments---
            X: data from which we calculate the mean and/or scale
            y: ignored
            sample_weight: sample weights for weighted mean centering
        """

        if sample_weight is not None:
            sample_weight = sample_weight / np.sum(sample_weight)

        self.sample_weight_ = sample_weight

        if self.with_mean:
            self.mean_ = np.average(X, weights=self.sample_weight_, axis=0)
            Xc = X - self.mean_
        else:
            self.mean_ = None
            Xc = X

        if self.with_scale:
            if self.scale_type == 'std':
                Xm = np.average(X, weights=self.sample_weight_, axis=0)
                feature_variances = np.average(
                    (X - Xm) ** 2, weights=self.sample_weight_, axis=0
                )
                if self.featurewise:
                    self.scale_ = np.sqrt(feature_variances)
                else:
                    self.scale_ = np.sqrt(np.sum(feature_variances))
            elif self.scale_type == 'norm':
                if self.featurewise:
                    if Xc.ndim > 1:
                        n_cols = Xc.shape[1]
                    else:
                        n_cols = 1
                    self.scale_ = np.linalg.norm(Xc, axis=0) \
                        / np.sqrt(Xc.shape[0] / n_cols)
                else:
                    self.scale_ = np.linalg.norm(Xc) / np.sqrt(Xc.shape[0])
        else:
            self.scale_ = None

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

        if self.scale_ is not None:
            X = X / self.scale_

        return X

    def inverse_transform(self, X):
        """
            Undo centering and scaling

            ---Arguments---
            X: data from which to remove centering and/or scaling

            ---Returns---
            X: unscaled and uncentered X
        """
        if self.scale_ is not None:
            X = X * self.scale_

        if self.mean_ is not None:
            X = X + self.mean_

        return X

class KernelNormScaler(BaseEstimator, TransformerMixin):
    """
        Centerer and scaler for kernels

        ---Attributes---
        with_mean: whether to center the kernel
        with_norm: whether to scale the kernel
        sample_weight_: sample weights from the fitting procedure

        ---Methods---
        fit: fit the scaler (with training data)
        transform: apply centering and scaling
        inverse_transform: undo centering and scaling
    """
    def __init__(self, with_mean=True, with_norm=True):
        self.with_mean = with_mean
        self.with_norm = with_norm

    def fit(self, K, y=None, sample_weight=None):
        """
            Compute the kernel centering and scaling

            ---Arguments---
            K: kernel with which to compute the centering and/or scaling
            y: ignored
            sample_weight: sample weights for weighted mean centering
        """

        if sample_weight is not None:
            sample_weight = sample_weight / np.sum(sample_weight)

        self.sample_weight_ = sample_weight

        if self.with_mean:
            self.K_fit_rows_ = np.average(K, weights=self.sample_weight_, axis=0)
            self.K_fit_all_ = np.average(self.K_fit_rows_, weights=self.sample_weight_)
            Kc = K - self.K_fit_rows_ \
                - np.average(
                    K, weights=self.sample_weight_, axis=1
                )[:, np.newaxis] \
                + self.K_fit_all_
        else:
            self.K_fit_rows_ = None
            self.K_fit_all_ = None
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
        if self.K_fit_all_ is not None:
            Kc = K - self.K_fit_rows_ \
                - np.average(
                    K, weights=self.sample_weight_, axis=1
                )[:, np.newaxis] \
                + self.K_fit_all_

        if self.norm_ is not None:
            Kc = K / self.norm_

        return Kc

class KernelConstructor(BaseEstimator, TransformerMixin):
    """
        Class to for building kernels with the custom
        kernel functions

        ---Attributes---
        kernel_params: dictionary of parameters passed
            to the kernel functions
        kernel: 'gaussian' or 'linear' kernel
        kernel_: kernel function

        ---Methods---
        fit: Store the training data
        transform: compute the kernel

    """
    def __init__(self, kernel='linear', kernel_params={}):
        self.kernel_params = kernel_params
        self.kernel = kernel
        self.kernel_ = None

    def fit(self, X, y=None):
        """
            Store the training data on which we base the kernel

            ---Arguments---
            X: data
            y: ignored

            ---Returns---
            self
        """

        # Need to do the function selection in fit
        # otherwise a pipeline can't clone the class
        if self.kernel == 'linear':
            self.kernel_ = linear_kernel
        elif self.kernel == 'gaussian':
            self.kernel_ = gaussian_kernel

        self.X_train_ = X
        return self

    def transform(self, X):
        """
            Compute the kernel

            ---Arguments---
            X: compute the kernel between X and self.X_train_

            ---Returns---
            K: the kernel
        """
        return self.kernel_(X, self.X_train_, **self.kernel_params)

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
        self.idxs_train = idxs
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

        return self.K[idxs, :][:, self.idxs_train]     

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
            self.model_.fit(self.X[idxs], **fit_params)

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
            T = self.model_.transform(self.X[idxs])
        else:
            T = self.X[idxs]

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

class SampleWeightSelector(BaseEstimator, TransformerMixin):
    """
        Wrapper class for passing sample weights as 
        a column of X

        ---Attributes---
        model: model that will be (deep) copied and used
            to fit/transform the (weighted) data
        model_: fitted model
        weight_col: the column of y that contains the weights.

        ---Methods---
        fit: fit the model with sample weights
        transform: transform with the weighted model
        inverse_transform: inverse transform with the weighted model

    """
    def __init__(self, model=None, weight_col=None):
        self.model = model
        self.model_ = None
        self.weight_col = weight_col

    def fit(self, X, y=None, **fit_params):
        """
            Copy and fit the stored model

            ---Arguments---
            X: data used to fit the model.
                If weight_col is not None, the weight_col-th
                column of X is extracted and used as the sample weights
                for the underlying model
            y: ignored and passed to the underlying model
            fit_params: additional parameters for fitting the model

            ---Returns---
            self: fitted wrapper model
        """

        X, sample_weight = extract_weights(X, self.weight_col)

        if self.model is not None:
            self.model_ = deepcopy(self.model)
            self.model_.fit(X, y=y, sample_weight=sample_weight, **fit_params)

        return self

    def transform(self, X):
        """
            Transform according to the stored model

            ---Arguments---
            X: data to transform

            ---Returns---
            T: transformed data
        """
        X, _ = extract_weights(X, self.weight_col)

        if self.model_ is not None:
            T = self.model_.transform(X)
        else:
            T = X

        return T

    def inverse_transform(self, X):
        """
            Inverse transform according to the stored model

            ---Arguments---
            X: data to inverse transform

            ---Returns---
            iT: inverse transformed data
        """
        if self.model_ is not None:
            iT = self.model_.inverse_transform(X)
        else:
            iT = X

        return iT

def extract_weights(X, weight_col=None):
    """
        Extracts a column of weights from a matrix

        ---Arguments---
        X: matrix
        weight_col: index of the column of X
            that contains the sample weights

        ---Returns---
        X: X with sample weights extracted
        sample_weight: sample weights extracted from X
    """

    if weight_col is not None:
        if X.ndim == 1:
            raise IndexError(
                'X must be at least 2D if it contains weights'
            )
        elif weight_col > X.shape[1]:
            raise IndexError(
                'Index of column containing the weights '
                'exceeds the number of columns in X'
            )
        sample_weight = X[:, weight_col]
        sample_weight = sample_weight / np.sum(sample_weight)
        X = np.delete(X, weight_col, axis=1)
    else:
        sample_weight = None

    return X, sample_weight

def sample_weight_scorer(
    y_true, y_pred, 
    scorer=mean_absolute_error, 
    weight_col=None,
    **kwargs
):
    """
        Wrapper function to use scorers that score with
        sample weighting in sklearn pipelines and cross-validation

        ---Arguments---
        y_true: the indices on which the score will be computed
        y_pred: the predictions to score
        weight_col: the column of y_true that contains the weights.
        scorer: the scoring function to use, that has the call signature
            (y_true, y_pred, **kwargs)
        kwargs: additional keyword arguments to pass to the scorer

        ---Returns---
        score: score computed on the indices idxs according
            to the scorer
    """
    y_true, sample_weight = extract_weights(y_true, weight_col)
    sample_weight = sample_weight / np.sum(sample_weight)
    score = scorer(y_true, y_pred, sample_weight=sample_weight, **kwargs)

    return score

def balanced_class_weights(labels):
    """
        Computes balanced class weights based
        on a set of labels

        ---Arguments---
        labels: class labels

        ---Returns---
        sample_weight: balanced class weights for each
            label. Weights for class i are computed as:
            n_samples / (n_classes * n_i),
            where n_i is the population of class i
    """
    n_samples = len(labels)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_labels)

    sample_weight = np.zeros(n_samples)
    for ul, lc in zip(unique_labels, label_counts):
        label_weight = n_samples / (n_classes * lc)
        sample_weight[labels == ul] = label_weight

    sample_weight /= np.sum(sample_weight)
    return sample_weight

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
        score = scorer(y[idxs], y_pred, **kwargs)

    # Functions as a normal scorer if y is not provided
    else:
        score = scorer(idxs, y_pred, **kwargs)

    return score

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

def rank_models(models, keys=None, separator='_'):
    """
        Get a subset of models corresponding to the given keys

        ---Arguments---
        models: dictionary with format {model_key: score}
        keys: list of keys to search for in the model_key
        separator: split separator for the string of keys.
            Used to determine exact matches to keys

        ---Returns---
        candidate_models: Model tuples with the format (model_key, score)
            matching the provided keys. Candidates are sorted
            by decreasing score
    """
        
    # Create a list of the relevant models
    if keys is not None:
        candidate_models = [
            (k, v) for k, v in models.items() if all(
                [key in k.split(separator) for key in keys]
            )
        ]
    else:
        candidate_models = [
            (k, v) for k, v in models.items()
        ]

    # Sort the candidates by score
    candidate_models = sorted(
        candidate_models, key=lambda item: item[1], reverse=True
    )
    return candidate_models

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
    
