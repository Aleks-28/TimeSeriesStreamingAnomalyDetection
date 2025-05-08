import numbers

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from src.templates.model import Model
from src.utils import get_optimal_n_bins


class LODA(Model):
    """Loda: Lightweight on-line detector of anomalies. See
    :cite:`pevny2016loda` for more information.

    Two versions of LODA are supported:        
    - Static number of bins: uses a static number of bins for all random cuts.
    - Automatic number of bins: every random cut uses a number of bins deemed 
      to be optimal according to the Birge-Rozenblac method
      (:cite:`birge2006many`).

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_bins : int or string, optional (default = 10)
        The number of bins for the histogram. If set to "auto", the 
        Birge-Rozenblac method will be used to automatically determine the 
        optimal number of bins.

    n_random_cuts : int, optional (default = 100)
        The number of random cuts.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    """

    def __init__(self, n_bins: int = 'auto', n_random_cuts=100, **kwargs):
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.weights = np.ones(n_random_cuts, dtype=float) / n_random_cuts
        self.histograms_ = []

    def train(self, X_train: np.ndarray, **kwargs) -> 'LODA':
        """Fit detector. y is ignored in unsupervised methods.

        args:
            X (np.ndarray): The input samples. Shape (n_samples, n_features).

        returns:
            self (LODA): The fitted estimator.
        """
        X_train = check_array(X_train)
        pred_scores = np.zeros([X_train.shape[0], 1])
        n_components = X_train.shape[1]
        n_nonzero_components = np.sqrt(n_components)
        n_zero_components = n_components - int(n_nonzero_components)

        self.projections_ = np.random.randn(self.n_random_cuts, n_components)

        # If set to auto: determine optimal n_bins using Birge Rozenblac method
        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":

            self.limits_ = []
            self.n_bins_ = []  # only used when n_bins is determined by method "auto"

            for i in range(self.n_random_cuts):
                rands = np.random.permutation(n_components)[:n_zero_components]
                self.projections_[i, rands] = 0.
                projected_data = self.projections_[i, :].dot(X_train.T)

                n_bins = get_optimal_n_bins(projected_data)
                self.n_bins_.append(n_bins)

                histogram, limits = np.histogram(
                    projected_data, bins=n_bins, density=False)
                histogram = histogram.astype(np.float64)
                histogram += 1e-12
                histogram /= np.sum(histogram)

                self.histograms_.append(histogram)
                self.limits_.append(limits)

                # calculate the scores for the training samples
                inds = np.searchsorted(limits[:n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    histogram[inds])

        elif isinstance(self.n_bins, numbers.Integral):

            self.histograms_ = np.zeros((self.n_random_cuts, self.n_bins))
            self.limits_ = np.zeros((self.n_random_cuts, self.n_bins + 1))

            for i in range(self.n_random_cuts):
                rands = np.random.permutation(n_components)[:n_zero_components]
                self.projections_[i, rands] = 0.
                projected_data = self.projections_[i, :].dot(X_train.T)
                self.histograms_[i, :], self.limits_[i, :] = np.histogram(
                    projected_data, bins=self.n_bins, density=False)
                self.histograms_[i, :] += 1e-12
                self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

                # calculate the scores for the training samples
                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])

        else:
            raise ValueError("n_bins must be an int or \'auto\', "
                             "got: %f" % self.n_bins)

        return self

    def update(self, X: np.ndarray) -> 'LODA':
        """Updates the detector with new data.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = X.reshape(1, -1) if X.ndim == 1 else X
        X = check_array(X)
        pred_scores = np.zeros([X.shape[0], 1])

        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":

            for i in range(self.n_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)

                inds = np.searchsorted(self.limits_[i][:self.n_bins_[i] - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i][inds])

        elif isinstance(self.n_bins, numbers.Integral):

            for i in range(self.n_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)
                # update the histograms
                self.histograms_[i, :], _ = np.histogram(
                    projected_data, bins=self.n_bins, density=False)
                self.histograms_[i, :] += 1e-12
                self.histograms_[i, :] /= np.sum(self.histograms_[i, :])

                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])
        else:
            raise ValueError("n_bins must be an int or \'auto\', "
                             "got: %f" % self.n_bins)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        X = X.reshape(1, -1) if X.ndim == 1 else X
        X = check_array(X)
        pred_scores = np.zeros([X.shape[0], 1])
        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":

            for i in range(self.n_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)
                inds = np.searchsorted(self.limits_[i][:self.n_bins_[i] - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i][inds])

        elif isinstance(self.n_bins, numbers.Integral):

            for i in range(self.n_random_cuts):
                projected_data = self.projections_[i, :].dot(X.T)

                inds = np.searchsorted(self.limits_[i, :self.n_bins - 1],
                                       projected_data, side='left')
                pred_scores[:, 0] += -self.weights[i] * np.log(
                    self.histograms_[i, inds])
        else:
            raise ValueError("n_bins must be an int or \'auto\', "
                             "got: %f" % self.n_bins)

        pred_scores /= self.n_random_cuts
        return pred_scores.ravel()

    def get_params(self):
        return {'n_bins': self.n_bins, 'n_random_cuts': self.n_random_cuts}
