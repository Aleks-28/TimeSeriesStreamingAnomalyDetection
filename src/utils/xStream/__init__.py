
from .import xStream_swig

import numpy as np

class xStream:
	"""xStream streaming outlier detection."""
	
	def __init__(self, k=100, c=100, d=15, nwindows=1, init_sample=256, cosine=False, seed=1):
		"""
		Parameters
		-----------
		k: int (default=100)
			Projection size.
			
		c: int (default=100)
			Number of chains.
			
		d: int (default=15)
			Depth.
			
		nwindows: int (default=1)
			> 0 if windowed.
			
		init_samples: int (default=256)
			Initial sample size.
			
		cosine: bool (default=False)
			Work in cosine space instead of Euclidean.
		"""
		
		self._xStream = xStream_swig.xStream(k, c, d, nwindows, init_sample, cosine, seed)
		
	def fit_predict(self, X, features=None):
		"""
		Process next chunk of data.
		
		Parameters
		---------------
		X: ndarray, shape (n_samples, n_features)
			The input data.
			
		features: list of strings, optional
			Feature names for the columns of X.
			
		Returns
		---------------
		y: ndarray, shape (n_samples,)
			Outlier scores for passed input data.
		"""
		scores = np.zeros(X.shape[0], dtype=np.float64)
		if features is None:
			features = [ b'i' + str(i).encode() for i in range(X.shape[1]) ]
		else:
			features = [ b's' + f.encode() for f in features ]

		self._xStream.fit_predict(features, np.array(X, dtype=np.float64), scores)

		# xStream provides an _inlier_ score. Apply a monotonically decreasing function
		# which is finite on [0,inf] to turn it into an outlier score
		return 1/(1+scores)
