
import numpy as np
from .utils import compute_similarities, compute_num_n, compute_nwknn_weight, \
    predict_class, predict_class_iknn, predict_class_nwknn


class KNN:
    METRICS = ['manhattan', 'euclidean', 'cosine', 'bm25']
    

    def __init__(self, n_neighbors=5, metric='euclidean', \
        metric_params={'k1':1.2, 'b':.75}):
        self._n_neighbors = n_neighbors
        
        assert metric in KNN.METRICS, f'your metric value must be on of {KNN.METRICS}'

        self._metric = metric
        
        if metric_params != None:
            assert isinstance(metric_params, dict), 'metric_params should be dict'
        self._metric_params = metric_params

    def fit(self, X, y):

        if isinstance(X, list):
            X = np.array(X)

        if isinstance(y, list):
            y = np.array(y)

        assert (isinstance(X, np.ndarray)) & (isinstance(y, np.ndarray)), \
            'You should pass X and y as numpy array'
        
        # Reshape class array into 1 dimension
        if y.ndim > 1:
            y = y.reshape(-1,)
    
        self._train_features = X
        self._train_label = y
        self.classes_ = np.unique(y)
        
    def predict(self, X):

        if isinstance(X, list):
            X = np.array(X)

        assert isinstance(X, np.ndarray), 'You should pass X as numpy array'
        
        if self._metric == 'bm25':
            assert ('int' not in str(X.dtype)) & ('float' not in str(X.dtype)), \
                'You should pass numpy array of string (corpus)'
        else:
            assert ('int' in str(X.dtype)) | ('float' in str(X.dtype)), \
                'You should pass numpy array of int dtype'

        similarity_matrix = compute_similarities(X, \
            self._train_features, self._metric, self._metric_params)

        predicted_class = []
        for i in range(X.shape[0]):
            class_ = predict_class(similarity_matrix[i], self._train_label, \
                self._n_neighbors)
            predicted_class.append(class_)
        
        return np.array(predicted_class)


class ImprovedKNN(KNN):

    def __init__(self, n_neighbors=5, metric='euclidean', \
        metric_params={'k1':1.2, 'b':.75}):
        
        super(ImprovedKNN, self).__init__(n_neighbors, metric, metric_params)

    def fit(self, X, y):

        super().fit(X, y)
        self.classes_, counts = np.unique(y, return_counts=True)
        class_freq = {label: freq for label, freq in zip(self.classes_, counts)}
        self._num_n = compute_num_n(class_freq, self._n_neighbors)

    def predict(self, X):

        if isinstance(X, list):
            X = np.array(X)

        assert isinstance(X, np.ndarray), 'You should pass X as numpy array'
        
        if self._metric == 'bm25':
            assert ('int' not in str(X.dtype)) & ('float' not in str(X.dtype)), \
                'You should pass numpy array of string (corpus)'
        else:
            assert ('int' in str(X.dtype)) | ('float' in str(X.dtype)), \
                'You should pass numpy array of int dtype'

        similarity_matrix = compute_similarities(X, \
            self._train_features, self._metric, self._metric_params)

        predicted_class = []
        for i in range(X.shape[0]):
            class_ = predict_class_iknn(similarity_matrix[i], self._train_label, \
                self._num_n)
            predicted_class.append(class_)
        
        return np.array(predicted_class)
        
 
class NWKNN(KNN):

    def __init__(self, n_neighbors=5, metric='euclidean', \
        metric_params={'k1':1.2, 'b':.75}, exponent=2):
        
        super(NWKNN, self).__init__(n_neighbors, metric, metric_params)
        self._exponent = exponent

    def fit(self, X, y):

        super().fit(X, y)
        self.classes_, counts = np.unique(y, return_counts=True)
        class_freq = {label: freq for label, freq in zip(self.classes_, counts)}
        self._class_weight = compute_nwknn_weight(class_freq, self._exponent)

    def predict(self, X):

        if isinstance(X, list):
            X = np.array(X)

        assert isinstance(X, np.ndarray), 'You should pass X as numpy array'
        
        if self._metric == 'bm25':
            assert ('int' not in str(X.dtype)) & ('float' not in str(X.dtype)), \
                'You should pass numpy array of string (corpus)'
        else:
            assert ('int' in str(X.dtype)) | ('float' in str(X.dtype)), \
                'You should pass numpy array of int dtype'

        similarity_matrix = compute_similarities(X, \
            self._train_features, self._metric, self._metric_params)

        predicted_class = []
        for i in range(X.shape[0]):
            class_ = predict_class_nwknn(similarity_matrix[i], self._train_label, \
                self._n_neighbors, self._class_weight)
            predicted_class.append(class_)
        
        return np.array(predicted_class)
        
        