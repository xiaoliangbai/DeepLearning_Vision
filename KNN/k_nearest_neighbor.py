import numpy as np

class KNearestNeighbor(object):
  """ 
  A kNN classifier with L2 distance
  Using vanilla Numpy array on CPU or
  Tensorflow on GPU
  """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    - model: 'numpy' or 'tensorflow'
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, model='numpy'):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if model == 'numpy':
      dists = self.compute_distances_numpy(X)
    elif model == 'tensorflow':
      dists = self.computer_distances_tensorflow(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

 
  def compute_distances_numpy(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using numpy matrix operations

    Input:   X - num_test flattened images with shape (N, D)
    Output:  dists - Numpy array of shape (num_test, num_train) where dists[i, j] is 
             the Euclidean distance between the ith test point and the jth training point
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    test2 = np.square(X)
    tr2 = np.square(self.X_train)
    dists = np.sum(test2, axis=1, keepdims=True) + np.sum(tr2, axis=1)
    dists = dists - 2* X.dot(self.X_train.T)

    return dists

  def compute_distances_tf(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using numpy matrix operations

    Input:   X - num_test flattened images with shape (N, D)
    Output:  dists - Numpy array of shape (num_test, num_train) where dists[i, j] is 
             the Euclidean distance between the ith test point and the jth training point
    """
    import tensorflow as tf
    
    num_test = X.shape[0]
    dim = X.shape[1]
    num_train = self.X_train.shape[0]
    #with tf.device('/gpu:0'):
    if True:
        test_data = tf.placeholder('int32', shape=(None, dim))
        train_data = tf.placeholder('int32', shape=(None, dim))
        test2 = tf.square(test_data)
        tr2 = tf.square(train_data)
        test2_sum = tf.reduce_sum(test2, reduction_indices=1, keep_dims=True)
        tr2_sum = tf.reduce_sum(tr2, reduction_indices=1)
        L2_dists_tmp1 = tf.add(test2_sum, tr2_sum)
        L2_dists = tf.subtract(L2_dists_tmp1, tf.mul(2, tf.matmul(test_data, train_data, transpose_b=True)))
	
        model = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config) as session:
	    dists = session.run(L2_dists, feed_dict={test_data: X, train_data: self.X_train}) 

    return dists


  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      closest_y = self.y_train[dists[i].argsort()[:k]]
    
      values, counts = np.unique(closest_y, return_counts=True)
      y_pred[i] = values[np.argmax(counts)]

    return y_pred

