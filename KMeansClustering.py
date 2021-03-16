import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
# digits.data = dataset. This is scaling it down so that they are within the values of -1 and 1. Saves time in
# calculations = faster and fewer outliers.
data = scale(digits.data)
y = digits.target

k = 10

# OR get k dynamically
# k = len(np.unique(y))

# 1000 instances by 728 features
samples, features = data.shape  # (1000,728)

'''
Will automatically score our unsupervised learning algs
'''


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      # using euclidian distance for silhouette score
                                      metric='euclidean')))


# no need for training/testing data

# clf = classified. K clusters. Intiation centroids are random. N_Init = number of times the program runs with different
# randomly selected starting points
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

'''
Valuable Resources:
https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
'''
