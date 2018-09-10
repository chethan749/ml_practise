import numpy as np
from sklearn import datasets
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import SVC

mnist = datasets.fetch_mldata('MNIST original')

features = np.array(mnist.data)
labels = np.array(mnist.target)

list_features = []

for feature in features:
	f = hog(feature.reshape((28, 28)), orientations = 9, pixels_per_cell = (14, 14), cells_per_block = (1, 1), visualise = False)
	list_features.append(f)
hog_features = np.array(list_features)

classifier = SVC(gamma = 0.001, probability = True)
classifier.fit(hog_features, labels)

joblib.dump(classifier, 'num_classifier.pkl', compress = 3)
