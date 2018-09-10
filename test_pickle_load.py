import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = None
with open('test.pkl', 'rb') as input:
	clf = pickle.load(input)

print(clf.feature_importances_)
print(clf.predict([[0, 0, 0, 0]]))
