from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

#prepare data
iris = datasets.load_iris()
features = pd.DataFrame(iris["data"], columns=iris["feature_names"])
target = iris["target"]

# model
model = RandomForestClassifier()
model.fit(features, target)

# save model
import pickle
pickle.dump(model, open("models/model_iris", "wb"))

