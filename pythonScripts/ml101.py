import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(iris.data.shape)
print(iris.target.shape)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head(10))
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
#             verbose=0, warm_start=False)
clf = RandomForestClassifier(n_estimators=20)
clf.fit(X_train, y_train)
value = clf.predict(X_test)
print(clf.predict_proba(X_test)[0:10])
print(accuracy_score(y_test, value))

scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# regression
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target
X_train, X_test, y_train, y_test = train_test_split(features, targets, train_size=0.8, random_state=42)
n_trees = [1,10,50,100,300,500]
scores=[]
for treeSize in n_trees:
    rf = RandomForestRegressor(n_estimators=treeSize, oob_score=True, random_state=0)
    rf.fit(X_train, y_train)
    predicted_test = rf.predict(X_test)
    test_score = mean_squared_error(y_test, predicted_test)
    print(test_score)
    scores.append(test_score)
plt.scatter(n_trees, scores, c='b')
plt.xlabel('number of trees')
plt.ylabel('MSE error')
plt.show()



