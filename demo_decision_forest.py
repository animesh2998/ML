import numpy as np
from algos import DecisionTreeClassifier, RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
print('DecisionTree accuracy:', accuracy_score(y_test, dt.predict(X_test)))

rf = RandomForestClassifier(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)
print('RandomForest accuracy:', accuracy_score(y_test, rf.predict(X_test)))
