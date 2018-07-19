import pandas as pd
import numpy as np
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler


cancer_data = pd.read_csv("../data/wdbc.csv", header = None)
##print(cancer_data.columns)

cancer_data = cancer_data.rename(index=str, columns={0: "id", 1: "target", 2: "mean radius", 3: "mean texture", 4: "mean perimeter", 5: "mean area",
                                  6: "smoothness", 7: "mean compactness", 8: "mean concavity", 9: "mean concave points", 
                                  10: "mean symmetry", 11: "mean fractal dimension", 12: "radius error", 13: "texture error", 
                                  14: "perimeter error", 15: "area error", 16: "smoothness error", 17: "compactness error", 
                                  18: "concavity error", 19: "concave points error", 20: "symmetry error", 21: "fractal dimension error", 
                                  22: "worst radius", 23: "worst texture", 24: "worst perimeter", 25: "worst area", 26: "worst smoothness", 
                                  27: "worst compactness", 28: "worst concavity", 29: "worst concave points", 30: "worst symmetry", 
                                  31: "worst fractal dimension"})
##print(cancer_data.columns)
##print(cancer_data.head())


X = np.array(cancer_data.iloc[:,2:32])
y = np.array(cancer_data.iloc[:,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1992)

print(X_train.shape)
print(X_test.shape)


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


parameters = {'C':[10**-4, 10**-3, 10**-2, 0.1, 1, 10, 10**2, 10**3, 10**4]}

svc = svm.LinearSVC()

cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
clf = GridSearchCV(svc, parameters, cv=cv)
clf.fit(X_train_scaled, y_train)
print("Optimal C for LinearSVM: {}".format(clf.best_params_.get('C')))
print("Maximum accuracy score of {:,.4f} reached with C = {}".
      format(clf.best_score_, clf.best_params_.get('C')))


svc = svm.LinearSVC(C=10, random_state=0)
svc.fit(X_train_scaled, y_train)


print("Test accuracy: {:.2f}".format(
    svc.score(X_test_scaled, y_test)))
