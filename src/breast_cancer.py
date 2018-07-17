import pandas as pd
import numpy as np
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit


cancer = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header = None)
cancer.to_csv("../data/bc.csv", encoding='utf-8', index=False, header=False)
