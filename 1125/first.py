#import matplotlib.pyplot as plt
#import seaborn as sns

import pandas as pd
import sklearn
from sklearn.datasets import load_iris

# + データの読み込み
iris = load_iris()

iris_features = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_features.head()