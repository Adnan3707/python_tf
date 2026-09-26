import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('./housing.csv')
# One Hot Encoding
encoded = pd.get_dummies(data)

y = encoded.iloc[:,8]
print(y.head())

x = encoded.drop(encoded.columns[8], axis=1)
print(x.head())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,FunctionTransformer
from copy import deepcopy

std_scaler = StandardScaler().fit(x_train.iloc[:, :2])
min_max_scaler = MinMaxScaler().fit(x_train.iloc[:, 2:])

def preprocessor(X):
    A = X.to_numpy(copy=True)
    A[:,:2]=std_scaler.transform(X.iloc[:,:2])
    A[:,2:]=min_max_scaler.transform(X.iloc[:,2:])
    return A
print(preprocessor(x_train))