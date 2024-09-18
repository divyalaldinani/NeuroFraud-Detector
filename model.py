from sklearn.model_selection import train_test_split
import numpy as np
from NN import NN
import pandas as pd

df = pd.read_csv('creditcard.csv')
data = np.array(df)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = [int(label) for label in y]
# print(y[:10])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

NN(X_train, y_train, True)
y_test = np.array([])
pred = NN(X_test, y_test, False)
print()
acc = NN.get_accuracy(y_test)
print(acc)


# NN(X_test, [], false)
# 