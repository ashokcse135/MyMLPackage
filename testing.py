from MyML import *
import numpy as np

y_actual = np.array([1, 0, 1, 0, 1, 0, 2, 2])
y_predicted = np.array([1.0, 0.0, 0.0, 0, 1, 1, 2, 0])
acc = metrics.accuracy(y_actual, y_predicted)
print(acc)
cm = metrics.confusion_matrix(y_actual, y_predicted)
print(cm)
