import numpy as np

def accuracy(y_actual, y_predicted):
    if y_actual.shape != y_predicted.shape:
        raise Exception("Dimension miss match: array of shape {} is not compatible with an array of shape {}".format(
            y_actual.shape, y_predicted.shape))
    else:
        accuracy = np.equal(y_actual, y_predicted).mean()
        return accuracy

def confusion_matrix(y_actual, y_predicted):
        '''
        It works on the assumption that the class labels are labeled from 0 to N-1.
        where N is the number of unique categories/ classes in the data set.
        :return: A confusion matrix
        '''
        if y_actual.shape != y_predicted.shape:
            raise Exception(
                "Dimension miss match: array of shape {} is not compatible with an array of shape {}".format(
                    y_actual.shape, y_predicted.shape))
        else:
            unique_classes = np.unique(y_actual)
            num_classes = len(unique_classes)
            confusion_matrix = np.zeros((num_classes, num_classes))
            y_actual = y_actual.astype(int)
            y_predicted = y_predicted.astype(int)
            for i, j in zip(y_actual, y_predicted):
                confusion_matrix[i][j] += 1
            return confusion_matrix