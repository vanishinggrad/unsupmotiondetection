import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, accuracy_score, normalized_mutual_info_score, adjusted_rand_score



def get_train_test_data(file_prefix, posfix, data_type, label_type, data_shape):
    x_train = np.fromfile(file_prefix + 'x_train' + posfix + '.dat', dtype=data_type)
    x_test = np.fromfile(file_prefix + 'x_test' + posfix + '.dat', dtype=data_type)
    x_train = np.reshape(x_train, (-1,) + data_shape)
    x_test = np.reshape(x_test, (-1,) + data_shape)

    y_train = np.fromfile(file_prefix + 'y_train' + posfix + '.dat', dtype=label_type)
    y_test = np.fromfile(file_prefix + 'y_test' + posfix + '.dat', dtype=label_type)

    print('train ds size ', x_train.shape, y_train.shape)
    unique, unique_counts = np.unique(y_train, return_counts=True)
    print('label counts | train : ', unique, unique_counts)

    print('test ds size ', x_test.shape, y_test.shape)
    unique, unique_counts = np.unique(y_test, return_counts=True)
    print('label counts | test : ', unique, unique_counts)

    return x_train, y_train, x_test, y_test



def clust_best_mapping(true_clusters, predicted_clusters, return_accuracy=True):
    cm = confusion_matrix(true_clusters, predicted_clusters)
    cost = -cm + cm.max()

    # note: unique_true would also work, but unique_combo is more general
    unique_true = np.unique(true_clusters)
    unique_predicted = np.unique(predicted_clusters)
    unique_combo = np.unique(np.hstack((unique_predicted, unique_true)), axis=0)

    row_ind, col_ind = linear_sum_assignment(cost)

    # placeholder
    best_mapping = np.copy(predicted_clusters)

    for (i, j) in zip(row_ind, col_ind):
        best_mapping[predicted_clusters == unique_combo[j]] = unique_combo[i]

    if return_accuracy:
        return best_mapping, accuracy_score(true_clusters, best_mapping)

    return best_mapping


def accuracy(true_row_labels, predicted_row_labels):
    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    cost = -cm + np.max(cm)
    indexes = linear_sum_assignment(cost)
    total = 0
    for row, column in zip(*indexes):
        value = cm[row][column]
        total += value
    return (total * 1. / np.sum(cm))


nmi = normalized_mutual_info_score
ari = adjusted_rand_score
acc = accuracy
