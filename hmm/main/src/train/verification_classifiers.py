from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np

def get_logisitc_regressor(positive_data, negative_data, random_state):
    labels_correct = np.ones(positive_data.shape)
    labels_incorrect = np.zeros(negative_data.shape)
    train_data = np.concatenate((positive_data, negative_data), axis=None).reshape((-1, 1))
    train_labels = np.concatenate((labels_correct, labels_incorrect), axis=None).reshape((-1,))

    clf = LogisticRegression(random_state=random_state, solver='sag').fit(train_data, train_labels)
    return clf

def get_neural_net_classifier(positive_data, negative_data, random_state):
    labels_correct = np.ones(positive_data.shape[0])
    labels_incorrect = np.zeros(negative_data.shape[0])
    train_data = np.concatenate((positive_data, negative_data), axis=0)
    train_labels = np.concatenate((labels_correct, labels_incorrect), axis=0).reshape((-1,))
    clf = MLPClassifier(hidden_layer_sizes=(50,50), learning_rate_init=0.0001, learning_rate='adaptive').fit(train_data, train_labels)
    return clf