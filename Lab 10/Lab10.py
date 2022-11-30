import pickle

import numpy as np


class Data:
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label


def prepare_data(data):
    labels = np.unique(np.array([dat.label for dat in data]))
    class_dim = len(labels)
    features = [[] for i in range(class_dim)]
    for dat in data:
        features[dat.label].append(dat.feature)
    return labels, features


def train(train_data):
    # Return
    # mean ... list with one entry per class
    #          each entry is the mean of the feature vectors of a class
    # covariance ... list with one entry per class
    #          each entry is the covariance of the feature vectors of a class
    labels, features = prepare_data(train_data)
    means = []
    covariances = []

    for label in labels:
        feature_matrix = np.array(features[label])
        mean_vector = feature_matrix.mean(axis=0)

        covariance_matrix = np.zeros((feature_matrix.shape[1], feature_matrix.shape[1]))
        for feature_vector in feature_matrix:
            q = feature_vector - mean_vector
            covariance_matrix += q * np.reshape(q, (q.shape[0], 1))
        N = feature_matrix.shape[0]
        covariance_matrix = 1 / (N - 1) * covariance_matrix

        means.append(mean_vector)

        covariances.append(covariance_matrix)
        # Degrade the classifier by assuming that the covariance matrix is the unity matrix.
        # As a result you should observe misclassifications:
        # covariances.append(np.identity(covariance_matrix.shape[0]))

    return means, covariances


def evaluateCost(feature_vector, m, c):
    # Input
    # feature_vector ... feature vector under test
    # m     mean of the feature vectors for a class
    # c     covariance of the feature vectors of a class
    # Output
    #   some scalar proportional to the logarithm for the probability d_j(feature_vector)
    vec = feature_vector - m
    if np.linalg.det(c) == 0:
        raise ValueError("Covariance matrix is singular and therefore can't be inverted: " + str(c))
    return -np.log(np.linalg.det(c)) - vec.T @ np.linalg.inv(c) @ vec


def classify(test_data, mean, covariance):
    decisions = []

    for data in test_data:
        decision_cost = None
        decision_label = None

        for label in range(0, len(mean)):
            cost = evaluateCost(data.feature, mean[label], covariance[label])

            if decision_cost is None or decision_cost < cost:
                decision_cost = cost
                decision_label = label

        decisions.append(decision_label)

    return decisions


def computeConfusionMatrix(decisions, test_data):
    labels, _ = prepare_data(test_data)
    confusion_matrix = np.zeros((labels.shape[0], labels.shape[0]))

    for i, data in enumerate(test_data):
        actual = data.label
        predicted = decisions[i]

        confusion_matrix[predicted, actual] += 1

    return confusion_matrix


def main():
    train_data = pickle.load(open("train_data.pkl", "rb"))
    test_data = pickle.load(open("test_data.pkl", "rb"))

    # Train: Compute mean and covariance for each object class from {0,1,2,3}
    # returns one list entry per object class
    mean, covariance = train(train_data)

    # Decide: Compute decision for each feature vector from test_data
    # return a list of class indices from the set {0,1,2,3}
    decisions = np.array(classify(test_data, mean, covariance))
    actual = np.array([dat.label for dat in test_data])
    accuracy = np.sum(actual == decisions) / decisions.shape[0]
    print("Decisions:\t" + str(decisions))
    print("Actual:\t\t" + str(actual))
    print("Accuracy:\t" + str(accuracy))

    # Compute the confusion matrix
    confusion_matrix = computeConfusionMatrix(decisions, test_data)
    print("Confusion Matrix:\n" + str(confusion_matrix))


if __name__ == "__main__":
    main()

#
# Output:
# -----------------------------------------------------------
# Decisions:	[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]
# Actual:		[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]
# Accuracy:	1.0
# Confusion Matrix:
# [[5. 0. 0. 0.]
#  [0. 5. 0. 0.]
#  [0. 0. 5. 0.]
#  [0. 0. 0. 5.]]
#
# Degrade the classifier by assuming that the covariance matrix is the unity matrix.
# As a result you should observe wrong classifications:
#
# Decisions:	[0 0 2 0 0 1 1 1 1 1 2 0 0 2 3 2 3 0 3 0]
# Actual:		[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]
# Accuracy:	0.65
# Confusion Matrix:
# [[4. 0. 2. 2.]
#  [0. 5. 0. 0.]
#  [1. 0. 2. 1.]
#  [0. 0. 1. 2.]]
#
