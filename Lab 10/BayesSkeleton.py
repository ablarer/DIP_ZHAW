import numpy as np

import pickle

from matplotlib import pyplot as plt
from skimage import metrics


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
    mean = []
    covariance = []
    for label in labels:
        feature_matrix = np.array(features[label])
        pass
        pass
    return mean, covariance


def evaluateCost(feature_vector, m, c):
    # Input
    # feature_vector ... feature vector under test
    # m     mean of the feature vectors for a class
    # c     covariance of the feature vectors of a class
    # Output
    #   some scalar proportional to the logarithm of the probability d_j(feature_vector)
    #dj
    scalar = (-np.ln(np.abs(c)) - (feature_vector-m).T * np.linalg.inv(c) * (feature_vector-m))
    return scalar

def classify(test_data, mean, covariance):
    pass
    pass
    pass


def computeConfusionMatrix(decisions, test_data):
    confusion_matrix = metrics.confusion_matrix(test_data, decisions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.show()


def main():
    train_data = pickle.load(open("train_data.pkl", "rb"))
    test_data = pickle.load(open("test_data.pkl", "rb"))

    # Train: Compute mean and covariance for each object class from {0,1,2,3}
    # returns one list entry per object class
    mean, covariance = train(train_data)

    # scalar = evaluateCost(feature_vector, mean, covariance)
    
    # Decide: Compute decision for each feature vector from test_data
    # return a list of class indices from the set {0,1,2,3}
    decisions = classify(test_data, mean, covariance)
    print(decisions)
    
    # Copmute the confusion matrix
    confusion_matrix = computeConfusionMatrix(decisions, test_data)
    print(confusion_matrix)

if __name__ == "__main__":
    main()
