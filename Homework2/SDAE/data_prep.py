# Please refer to the description for categorization of data
# We apply hold-out data split by having mutually exclusive data for training, validation, and testing
# Note: The dataset was already extracted, grouped into different files for simplicity including
#   1. X_train1D.pkl, y_train1D.pkl >> Training instances
#       -- Dimension (760000,5000) for X
#       -- Format of X : [[Packet Directions,...] e.g. [[+1,+1,-1,...,0,0],...,[+1,-1,-1,...,0,0]]
#       -- Dimension (760000,) for y
#       -- Format of Y : [site's label,...] e.g. [25, 7, 9, 0, 94, ... ,52 ]
#       -- already shuffle and ready to feed to train the classifier
#   2. X_valid1D.pkl, y_valid1D.pkl >> Validation instances
#       -- Dimension (9500, 5000) for X
#       -- Dimension (9500,) for Y
#       -- Same format as X_train1D, y_train1D
#   3. X_test1D.pkl, y_test1D.pkl >> Test instances
#       -- Dimension (9500, 5000) for X
#       -- Dimension (9500,) for Y
#       -- Same format as X_train1D, y_train1D

import pickle
import numpy as np

DATASET_DIR = "dataset"


def LoadDataMon_Large():
    return load_data("Large")

def LoadDataMon_Small():
    return load_data("Small")

def load_data(size):

    with open('{0}/{1}/X_train1D_{1}.pkl'.format(DATASET_DIR, size), 'rb') as handle:
        X_train = pickle.load(handle, encoding='iso-8859-1')
    with open('{0}/{1}/y_train1D_{1}.pkl'.format(DATASET_DIR, size), 'rb') as handle:
        Y_train = pickle.load(handle, encoding='iso-8859-1')
    with open('{0}/{1}/X_valid1D_{1}.pkl'.format(DATASET_DIR, size), 'rb') as handle:
        X_valid = pickle.load(handle, encoding='iso-8859-1')
    with open('{0}/{1}/y_valid1D_{1}.pkl'.format(DATASET_DIR, size), 'rb') as handle:
        Y_valid = pickle.load(handle, encoding='iso-8859-1')
    with open('{0}/{1}/X_test1D_{1}.pkl'.format(DATASET_DIR, size), 'rb') as handle:
        X_test = pickle.load(handle, encoding='iso-8859-1')
    with open('{0}/{1}/y_test1D_{1}.pkl'.format(DATASET_DIR, size), 'rb') as handle:
        Y_test = pickle.load(handle, encoding='iso-8859-1')

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test