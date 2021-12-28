import numpy as np

def predict(X_test):    
    # number of example
    W1 = np.loadtxt('trained_param\\W1.txt')
    W2 = np.loadtxt('trained_param\\W2.txt')
    b1 = np.loadtxt('trained_param\\b1.txt')
    b2 = np.loadtxt('trained_param\\b2.txt')

    m = X_test.shape[0]
    y_pred = np.zeros((1,m))
    A = sigmoid(X_test.dot(W1) + b1) 
    y_pred = softmax(A.dot(W2) + b2) 
    pred = np.zeros(X_test.shape[0])
    temp = np.where(y_pred==y_pred.max())
    temp = str(temp[0]).lstrip('[').rstrip(']')   
    pred = int(temp)
    genre = mapping(pred)
    return genre


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=0, keepdims=True)


def mapping(label):

    my_dict = dict()
    my_dict = {
    0: 'Blues',
    1: 'Country',
    2: 'Electronic',
    3: 'Folk',
    4: 'International',
    5: 'Jazz',
    6: 'Latin',
    7: 'New Age',
    8: 'Pop_Rock',
    9: 'Rap',
    10: 'Reggae',
    11: 'RnB',
    12: 'Vocal'
    }
    return my_dict[label]