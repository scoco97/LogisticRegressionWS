import numpy as np
import logging
import json
from utility import *

FILE_NAME_TRAIN = 'train.csv' 
FILE_NAME_TEST = 'test.csv' 
ALPHA = 2.7
EPOCHS = 50000
MODEL_FILE = 'models/model1'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
def appendIntercept(X):
    col = np.ones((X.shape[0],1))
    X = np.hstack((col,X))
    return X

def initialGuess(n_thetas):
    return np.zeros(n_thetas)

def train(theta, X, y, model):
     J = [] 
     m = len(y)
     for iteration in range(1,EPOCHS):
        y_predicted = predict(X,theta)
        cost = costFunc(m,y,y_predicted)
        J.append(cost)
        gradients = calcGradients(X,y,y_predicted,m)
        theta = makeGradientUpdate(theta,gradients)
     model['J'] = J
     model['theta'] = list(theta)
     return model

def costFunc(m,y,y_predicted):
    return (-y)*np.log(y_predicted) - (1 - y)*np.log(1 - y_predicted)

def calcGradients(X,y,y_predicted,m):
    return np.dot((y_predicted-y),X)/m

def makeGradientUpdate(theta, grads):
    return theta - ALPHA * grads

def predict(X,theta):
    hypo = 1 + np.exp(-1 * np.dot(X,theta.T))
    return 1. / hypo


######################## Main Function ###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        print "TRAINING DATA ACCURACY : " , accuracy(X,y,model)
        X_df, y_df = loadData(FILE_NAME_TEST)
        X,y = normalizeTestData(X_df, y_df, model)
        X = appendIntercept(X)
        print "TESTING DATA ACCURACY : " , accuracy(X,y,model)

if __name__ == '__main__':
    main()
