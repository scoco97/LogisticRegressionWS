# Required Python Packages
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

FILE_NAME_TRAIN = 'train.csv' 
FILE_NAME_TEST = 'test.csv'

def dataset_headers(dataset):
    return list(dataset.columns.values)

def train_logistic_regression(train_x, train_y):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model

def model_accuracy(trained_model, features, targets):
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score

def main():
    train_dataset = pd.read_csv(FILE_NAME_TRAIN)
    test_dataset = pd.read_csv(FILE_NAME_TEST)
    headers = dataset_headers(train_dataset)
    training_features = ['company_rating', 'model_rating', 'bought_at', 'months_used', 'issues_rating', 'resale_value']
    target = ['output']
    # print "Number of Observations :: ", len(dataset)
    # print "Data set headers :: {headers}".format(headers = headers)
    # print dataset.head()
    train_x = train_dataset[training_features]
    train_y = train_dataset[target]
    test_x = test_dataset[training_features]
    test_y = test_dataset[target]
    trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
    print "Accuracy of Trained Model : " , model_accuracy(trained_logistic_regression_model, train_x, train_y)
    print "Accuracy of Test Model : " , model_accuracy(trained_logistic_regression_model, test_x, test_y)

if __name__ == "__main__":
    main()