import numpy as np
from sklearn.naive_bayes import GaussianNB
from Evaluation import evaluation


def Model_BL(train_data,train_target,test_data,test_target):
    if len(train_data.shape) == 3:
        train_data = np.resize(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
        test_data = np.resize(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))

    # Create a Naive Bayes classifier
    model = GaussianNB()
    model.fit(train_data, train_target)

    # Predict the labels of the images
    pred = model.predict(test_data)
    pred = np.reshape(pred, (-1, 1))
    Eval = evaluation(pred, test_target)
    return Eval, pred

