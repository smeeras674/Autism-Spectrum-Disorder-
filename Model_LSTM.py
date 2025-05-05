import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Evaluation import evaluation


def Model_LSTM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [50, 2]
    out, model = LSTM_train(train_data, train_target, test_data, sol)
    pred = np.asarray(out)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred


def LSTM_train(trainX, trainY, testX, sol):
    trainX = np.resize(trainX, (trainX.shape[0], 1, 200))
    testX = np.resize(testX, (testX.shape[0], 1, 200))
    model = Sequential()
    model.add(LSTM(int(sol[0]), input_shape=(1, 200)))
    model.add(Dense(trainY.shape[1]))
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=int(sol[1]), batch_size=1, verbose=2)

    testPredict = model.predict(testX)
    return testPredict, model

