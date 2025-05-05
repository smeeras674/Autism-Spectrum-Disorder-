from Evaluation import evaluation
import numpy as np
from keras.layers import Input, Conv1D, LSTM
from keras.models import Model
from keras.layers import Dense


def Model_A_LSTM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [5, 0, 5, 4]

    input_shape = (100, 20)
    optimizer = ['SGD', 'Adagrad', 'AdaDelta', 'RMSProp', 'Adam']
    train_data = np.resize(train_data, (train_data.shape[0], 100, 20))
    test_data = np.resize(test_data, (test_data.shape[0], 100, 20))

    inputs = Input(shape=input_shape)
    conv_layer = Conv1D(32, 3, activation='relu')(inputs)
    # First LSTM layer in the cascade
    lstm1 = LSTM(64, return_sequences=True)(conv_layer)
    # Second LSTM layer in the cascade
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    # Third LSTM layer in the cascade
    lstm3 = LSTM(sol[0])(lstm2)

    outputs = Dense(train_target.shape[1], activation='sigmoid')(lstm3)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer=optimizer[int(sol[1])], loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_target, epochs=int(sol[2]), batch_size=int(sol[3]))
    predict = model.predict(test_data)
    pred = np.asarray(predict)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, predict
