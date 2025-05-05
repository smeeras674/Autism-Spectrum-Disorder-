from Evaluation import evaluation
from Model_ALSTM import Model_A_LSTM
from Model_BL import Model_BL


def Model_ALSTM_BL(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [5, 0, 5, 4]
    print('Model_ALSTM_BL')
    Eval, pred_RNN = Model_A_LSTM(train_data, train_target, test_data, test_target, sol=sol)
    Eval, pred_LSTM = Model_BL(train_data, train_target, test_data, test_target)
    pred = (pred_RNN + pred_LSTM) / 2
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval
