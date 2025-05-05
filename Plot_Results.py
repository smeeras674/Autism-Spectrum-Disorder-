import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle
from Image_results import ORI_Image_Results, hist_image_result


def plot_results():
    eval1 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'HHO-ACAL-BL', 'JAYA-ACAL-BL', 'DHOA-ACAL-BL', 'MPA-ACAL-BL', 'RFMPA-ACAL-BL']
    Classifier = ['TERMS', 'LSTM', 'CNN', 'ANN', 'AUTOENCODER', 'LSTM-BAYESIAN_LEARNING', 'RFMPA-ACAL-BL']

    value1 = eval1[0, 4, :, 4:]

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value1[j, :])
    print('-------------------------------------------------- Algorithm Comparison',
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
    print('-------------------------------------------------- Classifier Comparison',
          '--------------------------------------------------')
    print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval1.shape[1], eval1.shape[2] + 1))
        for k in range(eval1.shape[1]):
            for l in range(eval1.shape[2]):
                    if j == 10:
                        Graph[k, l] = eval1[0, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[0, k, l, Graph_Terms[j] + 4]

        X = np.arange(len(learnper))
        plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='*', markerfacecolor='blue', markersize=12,
                 label="HHO-ACAL-BL")
        plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
                 label="JAYA-ACAL-BL")
        plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='*', markerfacecolor='green', markersize=12,
                 label="DHOA-ACAL-BL")
        plt.plot(learnper, Graph[:, 3], color='c', linewidth=3, marker='*', markerfacecolor='yellow', markersize=12,
                 label="MPA-ACAL-BL")
        plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='white',markersize=12,
                 label="RFMPA-ACAL-BL")
        plt.xticks(X, ('35', '45', '55', '65', '75', '85'))
        plt.xlabel('Learning Percentage (%)')
        plt.ylabel(Terms[Graph_Terms[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/%s_line_lrean.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.bar(X + 0.00, Graph[:, 5], color='r', edgecolor='w', width=0.10, label="LSTM")
        ax.bar(X + 0.10, Graph[:, 6], color='g', edgecolor='k', width=0.10, label="CNN")
        ax.bar(X + 0.20, Graph[:, 7], color='b', edgecolor='w', width=0.10, label="ANN")
        ax.bar(X + 0.30, Graph[:, 8], color='y', edgecolor='k', width=0.10, label="AUTOENCODER")
        ax.bar(X + 0.40, Graph[:, 9], color='c', edgecolor='w', width=0.10, label="LSTM-BAYESIAN_LEARNING")
        ax.bar(X + 0.50, Graph[:, 4], color='k', edgecolor='k', width=0.10, label="RFMPA-ACAL-BL")
        plt.xticks(X + 0.10, ('35', '45', '55', '65', '75', '85'))
        plt.xlabel('Learning Percentage (%)')
        plt.ylabel(Terms[Graph_Terms[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/%s_bar_lrean.png" % (Terms[Graph_Terms[j]])
        plt.savefig(path1)
        plt.show()

def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_results_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'HHO-ACAL-BL', 'JAYA-ACAL-BL', 'DHOA-ACAL-BL', 'MPA-ACAL-BL', 'RFMPA-ACAL-BL']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):
        Conv_Graph[j, :] = stats(Fitness[0, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report ',
          '--------------------------------------------------')

    print(Table)
    length = np.arange(10)
    Conv_Graph = Fitness[0]

    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red', markersize=12,
             label='HHO-ACAL-BL')
    plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12,
             label='JAYA-ACAL-BL')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12,
             label='DHOA-ACAL-BL')
    plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12,
             label='MPA-ACAL-BL')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12,
             label='RFMPA-ACAL-BL')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png")
    plt.show()


def ROC_Graph():
    lw=2
    cls = ['LSTM', 'CNN', 'ANN', 'AUTOENCODER', 'LSTM-BAYESIAN_LEARNING', 'RFMPA-ACAL-BL']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "aqua", "black"])
    for i, color in zip(range(6), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path1 = "./Results/ROC.png"
    plt.savefig(path1)
    plt.show()


if __name__ == '__main__':
    plot_results_conv()
    ROC_Graph()
    plot_results()
    ORI_Image_Results()
    hist_image_result()
