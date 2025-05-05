import numpy as np
import os
import cv2 as cv
import pandas as pd
from numpy import matlib
import nibabel as nib
from Global_Vars import Global_Vars
from Image_results import ORI_Image_Results, hist_image_result
from MPA import MPA
from Model_ALSTM_BL import Model_ALSTM_BL
from Model_ANN import Model_ANN
from Model_Resnet import Model_RESNET_FEAT
from Objective_Function import objfun_cls
from Model_Autoencoder import Model_AutoEncoder
from Model_CNN import Model_CNN
from Model_LSTM import Model_LSTM
from DHOA import DHOA
from HHO import HHO
from Jaya import JAYA
from PROPOSED import PROPOSED
from Plot_Results import plot_results_conv, ROC_Graph, plot_results

no_of_dataset = 1

# Read the Dataset
an = 0
if an == 1:
    Dataset = './nilearn_data/ABIDE_pcp/cpac/nofilt_noglobal'  # path of dataset
    file = './nilearn_data/ABIDE_pcp/Dataset.txt'
    path = os.listdir(Dataset)  # Directory of the dataset
    IMAGE = []
    Target = []
    uni = []
    Tar = []
    df = pd.read_csv(file, sep=" ")
    Values = df.values
    for t in range(len(Values)):
        print(t, len(Values))
        val = Values[t].astype(str)
        targ = val[0].split('\t')
        Tar.append(targ[2])
    Tar = np.asarray(Tar)
    for i in range(len(path)):
        print(i)
        fold_1 = Dataset + '/' + path[i]
        if '.gz' in fold_1:
            img_1 = nib.load(str(fold_1))
            image_1 = img_1.get_fdata()
            img = image_1[:, :, :, 0]
            for j in range(32, 40):
                print(i, len(path), j, img.shape)
                imge = img[:, :, j]
                imae = np.uint8(imge)
                uni.append(len(np.unique(imae)))
                image = cv.resize(imae, (512, 512))
                IMAGE.append(image)
                Target.append((Tar[i]).astype(int))
    np.save('Dataset.npy', IMAGE)
    np.save('Target.npy', np.reshape(Target, (-1, 1)))

#  Pre-processing
an = 0
if an == 1:
    Preprocess = []
    Images = np.load('Dataset.npy', allow_pickle=True)  # load the image
    for i in range(Images.shape[0]):  # for all images
        print(i)
        Image = Images[i]
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)
        enhanced = cv.convertScaleAbs(Image, alpha=alpha, beta=beta)  # Contrast Enhancement with median filter
        Preprocess.append(np.uint8(enhanced))
    np.save('Preprocess.npy', Preprocess)

# Deep feature extraction
an = 0
if an == 1:
    data = np.load('Preprocess.npy', allow_pickle=True)
    tar = np.load('Target.npy', allow_pickle=True)
    Images = Model_RESNET_FEAT(data, tar)
    np.save('RESNET_Feat.npy', Images)

# Optimization
an = 0
if an == 1:
    Feat = np.load('RESNET_Feat.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 4  # one for Hidden Neuron count, one for optimizer, one for epoch, one for batch size
    xmin = matlib.repmat(np.asarray([5, 0, 50, 4]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 4, 100, 1024]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 10

    print("HHO...")
    [bestfit1, fitness1, bestsol1, time1] = HHO(initsol, fname, xmin, xmax, Max_iter)  # HHO

    print("JAYA...")
    [bestfit2, fitness2, bestsol2, time2] = JAYA(initsol, fname, xmin, xmax, Max_iter)  # JAYA

    print("DHOA...")
    [bestfit3, fitness3, bestsol3, time3] = DHOA(initsol, fname, xmin, xmax, Max_iter)  # DHOA

    print("MPA...")
    [bestfit4, fitness4, bestsol4, time4] = MPA(initsol, fname, xmin, xmax, Max_iter)  # MPA

    print("Improved_MPA...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved-MPA

    BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                   bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', fitness)
    np.save('BestSol_CLS.npy', BestSol_CLS)

# Classification
an = 0
if an == 1:
    Eval_all = []
    Prep_Image = np.load('Preprocess.npy', allow_pickle=True)  # Load the Feat
    Target = np.load('Target.npy', allow_pickle=True)  # Load the Targets
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)  # Load the Bestsol_cls
    Feat = Prep_Image
    Learnper = [0.35, 0.45, 0.55, 0.65, 0.754, 0.85]
    for learn in range(len(Learnper)):
        learnperc = round(Feat.shape[0] * Learnper[learn])
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval = np.zeros((11, 14))
        for j in range(BestSol.shape[0]):
            print(learn, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :] = Model_ALSTM_BL(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval[5, :], pred_1 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred_2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred_3 = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred_4 = Model_AutoEncoder(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :] = Model_ALSTM_BL(Train_Data, Train_Target, Test_Data, Test_Target, [5, 0, 5, 4])
        Eval[10, :] = Eval[4, :]
        Eval_all.append(Eval)
    np.save('Eval_all.npy', Eval_all)  # Save the Eval_all


plot_results_conv()
ROC_Graph()
plot_results()
ORI_Image_Results()
hist_image_result()
