import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

no_of_dataset = 1


def ORI_Image_Results():
    Images = np.load('Dataset.npy', allow_pickle=True)
    Image = [25, 30, 50, 100, 105]
    for i in range(len(Image)):
        # fig, ax = plt.subplots(1, 5)
        plt.subplot(1, 5, 1).axis('off')
        plt.title('Image 1')
        plt.imshow(Images[Image[0]])

        plt.subplot(1, 5, 2).axis('off')
        plt.title('Image 2')
        plt.imshow(Images[Image[1]])

        plt.subplot(1, 5, 3).axis('off')
        plt.title('Image 3')
        plt.imshow(Images[Image[2]])

        plt.subplot(1, 5, 4).axis('off')
        plt.title('Image 4')
        plt.imshow(Images[Image[3]])

        plt.subplot(1, 5, 5).axis('off')
        plt.title('Image 5')
        plt.imshow(Images[Image[4]])

        plt.tight_layout()
        # path = "./Results/Image_Results/Image_%s.png" % (i + 1)
        # plt.savefig(path)
        plt.show()
        # cv.imwrite('./Results/Image_Results/Dataset-orig-' + str(i + 1) + '.png', Images[Image[i]])


def hist_image_result():
    Images = np.load('Dataset.npy', allow_pickle=True)
    Image = [25, 30, 50, 100, 105]
    for i in range(len(Image)):
        equ = cv.equalizeHist(Images[Image[i]])
        cv.imshow('Equalised image', equ)
        cv.waitKey(0)


if __name__ == '__main__':
    ORI_Image_Results()
    hist_image_result()
