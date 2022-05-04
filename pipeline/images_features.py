import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
from skimage.feature import texture
from tqdm import trange
import sys, time
import time


class ImgFeature:
    def __init__(self, imgList_rgb, imgList_gray, imgList_gbr):
        self.imgList_rgb = imgList_rgb
        self.imgList_gray = imgList_gray
        self.imgList_gbr = imgList_gbr

    def NN_feature(self, method):
        """
        extract images feature using imagenet
        :param model: One of {VGG19", "VGG16"}
        :return: a list of images feature
        """
        featureList = []
        if method == "VGG19":
            model = VGG19(weights='imagenet',
                          include_top=False,
                          input_shape=(200, 200, 3))

        elif method == "VGG16":
            model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=(200, 200, 3))

        elif method == 'ResNet50':
            model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_tensor=None,
                input_shape=(200, 200, 3),
                pooling=None,
                # classes=1000,
            )

        model.summary()

        for i in range(len(self.imgList_rgb)):
            x = np.expand_dims(self.imgList_rgb[i], axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            featureList.append(features)
            if i + 1 == len(self.imgList_rgb):
                percent = 100.0
                print('current progress : %s [%d/%d]' % (str(percent) + '%', i + 1, len(self.imgList_rgb)), end='\n')
            else:
                percent = round(1.0 * i / len(self.imgList_rgb) * 100, 2)
                print('current progress : %s [%d/%d]' % (str(percent) + '%', i + 1, len(self.imgList_rgb)), end='\r')
            time.sleep(0.01)
        return featureList

    def get_glcm(self, prop):
        """
        get GLCM of image in gray mode
        :imgList: List of images in gray mode
        :prop: The property of the GLCM to compute, one of{'contrast','dissimilarity','homogeneity','energy','correlation','ASM'}
        :return: a list of texture properties of a GLCM.
        """
        resultList = []
        for i in range(len(self.imgList_gray)):
            result = texture.greycomatrix(self.imgList_gray[i], [1, 4], [0, np.pi / 2], normed=True, symmetric=True)
            feature = texture.greycoprops(result, prop)
            resultList.append(feature)
        return resultList

    def get_lbp(self, method):
        """
        get LBP of image in GRAY mode
        :imgList: List of images in GRAY mode
        :method: Method to determine the pattern.{'default','ror','uniform','var'}
        :return: a list of LBP images.
        """
        resultList = []
        for i in range(len(self.imgList_gray)):
            feature = texture.local_binary_pattern(self.imgList_gray[i], 8, np.pi / 2, method=method)
            resultList.append(feature)
        return resultList

    def color_moments(self):
        """
        Compute low order moments(1,2,3)
        :imgList: List of images in RGB mode
        :return: a list of color moment
        """
        momentsList = []
        for i in range(len(self.imgList_gbr)):
            img = self.imgList_gbr[i]
            # Convert BGR to HSV colorspace
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Split the channels - h,s,v
            h, s, v = cv2.split(hsv)
            # Initialize the color feature
            color_feature = []
            # N = h.shape[0] * h.shape[1]
            # The first central moment - average 
            h_mean = np.mean(h)  # np.sum(h)/float(N)
            s_mean = np.mean(s)  # np.sum(s)/float(N)
            v_mean = np.mean(v)  # np.sum(v)/float(N)
            color_feature.extend([h_mean, s_mean, v_mean])
            # The second central moment - standard deviation
            h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
            s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
            v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
            color_feature.extend([h_std, s_std, v_std])
            # The third central moment - the third root of the skewness
            h_skewness = np.mean(abs(h - h.mean()) ** 3)
            s_skewness = np.mean(abs(s - s.mean()) ** 3)
            v_skewness = np.mean(abs(v - v.mean()) ** 3)
            h_thirdMoment = h_skewness ** (1. / 3)
            s_thirdMoment = s_skewness ** (1. / 3)
            v_thirdMoment = v_skewness ** (1. / 3)
            color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
            momentsList.append(color_feature)
        return momentsList


def features_data_filter(df_train, df_test, dropnRecs, dropZeroctr, dropnReads):
    if dropnRecs == 1:
        index = df_test[df_test['nRecs'] <= 10].index
        df_train = df_train.drop(index)
        df_test = df_test.drop(index)

    if dropZeroctr == 1:
        index = df_test[df_test['ctr'] == 0].index
        df_train = df_train.drop(index)
        df_test = df_test.drop(index)

    if dropnReads == 1:
        index = df_test[df_test['nReads'] == 0].index
        df_train = df_train.drop(index)
        df_test = df_test.drop(index)
    return df_train, df_test


def feature_reshape(List):
    reshapeList = []
    for i in range(len(List)):
        features_reshape = np.array(List[i]).flatten()
        reshapeList.append(features_reshape)
    return reshapeList
