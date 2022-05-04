import os
import cv2
import numpy as np
# import string


class LoadImg:
    def __init__(self, df, dirPath):
        self.nameList = []
        self.imgList = []
        self.df = df
        self.dirPath = dirPath

    def data_filter(self, identifier):
        """
        Args:
        identifier: reference in the dataframe
        dropZeroctr: choice of {1,0}
        dropnRecs: choice of {1,0}

        Returns:filtered dataframe
        """
        df_new = self.df
        # [['article', 'ctr', 'nRecs']]
        df_new = df_new.dropna(axis=0, how='any', inplace=False)
        # if dropnRecs == 1:
        #     df_new = df_new.drop(df_new[df_new['nRecs'] < 10].index)
        # if dropZeroctr == 1:
        #     df_new = df_new.drop(df_new[df_new['ctr'] == 0].index)
        # if dropnReads == 1:
        #     df_new = df_new.drop(df_new[df_new['nReads'] == 0].index)
        #
        self.df = df_new
        self.fileList = df_new[identifier].tolist()

        return df_new

    def load_image(self, color_mode, norm):
        """
        Load the images and rize the images into 200*200
        :param dirPath: directory Path
        :param color_mode: One of {"RGB", "GRAY", "HSV","BGR"}
        :norm: normalization or not choice of {0,1}
        :return: a dict stores images according to article id
        a list contains images
        a list contains the name of the file according to article id
        """
        # Traverse the files in the directory
        # osfileList = [str(x)+'.jpg' for x in self.fileList]
        self.imgList = list()
        self.newList = list()
        self.nameList = list()
        for file in self.fileList:
            # Determine whether it is a file
            if os.path.isfile(os.path.join(self.dirPath, file)) == True:
                c = os.path.basename(file)
                name = self.dirPath + c
                img = cv2.imread(name)
                img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)  # rize to 200,200
                if color_mode == "HSV":
                    img = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # RGB to HSV
                elif color_mode == "GRAY":
                    img = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  # RGB to GRAY
                elif color_mode == "RGB":
                    img = img[:, :, [2, 1, 0]]  # BGR to RGB
                elif color_mode == "BGR":
                    img = img

                if norm == 1:
                    self.imgList.append(img / 255)
                else:
                    self.imgList.append(img)

                self.nameList.append(str(c))
                index = self.df[(self.df.imgFile == str(c))].index.tolist()
                self.newList.append(int(self.df.iid[index]))
        #         # normalization
        # if norm == 1:
        #     for i in range(len(self.imgList)):
        #         self.imgList[i] = self.imgList[i] / 255

        # data cleaning
        a = [x for x in self.nameList if x not in self.fileList]
        for i in range(len(a)):
            index = self.nameList.index(a[i])
            self.imgList.pop(index)
            self.newList.pop(index)
        return self.imgList, self.newList


    def topctr_label(self):
        """

        Args:
        df: filtered dataframe
        newList: cleaned filename list
        Returns: ctr label, in top 100 labeled as 1, not in top 100 labeled as 0

        """
        ctrList = []
        for i in range(len(self.newList)):
            ctr = self.df.loc[self.df['article'] == self.newList[i]]['ctr'].tolist()[0]
            if ctr > self.df.sort_values("ctr", ascending=False, ignore_index=True).at[100, 'ctr']:
                ctrList.append(1)
            else:
                ctrList.append(0)
        return ctrList

    def zeroctr_label(self, uid):
        """
        Args:
        df: filtered dataframe
        newList: cleaned filename list
        uid : uid of article, marchdata is iid
        Returns: ctr label, non-zero ctr as 1, zero ctr as 0
        """
        ctrList = []
        for i in range(len(self.newList)):
            ctr = self.df.loc[self.df[uid] == int(self.newList[i])]['ctr'].tolist()[0]
            if ctr == 0:
                ctrList.append(0)
            else:
                ctrList.append(1)
        return ctrList


# def reshape(List):
#     """
#
#     Args:
#         List: feature List extracted by deep learning method
#
#     Returns: flattened feature List
#
#     """
#     reshapeList = []
#     n, x, y, z = np.shape(List[0])
#     for i in range(len(List)):
#         features_reshape = List[0].reshape((n * x * y * z))
#         reshapeList.append(features_reshape)
#     return reshapeList

