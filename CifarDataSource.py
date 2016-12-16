import numpy as np
import random

class DataSource:

    __trainLabel=None
    __trainImage=None
    __testLab=None
    __testImage=None

    def singleClassToShape(self,ary,class_size):
        result=np.zeros([ary.shape[0],class_size],dtype='float32')
        for i in range(0, ary.shape[0]):
            result[i][ary[i]] = 1
        return result

    def __init__(self):
        bytesStream = open('./cifar_10/test_batch.bin', 'rb')
        buf = bytesStream.read(10000 * (32 * 32 * 3 + 1))
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(10000, 32 * 32 * 3 + 1)
        data = np.hsplit(data, [1])
        self.__testLab = data[0].reshape(10000)
        self.__testLab = self.singleClassToShape(self.__testLab, 10)
        self.__testImage = data[1]
        targetImage=self.__testImage.copy()
        print('Reading Image From Cifar-10 test set')
        for j in range(0, 10000):
            for k in range(0, 32 * 32):
                targetImage[j][3 * k + 0] = self.__testImage[j][k]
                targetImage[j][3 * k + 1] = self.__testImage[j][k + 32 * 32]
                targetImage[j][3 * k + 2] = self.__testImage[j][k + 32 * 32 * 2]

        self.__testImage=targetImage.reshape(10000, 32, 32, 3)

        print('Reading Image From Cifar-10 DataSet')
        for i in range(1,6):
            print('sub batch :'+str(i))
            bytesStream = open('./cifar_10/data_batch_'+str(i)+'.bin', 'rb')
            buf = bytesStream.read(10000 * (32 * 32 * 3 + 1))
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(10000, 32 * 32 * 3 + 1)
            data = np.hsplit(data, [1])
            tempLabel=data[0].reshape(10000)
            tempLabel=self.singleClassToShape(tempLabel, 10)
            tempImage=data[1]
            targetImage=tempImage.copy()
            for j in range(0,10000):
                for k in range(0,32*32):
                    targetImage[j][3*k+0]=tempImage[j][k]
                    targetImage[j][3*k+1]=tempImage[j][k+32*32]
                    targetImage[j][3*k+2]=tempImage[j][k+32*32*2]

            tempImage=targetImage.reshape(10000, 32, 32, 3)
            if self.__trainLabel==None:
                self.__trainLabel=tempLabel
                self.__trainImage=tempImage
            else:
                self.__trainLabel=np.concatenate((self.__trainLabel,tempLabel),axis=0)
                self.__trainImage=np.concatenate((self.__trainImage,tempImage),axis=0)

    def getTrainBatch(self,batch_size):
        place=random.randint(0,50000-batch_size)
        return self.__trainImage[place:(place+batch_size)],self.__trainLabel[place:(place+batch_size)]

    def getTestData(self):
        return self.__testImage,self.__testLab

    def getTestBatch(self):
        place=random.randint(0,9000)
        return self.__testImage[place:(place+1000)],self.__testLab[place:(place+1000)]