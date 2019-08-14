import numpy as np
import random
from function import RVFL_train_val
from car_R import data
from car_conxuntos_kfold import index   #4-fold CV
from option import option as op


data = data[:,1:]
dataX = data[:,0:-1]
dataY = data[:,[-1]]
#print(dataX)


# do normalization for each feature
dataX_mean=np.mean(dataX,axis=0)
dataX_std=np.std(dataX,axis=0)
dataX=(dataX-dataX_mean)/dataX_std

ACC_CV = np.zeros((1,4))
train_accuracy = np.zeros((1,4))


#Look at the documentation of RVFL_train_val function file 
option=op(100,False,True,'radbas',0,1,'Uniform',1,1)
option.N = 405
option.C = 1
option.bias = 1
option.link = 1
option.mode = 1
option.ActivationFunction='sig'
option.Scalemode=2
#a=np.array([1,2,3,4,5,6])
#test=dataX[a,:]
#print(test)
for i in range(0,4):

    trainX = dataX[index[2*i-2],:]
    trainY = dataY[index[2*i-2],:]


    testX = dataX[index[2*i-1],:]
    testY = dataY[index[2*i-1],:]
    train_accuracy[0,i],ACC_CV[0,i] = RVFL_train_val(trainX,trainY,testX,testY,option)
 
    
# print(train_accuracy)
Train_Accuarcy = np.mean(train_accuracy,axis=1)
Test_Accuracy = np.mean(ACC_CV,axis=1)


print('The average training accuracy is: %.4f'%Train_Accuarcy)
print('The average testing accuracy is: %.4f'%Test_Accuracy)