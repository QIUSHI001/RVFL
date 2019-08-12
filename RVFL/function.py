import numpy as np
import numpy.matlib
import sys
#this is the  function to train and evaluate rvfl for classification
#problem.
#option.n :      number of hidden neurons
#option.bias:    whether to have bias in the output neurons
#option.link:    whether to have the direct link.
#option.activationfunction:activation functions used.   
#option.seed:    random seeds
#option.mode     1: regularized least square, 2: moore-penrose pseudoinverse
#option.randomtype: different randomnization methods. currently only support gaussian and uniform.
#option.scale    linearly scale the random features before feedinto the nonlinear activation function. 
#                in this implementation, we consider the threshold which lead to 0.99 of the maximum/minimum value of the activation function as the saturating threshold.
#                option.scale=0.9 means all the random features will be linearly scaled
#                into 0.9* [lower_saturating_threshold,upper_saturating_threshold].
#option.scalemode scalemode=1 will scale the features for all neurons.
#                scalemode=2  will scale the features for each hidden
#                neuron separately.
#                scalemode=3 will scale the range of the randomization for
#                uniform diatribution.
#this software package has been developed by le zhang(c) 2015
#based on this paper: a comprehensive evaluation of random vector functional link neural network variants
# for technical support and/or help, please contact lzhang027@e.ntu.edu.sg
#this package has been downloaed from https://sites.google.com/site/zhangleuestc/
def RVFL_train_val(trainX,trainY,testX,testY,option):
    U_trainY=np.unique(trainY)
    nclass=U_trainY.size
    trainY_temp=np.zeros((trainY.size,nclass))
    # 0-1 coding for the target 
    for i in range (nclass):
        for j in range (trainY.size):
            if trainY[j]==U_trainY[i]:
                trainY_temp[j,i]=1
    [Nsample,Nfea]=trainX.shape
    N=option.N
    if option.RandomType=='Uniform':
        if option.Scalemode==3:
            np.random.seed(option.seed)
            Weight= option.Scale*(np.random.rand(Nfea,N)*2-1)
            np.random.seed(option.seed)
            Bias= option.Scale*np.random.rand(1,N)
        else:
            np.random.seed(option.seed)
            Weight=np.random.rand(Nfea,N)*2-1
            np.random.seed(option.seed)
            Bias=np.random.rand(1,N)
    else:
        if option.RandomType=='Gaussian':
            np.random.seed(option.seed)
            Weight=np.random.rand(Nfea,N)
            np.random.seed(option.seed)
            Bias=np.random.randn(1,N)
        else:
            print('only Gaussian and Uniform are supported')

    Bias_train=np.matlib.repmat(Bias,Nsample,1)
    H=np.matmul(trainX,Weight)+Bias_train
    
    if option.ActivationFunction.lower()=='sig' or option.ActivationFunction.lower()=='sigmoid': 
        
        if option.Scale:
            
            Saturating_threshold=np.array([-4.6,4.6])
            Saturating_threshold_activate=np.array([0,1])
            if option.Scalemode==1:
                
                [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale)
                
            elif option.Scalemode==2:

                [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale)
                print(H)
    #np.set_printoptions(threshold=sys.maxsize)            
    #print(trainY_temp.size)
    return 0



def Scale_feature(Input,Saturating_threshold,ratio):
    Min_value=Input.min()
    Max_value=Input.max()
    min_value=Saturating_threshold[0]*ratio
    max_value=Saturating_threshold[1]*ratio
    k=(max_value-min_value)/(Max_value-Min_value)
    b=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value)
    Output=Input*k+b
    return Output,k,b


def Scale_feature_separately(Input,Saturating_threshold,ratio):
    nNeurons=Input.shape[1]
    k=np.zeros((1,nNeurons))
    b=np.zeros((1,nNeurons))
    Output=np.zeros(Input.shape)
    min_value=Saturating_threshold[0]*ratio
    max_value=Saturating_threshold[1]*ratio
    for i in range(0,nNeurons):
        Min_value=np.min(Input[:,i])
        Max_value=np.max(Input[:,i])
        k[0,i]=(max_value-min_value)/(Max_value-Min_value)
        b[0,i]=(min_value*Max_value-Min_value*max_value)/(Max_value-Min_value)
        Output[:,i]=Input[:,i]*k[0,i]+b[0,i]
    return Output,k,b
