import numpy as np
import numpy.matlib
import sys
# hard limit activation function
hardlim = (lambda x: np.array(x > 0.0, dtype=double))
# triangular activation function
tribas = (lambda x: np.clip(1.0 - np.fabs(x), 0.0, 1.0))
# radbas activation function
radbas = (lambda x: np.exp(-x**2))
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

        H = 1 / (1 + np.exp(-H))
       

    elif option.ActivationFunction.lower()=='sin' or option.ActivationFunction.lower()=='sine':

        if option.Scale:
            
            Saturating_threshold=np.array([-np.pi/2,np.pi/2])
            Saturating_threshold_activate=np.array([-1,1])
            if option.Scalemode==1:
                
                [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale)
                
            elif option.Scalemode==2:

                [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale)
                
        H=np.sin(H)
        #print(H.min(),H.max())

    elif option.ActivationFunction.lower()=='hardlim':

        H=hardlim(H)

    elif option.ActivationFunction.lower()== 'tribas':
        if option.Scale:

            Saturating_threshold=np.array([-1,1])
            Saturating_threshold_activate=np.array([0,1])
            if option.Scalemode==1:

                [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale)
            elif option.Scalemode==2:
                [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale)
              
        H=tribas(H)

    elif option.ActivationFunction.lower()== 'radbas':
        if option.Scale:
           
            Saturating_threshold=np.array([-2.1,2.1])
            Saturating_threshold_activate=np.array([0,1])
            if option.Scalemode==1:

                [H,k,b]=Scale_feature(H,Saturating_threshold,option.Scale)
            elif option.Scalemode==2:   
                
                [H,k,b]=Scale_feature_separately(H,Saturating_threshold,option.Scale);

        H = radbas(H)

    elif option.ActivationFunction.lower()== 'sign':

        H=np.sign(H)
    
    if option.bias:
        H=np.concatenate([H,np.ones((Nsample,1))],axis=1)

    if option.link:

        if option.Scalemode==1:
            trainX_temp=trainX*k+b
            H=np.concatenate([H,trainX_temp],axis=1)
            
        elif option.Scalemode==2:
            [trainX_temp,ktr,btr]=Scale_feature_separately(trainX,Saturating_threshold_activate,option.Scale)
            H=np.concatenate([H,trainX_temp],axis=1)
            
        else:
            H=np.concatenate([H,trainX],axis=1)
            
    H[np.isnan(H)]=0
    
    if option.mode==2:
        beta=np.matmul(np.linalg.pinv(H),trainY_temp)
        #print(beta.shape)

    elif option.mode==1:

        if not option.C:
            option.C=0.1

        C=option.C

        if N<Nsample:
            beta=np.matmul(np.matmul(np.linalg.inv(np.identity(H.shape[1])/C+np.matmul(H.T,H)),H.T),trainY_temp)
        else:
            beta=np.matmul(H.T,np.matmul(np.linalg.inv(np.identity(H.shape[0])/C+np.matmul(H,H.T)),trainY_temp))
            
    
    else:
        print('Unsupport mode, only Regularized least square and Moore-Penrose pseudoinverse are allowed. ')
    

    trainY_temp=np.matmul(H,beta)
    Y_temp=np.zeros((Nsample,1))

    # decode the target
    for i in range(0,Nsample):
        idx=np.argmax(trainY_temp[i,:])
        Y_temp[i]=U_trainY[idx]

    Bias_test=np.matlib.repmat(Bias,np.size(testY),1)
    H_test=np.matmul(testX,Weight)+Bias_test


    if option.ActivationFunction.lower()=='sig' or option.ActivationFunction.lower()=='sigmoid':
        ####### Sigmoid 
        if option.Scale:
            if option.Scalemode==1:
                H_test=H_test*k+b
            elif option.Scalemode==2:
                nSamtest=H_test.shape[0]
                kt=np.matlib.repmat(k,nSamtest,1)
                bt=np.matlib.repmat(b,nSamtest,1)
                H_test=H_test*kt+bt

        H_test=1 / (1 + np.exp(-H_test))

    elif option.ActivationFunction.lower()=='sin' or option.ActivationFunction.lower()=='sine':

        if option.Scale:
            if option.Scalemode==1:
                H_test=H_test*k+b
            elif option.Scalemode==2:
                nSamtest=H_test.shape[0]
                kt=np.matlib.repmat(k,nSamtest,1)
                bt=np.matlib.repmat(b,nSamtest,1)
                H_test=H_test*kt+bt

        H_test=np.sin(H_test)

    elif option.ActivationFunction.lower()=='hardlim':

        H_test=hardlim(H_test)

    elif option.ActivationFunction.lower()=='tribas':

        if option.Scale:
            if option.Scalemode==1:
                H_test=H_test*k+b
            elif option.Scalemode==2:
                nSamtest=H_test.shape[0]
                kt=np.matlib.repmat(k,nSamtest,1)
                bt=np.matlib.repmat(b,nSamtest,1)
                H_test=H_test*kt+bt

        H_test=tribas(H_test)

    elif option.ActivationFunction.lower()== 'radbas':

        if option.Scale:
            if option.Scalemode==1:
                H_test=H_test*k+b
            elif option.Scalemode==2:
                nSamtest=H_test.shape[0]
                kt=np.matlib.repmat(k,nSamtest,1)
                bt=np.matlib.repmat(b,nSamtest,1)
                H_test=H_test*kt+bt

        H_test = radbas(H_test)

    elif option.ActivationFunction.lower()== 'sign':

        H_test=np.sign(H_test)
    
    if option.bias:
        H_test=np.concatenate([H_test,np.ones((np.size(testY),1))],axis=1)

    if option.link:
        if option.Scalemode==1:
            testX_temp=testX*k+b
            H_test=np.concatenate([H_test,testX_temp],axis=1)

        elif option.Scalemode==2:
            nSamtest=H_test.shape[0]
            kt=np.matlib.repmat(ktr,nSamtest,1)
            bt=np.matlib.repmat(btr,nSamtest,1)
            testX_temp=testX*kt+bt
            H_test=np.concatenate([H_test,testX_temp],axis=1)

        else:
            H_test=np.concatenate([H_test,testX],axis=1)


    H_test[np.isnan(H_test)]=0
    testY_temp=np.matmul(H_test,beta)
    Yt_temp=np.zeros((np.size(testY),1))

    for i in range(0,np.size(testY)):
        idx=np.argmax(testY_temp[i,:])
        Yt_temp[i]=U_trainY[idx]


    train_num=0
    for i in range (0,Y_temp.shape[0]):
        if Y_temp[i]==trainY[i]:
            train_num+=1

    train_accuracy=train_num/Nsample

    test_num=0
    for i in range (0,Yt_temp.shape[0]):
        if Yt_temp[i]==testY[i]:
            test_num+=1
    test_accuracy=test_num/np.size(testY)

    return train_accuracy,test_accuracy
    #np.set_printoptions(threshold=sys.maxsize)            
    #print(trainY_temp.size)
  

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
