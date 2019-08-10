def RVFL_train_val(trainX,trainY,testX,testY,option):
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
    


   return train_accuracy,test_accuracy


