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
class option:
 def __init__(self, N, bias, link, ActivationFunction, seed, mode, RandomType, Scale, Scalemode):
        self.N=N
        self.bias=bias
        self.link=link
        self.ActivationFunction=ActivationFunction
        self.seed=seed
        self.mode=mode
        self.RandomType=RandomType
        self.Scale=Scale
        self.Scalemode=Scalemode