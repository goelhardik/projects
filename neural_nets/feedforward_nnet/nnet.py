
from mlpython.learners.generic import Learner
import numpy as np
import math

class NeuralNetwork(Learner):
    """
    Neural network for classification.
 
    Option ``lr`` is the learning rate.
 
    Option ``dc`` is the decrease constante for the learning rate.
 
    Option ``sizes`` is the list of hidden layer sizes.
 
    Option ``L2`` is the L2 regularization weight (weight decay).
 
    Option ``L1`` is the L1 regularization weight (weight decay).
 
    Option ``seed`` is the seed of the random number generator.
 
    Option ``tanh`` is a boolean indicating whether to use the
    hyperbolic tangent activation function (True) instead of the
    sigmoid activation function (True).
 
    Option ``n_epochs`` number of training epochs.
 
    **Required metadata:**
 
    * ``'input_size'``: Size of the input.
    * ``'targets'``: Set of possible targets.
 
    """
    
    def __init__(self,
                 lr=0.001,
                 dc=1e-10,
                 sizes=[200,100,50],
                 L2=0.001,
                 L1=0,
                 seed=1234,
                 tanh=True,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.sizes=sizes
        self.L2=L2
        self.L1=L1
        self.seed=seed
        self.tanh=tanh
        self.n_epochs=n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0 

    def initialize(self,input_size,n_classes):
        """
        This method allocates memory for the fprop/bprop computations (DONE)
        and initializes the parameters of the neural network (TODO)
        """

        self.n_classes = n_classes
        self.input_size = input_size

        n_hidden_layers = len(self.sizes)
        #############################################################################
        # Allocate space for the hidden and output layers, as well as the gradients #
        #############################################################################
        self.hs = []
        self.grad_hs = []
        for h in range(n_hidden_layers):         
            self.hs += [np.zeros((self.sizes[h],))]       # hidden layer
            self.grad_hs += [np.zeros((self.sizes[h],))]  # ... and gradient
        self.hs += [np.zeros((self.n_classes,))]       # output layer
        self.grad_hs += [np.zeros((self.n_classes,))]  # ... and gradient
        
        ##################################################################
        # Allocate space for the neural network parameters and gradients #
        ##################################################################
        self.weights = [np.zeros((self.input_size,self.sizes[0]))]       # input to 1st hidden layer weights
        self.grad_weights = [np.zeros((self.input_size,self.sizes[0]))]  # ... and gradient

        self.biases = [np.zeros((self.sizes[0]))]                        # 1st hidden layer biases
        self.grad_biases = [np.zeros((self.sizes[0]))]                   # ... and gradient

        for h in range(1,n_hidden_layers):
            self.weights += [np.zeros((self.sizes[h-1],self.sizes[h]))]        # h-1 to h hidden layer weights
            self.grad_weights += [np.zeros((self.sizes[h-1],self.sizes[h]))]   # ... and gradient

            self.biases += [np.zeros((self.sizes[h]))]                   # hth hidden layer biases
            self.grad_biases += [np.zeros((self.sizes[h]))]              # ... and gradient

        self.weights += [np.zeros((self.sizes[-1],self.n_classes))]      # last hidden to output layer weights
        self.grad_weights += [np.zeros((self.sizes[-1],self.n_classes))] # ... and gradient

        self.biases += [np.zeros((self.n_classes))]                   # output layer biases
        self.grad_biases += [np.zeros((self.n_classes))]              # ... and gradient
            
        #########################
        # Initialize parameters #
        #########################

        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator

        ## PUT CODE HERE ##
        for h in range(n_hidden_layers + 1):    # into hidden layers + last hidden to output layer weights
            b = float(math.sqrt(6)) / float(math.sqrt(len(self.weights[h][0]) + len(self.weights[h])))  # use b = sqrt(6) / sqrt(H_k + H_k-1)
            self.weights[h] = self.rng.uniform(low = -b, high = b, size = (len(self.weights[h]), len(self.weights[h][0])))  # initialize weights to random values from [-b, b]

        
        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate

    def forget(self):
        """
        Resets the neural network to its original state (DONE)
        """
        self.initialize(self.input_size,self.targets)
        self.epoch = 0
        
    def train(self,trainset):
        """
        Trains the neural network until it reaches a total number of
        training epochs of ``self.n_epochs`` since it was
        initialize. (DONE)

        Field ``self.epoch`` keeps track of the number of training
        epochs since initialization, so training continues until 
        ``self.epoch == self.n_epochs``.
        
        If ``self.epoch == 0``, first initialize the model.
        """

        if self.epoch == 0:
            input_size = trainset.metadata['input_size']
            n_classes = len(trainset.metadata['targets'])
            self.initialize(input_size,n_classes)
            
        for it in range(self.epoch,self.n_epochs):
            count = 0
            m = 0
            for input,target in trainset:
                self.fprop(input,target)
                self.bprop(input,target)
                self.update()
                ind = np.argmax(self.hs[len(self.sizes)])
                m += 1
                if (ind == target):
                    count += 1
        self.epoch = self.n_epochs
        
    def sigmoid(self, a):
        """
        Calculate the sigmoid of the given vector
        """
        return np.divide(np.ones(len(a)), np.add(np.ones(len(a)), np.exp(-a)))

    def tanh_calc(self, a):
        """
        Calculate the tanh of the given vector
        """
        expa = np.exp(a)
        expan = np.exp(-a)
        return np.divide(np.subtract(expa, expan), np.add(expa, expan))

    def sigmoid_activation(self, a):
        """
        Calculate and return the sigmoid activation of the given input vector a
        """
        if (self.tanh == False):
            # sigmoid function
            return self.sigmoid(a) 
        else:
            return self.tanh_calc(a) 

    def softmax_activation(self, a):
        """
        Calculate and return the softmax activation of the given input vector a
        """
        #print(a)
        expa = np.exp(a)
        den = np.sum(expa)
        #print(den)
        den = np.array([den for i in range(len(a))])
        return np.divide(expa, den)

    def fprop(self,input,target):
        """
        Forward propagation: 
        - fills the hidden layers and output layer in self.hs
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        ## PUT CODE HERE ##
        n_hidden_layers = len(self.sizes)

        a = np.add(self.biases[0], np.dot(self.weights[0].T, input)) # calculate pre-activation of 1st hidden layer
        self.hs[0] = self.sigmoid_activation(a)  # activation of 1st hidden layer

        for h in range(1, n_hidden_layers): # all the hidden layers
            a = np.add(self.biases[h], np.dot(self.weights[h].T, self.hs[h - 1])) # calculate pre-activation of layer h
            self.hs[h] = self.sigmoid_activation(a) # activation of hth hidden layer

        a = np.add(self.biases[n_hidden_layers], np.dot(self.weights[n_hidden_layers].T, self.hs[n_hidden_layers - 1])) # calculate pre-activation of output layer
        self.hs[n_hidden_layers] = self.softmax_activation(a)  # activation of 1st hidden layer

        return self.training_loss(self.hs[n_hidden_layers], target)

        
    def training_loss(self,output,target):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given output vector (probabilities of each class) and target (class ID)
        """

        ## PUT CODE HERE ##
        loss = -np.log(output[target])
        #print(loss)
        return loss

    def sigmoid_grad(self, a):
        """
        Calculates the gradient of hidden layer (before activation)
        """
        if (self.tanh == False):
            # calculate sigmoid gradient g(a)(1 - g(a))
            return np.multiply(a, np.subtract(np.ones(len(a)), a))
        else:
            # calculate tanh gradient
            return np.subtract(np.ones(len(a)), np.multiply(a, a))

    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the hidden layers and output layer gradients in self.grad_hs
        - fills in the neural network gradients of weights and biases in self.grad_weights and self.grad_biases
        - returns nothing
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """
        n_hidden_layers = len(self.sizes)

        self.theta = np.array([]) # initialize theta
        for h in range(n_hidden_layers + 1):
            self.theta = np.concatenate((self.theta, self.weights[h].flatten('F')))
            self.theta = np.concatenate((self.theta, self.biases[h]))

        # output layer gradient (pre-activation) -(e(y) - f(x))
        ey = np.zeros(len(self.hs[n_hidden_layers]))
        ey[target] = 1
        self.grad_hs[n_hidden_layers] = np.subtract(self.hs[n_hidden_layers], ey)

        # hidden layer gradients
        for k in range(n_hidden_layers, 0, -1):
            # gradient of weights
            self.grad_weights[k] = np.outer(self.hs[k - 1], self.grad_hs[k])
            # gradient of biases
            self.grad_biases[k] = self.grad_hs[k]
            # gradient of hidden layer below
            hid = np.dot(self.weights[k], self.grad_hs[k])
            # gradient of hidden layer below (before activation)
            self.grad_hs[k - 1] = np.multiply(hid, self.sigmoid_grad(self.hs[k - 1]))

        # first hidden layer gradients
        self.grad_weights[0] = np.outer(input, self.grad_hs[0])
        self.grad_biases[0] = self.grad_hs[0]

        ## PUT CODE HERE ##

        # raise NotImplementedError()

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the neural network parameters self.weights and self.biases,
          using the gradients in self.grad_weights and self.grad_biases
        """

        ## PUT CODE HERE ##
        n_hidden_layers = len(self.sizes)

        for h in range(n_hidden_layers + 1):
            # update weights
            alpha = np.array([[self.lr for i in range(len(self.weights[h][0]))] for j in range(len(self.weights[h]))])
            self.weights[h] = np.subtract(self.weights[h], np.multiply(alpha, self.grad_weights[h]))
            # update biases
            alpha = np.array([self.lr for i in range(len(self.biases[h]))])
            self.biases[h] = np.subtract(self.biases[h], np.multiply(alpha, self.grad_biases[h]))

        # raise NotImplementedError()
           
    def use(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a Numpy 2D array of size
          len(dataset) by (nb of classes + 1)
        - the ith row of the array contains the outputs for the ith example
        - the outputs for each example should contain
          the predicted class (first element) and the
          output probabilities for each class (following elements)
        Argument ``dataset`` is an MLProblem object.
        """
 
        m = 0
        n_hidden_layers = len(self.sizes)
        outputs = np.zeros((len(dataset),self.n_classes+1))
        for input, output in dataset:
            self.fprop(input, output)
            ind = np.argmax(self.hs[n_hidden_layers])
            outputs[m][0] = ind
            outputs[m][1:] = list(self.hs[n_hidden_layers])
            m += 1

        ## PUT CODE HERE ##

        # raise NotImplementedError()
            
        return outputs
        
    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of 
        those outputs for ``dataset``:
        - the errors should be a Numpy 2D array of size
          len(dataset) by 2
        - the ith row of the array contains the errors for the ith example
        - the errors for each example should contain 
          the 0/1 classification error (first element) and the 
          regularized negative log-likelihood (second element)
        Argument ``dataset`` is an MLProblem object.
        """
          
        outputs = self.use(dataset)
        errors = np.zeros((len(dataset),2))
        
        ## PUT CODE HERE ##
        count = 0
        m = 0
        for input, output in dataset:
            if (outputs[m][0] == output):
                count += 1
                errors[m][0] = 0
            else:
                errors[m][0] = 1
                errors[m][1] = self.training_loss(outputs[m][1:], output) 
            m += 1

        #raise NotImplementedError()
            
        print("Test set accuracy = " + str(count) + "/" + str(m))
        return outputs, errors
 
    def verify_gradients(self):
        """
        Verifies the implementation of the fprop and bprop methods
        using a comparison with a finite difference approximation of
        the gradients.
        """
        
        print 'WARNING: calling verify_gradients reinitializes the learner'
  
        rng = np.random.mtrand.RandomState(1234)
  
        self.seed = 1234
        self.sizes = [4,5]
        self.initialize(20,3)
        example = (rng.rand(20)<0.5,2)
        input,target = example
        epsilon=1e-6
        self.lr = 0.1
        self.decrease_constant = 0
  
        self.fprop(input,target)
        self.bprop(input,target) # compute gradients

        import copy
        emp_grad_weights = copy.deepcopy(self.weights)
  
        for h in range(len(self.weights)):
            for i in range(self.weights[h].shape[0]):
                for j in range(self.weights[h].shape[1]):
                    self.weights[h][i,j] += epsilon
                    a = self.fprop(input,target)
                    self.weights[h][i,j] -= epsilon
                    
                    self.weights[h][i,j] -= epsilon
                    b = self.fprop(input,target)
                    self.weights[h][i,j] += epsilon
                    
                    emp_grad_weights[h][i,j] = (a-b)/(2.*epsilon)


        print 'grad_weights[0] diff.:',np.sum(np.abs(self.grad_weights[0].ravel()-emp_grad_weights[0].ravel()))/self.weights[0].ravel().shape[0]
        print 'grad_weights[1] diff.:',np.sum(np.abs(self.grad_weights[1].ravel()-emp_grad_weights[1].ravel()))/self.weights[1].ravel().shape[0]
        print 'grad_weights[2] diff.:',np.sum(np.abs(self.grad_weights[2].ravel()-emp_grad_weights[2].ravel()))/self.weights[2].ravel().shape[0]
  
        emp_grad_biases = copy.deepcopy(self.biases)    
        for h in range(len(self.biases)):
            for i in range(self.biases[h].shape[0]):
                self.biases[h][i] += epsilon
                a = self.fprop(input,target)
                self.biases[h][i] -= epsilon
                
                self.biases[h][i] -= epsilon
                b = self.fprop(input,target)
                self.biases[h][i] += epsilon
                
                emp_grad_biases[h][i] = (a-b)/(2.*epsilon)

        print 'grad_biases[0] diff.:',np.sum(np.abs(self.grad_biases[0].ravel()-emp_grad_biases[0].ravel()))/self.biases[0].ravel().shape[0]
        print 'grad_biases[1] diff.:',np.sum(np.abs(self.grad_biases[1].ravel()-emp_grad_biases[1].ravel()))/self.biases[1].ravel().shape[0]
        print 'grad_biases[2] diff.:',np.sum(np.abs(self.grad_biases[2].ravel()-emp_grad_biases[2].ravel()))/self.biases[2].ravel().shape[0]

