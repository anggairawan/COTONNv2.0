import tensorflow as tf
import os
import NeuralNetworkManager
import DataSet

# Disable session clear debug log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# The MLP class as presented in this file is a bare bone object oriented approach to creating an MLP
# for the purpose of capturing the behaviour of a static controller (table). 
class MLP:   
        def __init__(self):
            self.debug_mode = False
      
            # Network parameters
            self.layers = []                    # list of number of neuron on each layer
            self.num_layers = 0                 # number of layers
            self.dropout_rate = 0.0             # 
            self.hidden_activation_function = None
            self.output_activation_function = None
            
            # Neural network general variables (MLP)            
            self.weights = []
            self.biases = []
            
            self.predictor = None
            
            # Neural network training variables
            self.loss_function = None          # loss function to be minimized
            self.train_function = None         # optimizer
            
            # Tensorflow specific
            tf.reset_default_graph()
            self.session = tf.Session()

            self.task_type = None

            
        # Setters
        def setDebugMode(self, value): self.debug_mode = value
        def setNeurons(self, layers):
            self.num_layers = len(layers)
            self.layers = layers
        def setDropoutRate(self, value): self.dropout_rate = value
        def setActivationFunctionHidden(self, value): self.hidden_activation_function = value
        def setActivationFunctionOutput(self, value): self.output_activation_function = value
        
        def setTaskTypes(self, value): self.task_type = value
      
        # Getters
        def getNumLayers(self): return self.num_layers
        def getLayers(self): return self.layers
        def getDropoutRate(self): return self.dropout_rate
        def getActivationFunctionHidden(self): return self.hidden_activation_function
        def getActivationFunctionOutput(self): return self.output_activation_function
        
        # tensorflow session functions
        def getSession(self): return self.session
        def runInSession(self, fetch, x, y, dropout_rate): return self.session.run(fetch, {self.x: x, self.y: y, self.dropout: dropout_rate})
    
        # Initialize network function which intializes an initial network with random weights and biases
        def initializeNetwork(self):
            self.x = tf.placeholder(tf.float32, [None, self.layers[0]], name='x-data')
            self.y = tf.placeholder(tf.float32, [None, self.layers[-1]], name='y-data')
            self.dropout = tf.placeholder(tf.float32, name="dropout_rate")

            layer = tf.layers.dense(inputs=self.x, units=self.layers[1], activation=self.activationFunctionHidden(), name="layer_0", kernel_initializer=tf.contrib.layers.xavier_initializer())
            layer = tf.layers.dropout(inputs=layer, rate=self.dropout, name="dropout_0")

            for i in range(1, self.num_layers - 1):
                if(i == self.num_layers - 2):
                    layer = tf.layers.dense(inputs=layer, units=self.layers[i+1], activation=self.activationFunctionOutput(), name="layer_" + str(i), kernel_initializer=tf.contrib.layers.xavier_initializer())
                else:
                    layer = tf.layers.dense(inputs=layer, units=self.layers[i+1], activation=self.activationFunctionHidden(), name="layer_" + str(i), kernel_initializer=tf.contrib.layers.xavier_initializer())
                    layer = tf.layers.dropout(inputs=layer, rate=self.dropout, name="dropout_" + str(i))

                #if(i != self.num_layers - 2):
                    #layer = tf.layers.dropout(inputs=layer, rate=self.dropout, name="dropout_" + str(i))

            # prediction output from activation function
            self.predictor = layer

            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        
        
        # Activation function
        def activationFunctionOutput(self):
            if self.output_activation_function == NeuralNetworkManager.NNActivationFunction.Sigmoid:
                return tf.sigmoid
            if self.output_activation_function == NeuralNetworkManager.NNActivationFunction.Relu:
                return tf.nn.relu
            if self.output_activation_function == NeuralNetworkManager.NNActivationFunction.tanh:
                return tf.tanh
            if self.output_activation_function == NeuralNetworkManager.NNActivationFunction.Softmax:
                return tf.nn.softmax
            if self.output_activation_function == NeuralNetworkManager.NNActivationFunction.Identity:
                return tf.identity
            else: 
                return tf.sigmoid

         # Activation function
        def activationFunctionHidden(self):
            if self.hidden_activation_function == NeuralNetworkManager.NNActivationFunction.Sigmoid:
                return tf.sigmoid
            if self.hidden_activation_function == NeuralNetworkManager.NNActivationFunction.Relu:
                return tf.nn.relu
            if self.hidden_activation_function == NeuralNetworkManager.NNActivationFunction.tanh:
                return tf.tanh
            if self.hidden_activation_function == NeuralNetworkManager.NNActivationFunction.Softmax:
                return tf.nn.softmax
            else: 
                return tf.sigmoid
 
        # Initialize loss function
        def initializeLossFunction(self):
            if((self.task_type == DataSet.EncodeTypes.Boolean) or (self.task_type == DataSet.EncodeTypes.Classification)):
                self.loss_function = tf.losses.log_loss(self.y, self.predictor)
            else:
                self.loss_function = tf.losses.mean_squared_error(self.y, self.predictor)
            tf.summary.scalar("loss", self.loss_function)
            
            return self.loss_function
        
        
        # Intialize training function
        def initializeTrainFunction(self, function, learning_rate):
            with tf.name_scope('optimizer'):
                  if(function == NeuralNetworkManager.NNOptimizer.Gradient_Descent):
                      self.train_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: Gradient Descent with learning rate: " + str(learning_rate))
                  elif(function == NeuralNetworkManager.NNOptimizer.Adadelta):
                      self.train_function = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: AdaDelta with learning rate: " + str(learning_rate))
                  elif(function == NeuralNetworkManager.NNOptimizer.Adagrad):
                      self.train_function = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: AdaGrad with learning rate: " + str(learning_rate))
                  elif(function == NeuralNetworkManager.NNOptimizer.Adam):
                      self.train_function = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: Adam with learning rate: " + str(learning_rate))
                  else:
                      self.train_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: Gradient Descent with learning rate: " + str(learning_rate))
           
                  self.session.run(tf.global_variables_initializer())
            return self.train_function
        
        
        # Training step with a batch
        def trainStep(self, batch, merged_summary):
            train, summary, loss = self.session.run([self.train_function, merged_summary, self.loss_function], {self.x: batch[0], self.y: batch[1], self.dropout: self.dropout_rate})
            return loss, summary
            
        
        # Estimator function which estimates the desired outcome based on an input
        def estimate(self, x):
            return self.session.run(self.predictor, {self.x: x, self.dropout: 0.0})
            
        
        # Calculate network data size assuming default activation function
        def calculateDataSize(self):
            floats = 0
            for i in range(0, self.num_layers - 1):
                # weights
                a = self.layers[i]
                b = self.layers[i+1]
                
                # weights
                floats += a*b 
                floats += 2; # weight matrix dimensions
                
                floats += b # biases
                floats += 1; # bias vector dimension
                
            return floats*4
        
        
        # Save
        def save(self, filename):
            self.saver.save(self.session, filename)

            
        # Close function which closes tensorflow session
        def close(self):
            self.session.close()
            
            
            
