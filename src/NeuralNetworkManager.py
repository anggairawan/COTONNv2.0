#import tensorflow as tf
from BinaryEncoderDecoder import BinaryEncoderDecoder
from Utilities import Utilities
from MLP import MLP
from enum import Enum
from DataSet import EncodeTypes
from DataSet import Ordering
from StaticController import StaticController

import tensorflow as tf
import math
import numpy as np
import signal
import time

import matplotlib.pyplot as plt
import random


class NNTypes(Enum):
      MLP = 1
      RBF = 2
      CMLP = 3
    
    
class NNOptimizer(Enum):
      Gradient_Descent = 1
      Adagrad = 2
      Adadelta = 3
      Adam = 4
      Ftrl = 5
      Momentum = 6
      RMSProp = 7    
    
    
class NNActivationFunction(Enum):
      Sigmoid = 1
      Relu = 2
      tanh = 3
      Softmax = 4
      Identity = 5      
 
# Class which will handle all the work done on neural networks and will contain all the functions which
# are called in tensorflow in order to generate neural networks from the controllers.
class NeuralNetworkManager:
    def __init__(self):
        self.type = None
        self.nn = MLP()
        self.training_method = None
        self.hidden_activation_function = None
        self.output_activation_function = None
        self.dropout_rate = 0.0
        
        self.training = True
        self.learning_rate = 0.1
        self.fitness_threshold = 0.75
        self.epoch_threshold = -1
        self.batch_size = 100
        self.shuffle_rate = 2500
        self.display_step = 1000
        
        self.epoch = 0
        
        self.layers = []
        
        self.data_set = None
        
        self.bed = BinaryEncoderDecoder()
        self.utils = Utilities()
        self.controller = StaticController()
        
        self.debug_mode = False
        
        # Plotting variables
        self.losses = []
        self.fitnesses = []
        self.iterations = []
        
        self.save_location = './nn/log/'

        self.encode = None

        self.non_det = False
        
    # Getters and setters
    def getType(self): return self.type
    def getTrainingMethod(self): return self.training_method
    def getActivationFunctionHidden(self): return self.hidden_activation_function
    def getActivationFunctionOutput(self): return self.output_activation_function
    def getLearningRate(self): return self.learning_rate
    def getFitnessThreshold(self): return self.fitness_threshold
    def getBatchSize(self): return self.batch_size
    def getDisplayStep(self): return self.display_step
    def getEpoch(self): return self.epoch
    def getEpochThreshold(self): return self.epoch_threshold
    def getDropoutRate(self): return self.dropout_rate
    def getShuffleRate(self): return self.shuffle_rate
    def getSaveLocation(self): return self.save_location
    
    def setType(self, value): self.type = value
    def setTrainingMethod(self, optimizer): self.training_method = optimizer
    def setActivationFunctionHidden(self, activation_function): self.hidden_activation_function = activation_function
    def setActivationFunctionOutput(self, activation_function): self.output_activation_function = activation_function
    def setLearningRate(self, value): self.learning_rate = value
    def setFitnessThreshold(self, value): self.fitness_threshold = value
    def setBatchSize(self, value): self.batch_size = value
    def setDisplayStep(self, value): self.display_step = value
    def setEpochThreshold(self, value): self.epoch_threshold = value
    def setDropoutRate(self, value): self.dropout_rate = value
    def setShuffleRate(self, value): self.shuffle_rate = value
    def setSaveLocation(self, value): self.save_location = value
    def setDebugMode(self, value): self.debug_mode = value
    def setDataSet(self, data_set): self.data_set = data_set
    
    def setEncodeTypes(self, value): self.encode = value
    
    # Hidden layer generation functions
    # Linearly increase/decrease neurons per hidden layer based on the input and ouput neurons
    def linearHiddenLayers(self, num_hidden_layers):
        self.layers = []
        
        x_dim = self.data_set.getXDim()
        y_dim = self.data_set.getYDim()
        
        a = (y_dim - x_dim)/(num_hidden_layers + 1)
        
        self.layers.append(x_dim)
        for i in range(1, num_hidden_layers + 1):
            self.layers.append(round(x_dim + a*i))
        self.layers.append(y_dim)
        
        return self.layers
    
    # Rectangular hidden layer
    def rectangularHiddenLayers(self, width, height):
        self.layers = []
        
        self.layers.append(self.data_set.getXDim())
        for i in range(width):
            self.layers.append(height)
        self.layers.append(self.data_set.getYDim())
  
      
    #Customize layer sturcture
    def customHiddenLayers(self, layer):
        self.layers = []
        
        x_dim = self.data_set.getXDim()
        y_dim = self.data_set.getYDim()
        
        self.layers.append(x_dim)
        for i in range(1, len(layer)+1):
            self.layers.append(layer[i-1])
        self.layers.append(y_dim)
        
        return self.layers


     # Initialize neural network
    def initializeNeuralNetwork(self):
        if(self.debug_mode):
            print("\nNeural network initialization:")

        if(self.type == NNTypes.MLP):
            self.nn = MLP()
            self.nn.setDebugMode(self.debug_mode)
            if(self.debug_mode):
                print("Neural network type: MLP")
            
        # Initialize network and loss function
        self.nn.setNeurons(self.layers)
        self.nn.setDropoutRate(self.dropout_rate)
        self.nn.setActivationFunctionHidden(self.hidden_activation_function)
        self.nn.setActivationFunctionOutput(self.output_activation_function)
        self.nn.setTaskTypes(self.encode)
        self.nn.initializeNetwork()
        
        # Print neural network status
        if(self.debug_mode):
            print("Generated network neuron topology: " + str(self.layers) + " with dropout rate: " + str(self.nn.getDropoutRate()))
        
        
    # Initialize training function
    def initializeTraining(self, learning_rate, fitness_threshold, batch_size, display_step, epoch_threshold = -1, shuffle_rate = 10000):      
        self.learning_rate = learning_rate
        self.fitness_threshold = fitness_threshold
        self.batch_size = batch_size
        self.display_step = display_step
        self.epoch_threshold = epoch_threshold
        self.shuffle_rate = shuffle_rate
        
        self.nn.initializeLossFunction()
        self.nn.initializeTrainFunction(self.training_method, self.learning_rate)
        
        
    # Initialize fitness function
    def initializeFitnessFunction(self):
        with tf.name_scope("fitness"):
            eta = self.data_set.getYEta()
            size = self.data_set.getSize()
            
            lower_bound = tf.subtract(self.nn.y, eta)
            upper_bound = tf.add(self.nn.y, eta)
            
            is_fit = tf.logical_and(tf.greater_equal(self.nn.predictor, lower_bound), tf.less(self.nn.predictor, upper_bound))
            non_zero = tf.to_float(tf.count_nonzero(tf.reduce_min(tf.cast(is_fit, tf.int8), 1)))
            self.fitness = non_zero/size

            tf.summary.scalar("fitness", self.fitness)
        
        
    # General initialization function to call all functions
    def initialize(self, learning_rate, fitness_threshold, batch_size, display_step, epoch_threshold = -1, shuffle_rate = 10000):
        self.initializeNeuralNetwork()
        self.initializeFitnessFunction()
        self.initializeTraining(learning_rate, fitness_threshold, batch_size, display_step, epoch_threshold, shuffle_rate)

        self.train_writer = tf.summary.FileWriter(self.save_location, self.nn.session.graph)
        
        
    # Check a state against the dataset and nn by using its id in the dataset
    def checkByIndex(self, index, out):
        x = self.data_set.x[index]
        estimation = self.nn.estimate([x])[0]
        y = self.data_set.getY(index)
        
        y_eta = self.data_set.getYEta()
        equal = True
        for i in range(self.data_set.getYDim()):
            if(not((y[i] - y_eta[i]) <= estimation[i] and (y[i] + y_eta[i]) > estimation[i])):
                equal = False
        
        if(out):
            print("u: " + str(y) + " u_: " + str(np.round(estimation,2)) + " within etas: " + str(equal))
            
        return equal
    
    
    # Check fitness of the neural network for a specific dataset and return wrong states
    # as of right now it assumes a binary encoding of the dataset
    def checkFitness(self, data_set):
        self.data_set = data_set
        
        size = self.data_set.getSize()
        fit = size
        
        wrong = []
        
        x, y = self.data_set.x, self.data_set.y
        y_eta = self.data_set.getYEta()
        y_dim = self.data_set.getYDim()
        
        estimation = self.nn.estimate(self.data_set.x)
        
        for i in range(size):
            equal = True
            for j in range(y_dim):
                if(not((y[i][j] - y_eta[j]) <= estimation[i][j] and (y[i][j] + y_eta[j]) > estimation[i][j]) and equal):
                    wrong.append(self.bed.baton(x[i]))
                    fit -= 1
                    equal = False
        fitness = fit/size*100
        return fitness, wrong

    # Check fitness of the neural network for a specific dataset and return wrong states
    # as of right now it assumes a binary encoding of the dataset
    def checkFitnessAllGridPoint(self, data_set):
        print("\nCalculating fitness and storing wrong states")
        self.data_set = data_set
        
        size = self.data_set.getSize()
        fit = size
        
        wrong = []
        
        x, y = self.data_set.x, self.data_set.y
        y_eta = self.data_set.getYEta()
        y_dim = self.data_set.getYDim()
        
        estimation = self.nn.estimate(self.data_set.x)
    
        # for binary
        labels = np.array(y)
        # predictions = np.array(np.round(estimation))
        # invalid_flag = (labels[:,-1] == 0)
        
        if(not self.non_det):
            # check the control input prediction and invalid flag based on controller type (D or ND)
            if(self.controller.con_det):
                predictions = np.array(np.round(estimation))
                # array of boolean with output flag equal to invalid control
                invalid_flag = (labels[:,-1] == 0)
                # check the control input prediction
                same_inputs = (predictions[:,:-1] == labels[:,:-1]).all(axis = 1)
                # check if the prediction and true output is equal
                same_flag = (predictions[:,-1] == labels[:,-1])
            else:
                predictions = np.array(estimation)
                invalid_flag = (labels[:,-1] == 1)
                predictions_soft = np.zeros_like(predictions)
                predictions_soft[np.arange(len(predictions)), predictions.argmax(1)] = 1
                same_inputs = (labels[np.arange(len(labels)), predictions_soft.argmax(1)])
                # check if the prediction flag and true output flag is equal 
                car_u = self.controller.input_total_gp
                flag_one = np.logical_and(np.argmax(predictions_soft, 1) == (car_u), labels[:,-1] == 1)
                flag_zero = np.logical_and(np.argmax(predictions_soft, 1) != (car_u), labels[:,-1] == 0)
                same_flag = np.logical_or(flag_one, flag_zero)
                
            # same_flag = (predictions[:,-1] == labels[:,-1])
            early_check = np.logical_and(invalid_flag, same_flag)

            # if it is same flag and same input, we have the performance here
            same_all = np.logical_and(same_flag, same_inputs)
            # we need to or to compensate incorrect prediction but does not matter because it is not winning domain
            logic_all = np.logical_or(early_check, same_all)
            # average the value to get fitness in the range of [0,1]
            fitness = (np.mean(logic_all))

            # valid inputs but wrong prediction           
            wrong_idx = np.where(np.logical_and((logic_all == 0), np.logical_not(invalid_flag)))
        else:
            predictions = np.array(np.round(estimation))
            fitness = (np.mean(predictions == labels))    
            wrong_idx = np.where((np.logical_not(predictions == labels)).any(axis=1))

        # get the index of wrong states
        states = np.array(self.data_set.x)
         
        if(self.data_set.encode == EncodeTypes.Classification):
            wrong = np.squeeze(states[wrong_idx])
        elif(self.data_set.encode == EncodeTypes.Boolean):
            if(self.data_set.order == Ordering.Original):
                wrong = states[wrong_idx]
                wrong_temp = list(map(self.bed.baton, wrong))
                
                wrong_x = list(map(self.controller.stox, wrong_temp))
                
                wrong = []
                for i in range(len(wrong_x)):
                    wrong.append([wrong_temp[i], wrong_x[i]])

            else:
                ss_dim = int(self.controller.state_space_dim)
                wrong_states = states[wrong_idx]
                if(wrong_states.size == 0):
                    return fitness, wrong
                
                # len_one_dim = int(len(wrong_states[0])/ss_dim)
                bit_dim = self.controller.bit_dim

                temp = []
                for i in range(ss_dim):
                    # bin_i = wrong_states[:,i*len_one_dim:(i+1)*len_one_dim]
                    bin_i = wrong_states[:,i*bit_dim[i]:(i+1)*bit_dim[i]]
                    array_i_add_dim = np.array(list(map(self.bed.baton, bin_i)))[:,None]
                    temp.append(array_i_add_dim)                
                
                wrong_temp = temp[0]
                for i in range(ss_dim-1):
                    wrong_temp = np.concatenate((wrong_temp, temp[i+1]), axis = 1) 

                wrong_x = list(map(self.controller.sstox, wrong_temp))
                wrong_s = list(map(self.controller.sstos, wrong_temp))

                wrong = []
                for i in range(len(wrong_s)):
                    wrong.append([wrong_s[i], wrong_x[i]])

        return fitness, wrong

    # Fitness modification for including as wel the non-winning domain to the NN
    # the last bit as the valid flag change the calculation quite a lot
    def allGridPointFitness(self, data_set):
        self.data_set = data_set
        
        size = self.data_set.getSize()
        fit = size
        
        wrong = []
        
        x, y = self.data_set.x, self.data_set.y
        y_eta = self.data_set.getYEta()
        y_dim = self.data_set.getYDim()
        
        estimation = self.nn.estimate(self.data_set.x)
        
        # calculate the fitness by using np array to optimize computation
        labels = np.array(y)
        # predictions = np.array(np.round(estimation))

        if(not self.non_det):
            # check the control input prediction and invalid flag based on controller type (D or ND)
            if(self.controller.con_det):
                # print('deterministic')
                predictions = np.array(np.round(estimation))
                # array of boolean with output flag equal to invalid control
                invalid_flag = (labels[:,-1] == 0)
                # check the control input prediction
                same_inputs = (predictions[:,:-1] == labels[:,:-1]).all(axis = 1)
                # check if the prediction flag and true output flag is equal 
                same_flag = (predictions[:,-1] == labels[:,-1])
            else:
                # print('determinizing')
                predictions = np.array(estimation)
                invalid_flag = (labels[:,-1] == 1)
                predictions_soft = np.zeros_like(predictions)
                predictions_soft[np.arange(len(predictions)), predictions.argmax(1)] = 1
                same_inputs = (labels[np.arange(len(labels)), predictions_soft.argmax(1)])
                # check if the prediction flag and true output flag is equal 
                car_u = self.controller.input_total_gp
                # print(car_u)
                # check the correctness of the output prediction
                # both for invalid and valid label 
                flag_one = np.logical_and(np.argmax(predictions_soft, 1) == (car_u), labels[:,-1] == 1)
                flag_zero = np.logical_and(np.argmax(predictions_soft, 1) != (car_u), labels[:,-1] == 0)
                
                same_flag = np.logical_or(flag_one, flag_zero)
                # print(np.mean(same_flag))
            
            # if it is invalid and it predicted right we have a flag that we do not have to check the rest of the bit
            early_check = np.logical_and(invalid_flag, same_flag)
            
            # if it is the same flag and same input, we have the performance here
            same_all = np.logical_and(same_flag, same_inputs)
            # we need to or to compensate incorrect prediction but does not matter because it is not winning domain
            logic_all = np.logical_or(early_check, same_all)
            # average the value to get fitness in the range of [0,1]
            fitness = (np.mean(logic_all))
        else:
            # print('non-deterministic')
            predictions = np.array(np.round(estimation))
            fitness = (np.mean(predictions == labels))

        # print(labels[:10])
        # print(predictions[:10])
        
        return fitness

    # Fitness modification for including as wel the non-winning domain to the NN
    # the last bit as the valid flag change the calculation quite a lot
    def allGridPointFitnessDeterminizing(self, data_set):
        self.data_set = data_set
        
        size = self.data_set.getSize()
        fit = size
        
        wrong = []
        
        x, y = self.data_set.x, self.data_set.y
        y_eta = self.data_set.getYEta()
        y_dim = self.data_set.getYDim()
        
        estimation = self.nn.estimate(self.data_set.x)
        
        # calculate the fitness by using np array to optimize computation
        labels = np.array(y)
        predictions = np.array(np.round(estimation))

        # array of boolean with output flag equal to invalid control
        invalid_flag = (labels[:,-1] == 1)
        # check if the prediction and true output is equal
        same_flag = (predictions[:,-1] == labels[:,-1])
        # if it is invalid and it predicted right we have a flag that we do not have to check the rest of the bit
        early_check = np.logical_and(invalid_flag, same_flag)
        
        # TO DO branch condition regression and classification
        # check the control input prediction
        # same_inputs = (predictions[:,:-1] == labels[:,:-1]).all(axis = 1)

        predictions_soft = np.zeros_like(predictions)
        predictions_soft[np.arange(len(predictions)), predictions.argmax(1)] = 1
        same_inputs = (labels[np.arange(len(labels)), predictions_soft.argmax(1)])
        
        # if it is same flag and same input, we have the performance here
        same_all = np.logical_and(same_flag, same_inputs)
        # we need to or to compensate incorrect prediction but does not matter because it is not winning domain
        logic_all = np.logical_or(early_check, same_all)
        # average the value to get fitness in the range of [0,1]
        fitness = (np.mean(logic_all))
        
        return fitness

    # Randomly check neural network against a dataset
    def randomCheck(self, data_set):
        self.data_set = data_set
        
        self.initializeFitnessFunction()        

        print("\nValidating:")
        for i in range(10):
            r = round(random.random()*(self.data_set.getSize()-1))
            self.checkByIndex(r, True)

        
    # Train network
    def train(self):
        self.clear()
        
        print("\nTraining (Ctrl+C to interrupt):")
        signal.signal(signal.SIGINT, self.interrupt)

        i, batch_index, loss, fit = 0,0,0,0.0
        old_epoch = 0
        stagnan = False

        self.merged_summary = tf.summary.merge_all()
        
        start_time = time.time()
        while self.training:
            batch = self.data_set.getBatch(self.batch_size, batch_index)
            loss, summary = self.nn.trainStep(batch, self.merged_summary)
                
            # if(i % self.shuffle_rate == 0 and i != 0): self.data_set.shuffle()
            # if(fit > 0.999):
                # fit = self.allGridPointFitness(self.data_set)
                # self.batch_size = 4096

            # if(i % self.display_step == 0 and i != 0):
            if((self.epoch % self.display_step == 0) and (old_epoch != self.epoch)):
                old_epoch = self.epoch

                # fit = self.nn.runInSession(self.fitness, self.data_set.x, self.data_set.y, 1.0)
                fit = self.allGridPointFitness(self.data_set)
                
                # self.addToLog(loss, fit, i)
                self.addToLog(loss, fit, self.epoch)
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.4f}".format(loss))) + "\tfit = " + str(float("{0:.4f}".format(fit))))
                self.train_writer.add_summary(summary, i)
                
            if(self.epoch >= self.epoch_threshold and self.epoch_threshold > 0):
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.4f}".format(loss))) + "\tfit = " + str(float("{0:.4f}".format(fit))))
                print("Finished training, epoch threshold reached")
                break
            
            if(fit >= self.fitness_threshold):
                print("Finished training")
                break

            if (len(self.fitnesses) > 40) and stagnan == False:
                if((self.fitnesses[-1] - 0.001) <= self.fitnesses[-40]):
                    print("Finished training, fitness did not improve after "+str(40*self.display_step)+" epoch")
                    stagnan = True
            
            if(math.isnan(loss)):
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.3f}".format(loss))) + "\tfit = " + str(float("{0:.3f}".format(fit))))
                print("Finished training, solution did not converge")
                break
            
            batch_index += self.batch_size
            if(batch_index >= self.data_set.getSize()): 
                batch_index = batch_index % self.data_set.getSize()
                self.data_set.shuffle()
                self.epoch += 1
            
            i += 1
 
        end_time = time.time()
        print("Time taken: " + self.utils.formatTime(end_time - start_time))
        
        
    # Interrupt handler to interrupt the training while in progress
    def interrupt(self, signal, frame):
        self.training = False
          
        
    # Plotting loss and fitness functions
    def plot(self):      
        plt.figure(1)
        plt.plot(self.iterations, self.losses, 'bo')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.grid()
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,y2+0.1))
        
        plt.figure(2)
        plt.plot(self.iterations, self.fitnesses, 'r-')
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.grid()
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,1))
        plt.show()
        

    # Add to log
    def addToLog(self, loss, fit, iteration):
        self.losses.append(loss)
        self.fitnesses.append(fit)
        self.iterations.append(iteration)
        
    # Get projected data size
    def getDataSize(self):
        size = self.nn.calculateDataSize()     
        print("Minimal neural network size of: " + self.utils.formatBytes(size))
        return size

    # Clear variables
    def clear(self):
        self.epoch = 0
        self.training = True

        self.fitnesses = []
        self.iterations = []
        self.losses = []
        
    # Save network
    def save(self, filename):
        print("\nSaving neural network")
        self.nn.save(filename)
    

    # Close session
    def close(self):
        self.nn.close()
        self.train_writer.close()
        
    # create loosing points to be plotted on matlab
    def createLoosingPoints(self, wrong_states):
        print("\nStoring loosing states for matlab simulation")
        # print("\nLoop", self.controller.state_total_gp)
        # for i in range(self.controller.state_total_gp):
            #if i not in self.controller.states or i in wrong:
            # if i not in self.controller.states:
                # loosing_states.append(i)
        total = set(range(self.controller.state_total_gp))
        winning = set(self.controller.states)
        loosing_states = total - winning 
        # print(len(total), len(winning), len(loosing_states))
        # print("\nConvert to x")
        loosing_states_x = list(map(self.controller.stox, loosing_states))
        return loosing_states_x


