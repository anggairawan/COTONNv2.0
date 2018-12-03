from BinaryEncoderDecoder import BinaryEncoderDecoder
from Utilities import Utilities

import random
import numpy as np

from enum import Enum

class EncodeTypes(Enum):
    Boolean = 0
    Classification = 1
    Regression = 2

class Ordering(Enum):
    Original = 0
    PerCoordinate = 1
    
# Dataset class which will contain the data for nn training and functions to read controllers into the specific
class DataSet:
    def __init__(self):
        self.x = []
        self.y = []
        
        self.x_bounds = [] # boundary elements of x 
        self.y_bounds = [] # boundary elements of y
        
        self.x_eta = [] # etas of x
        self.y_eta = [] # etas of y
        
        self.x_dim = 1 # dimension of elements in x
        self.y_dim = 1 # dimension of elements in y
        
        self.size = 0 # amount of elements in dataset

        self.var_order = True
        # self.encode = EncodeTypes.Boolean
        self.encode = EncodeTypes.Boolean
        self.order = Ordering.Original


    # Setter
    def setEncodeType(self, value): self.encode = value
    def setVariableOrder(self, value): self.order = value

    # Getters
    def getX(self, i): return self.x[i]
    def getY(self, i): return self.y[i]
    def getSize(self): return self.size
    
    def getPair(self, i): return [self.x[i], self.y[i]]
    def getPairByX(self, x):
        for i in range(self.size):
            if(self.x[i] == x):
                return [self.x[i], self.y[i]]
        return None
    
    def getLowestX(self): 
        return min(int(x) for x in self.x)

    def getHighestX(self): 
        return max(int(x) for x in self.x)
    
    def getLowestY(self):
        if(type(self.y[0]) == list):   
            return min(min(self.y))
        else:
            return min(int(y) for y in self.y)
            
    def getHighestY(self):
        if(type(self.y[0]) == list):   
            return max(max(self.y))
        else:
            return max(int(y) for y in self.y)

    def getXBounds(self): return self.x_bounds
    def getYBounds(self): return self.y_bounds
    
    def getXDim(self): return self.x_dim
    def getYDim(self): return self.y_dim
    
    def getXEta(self): return self.x_eta
    def getYEta(self): return self.y_eta
    
    # Add pair to dataset
    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.size = len(self.x)
        
    # Read dataset from controller
    def readSetFromController(self, controller):
        for i in range(controller.getSize()):
            pair = controller.getPairFromIndex(i)
            self.x.append(pair[0])
            self.y.append(pair[1])
            
        self.size = len(self.x)
        
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        
        self.x_eta = controller.getStateSpaceEtas()
        self.y_eta = controller.getInputSpaceEtas()
        
        utils = Utilities()
        print("Dataset size: " + str(self.size) + " - " + utils.formatBytes(self.size*2*4) + " (int32)")
            
    # Read pseudo random subset from controller
    def readSubsetFromController(self, controller, percentage):
        size = controller.getSize()
        ids = []
        new_size = round(size*percentage)
        
        # add highest and lowest controller to make sure the set had the same input and output format
        h_s, l_s = controller.getHighestState(), controller.getLowestState()
        ids.append(controller.getIndexOfState(l_s))
        ids.append(controller.getIndexOfState(h_s))
        
        # get random ids 
        random_ids = random.sample(range(0, size), (new_size - 2))
        ids += random_ids
        
        # fill dataset
        for i in range(new_size):
            pair = controller.getPairFromIndex(ids[i])
            self.x.append(pair[0])
            self.y.append(pair[1])
            
        self.size = len(self.x)
                    
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        
        self.x_eta = controller.getStateSpaceEtas()
        self.y_eta = controller.getInputSpaceEtas()
        
        utils = Utilities()
        print("Dataset size: " + str(self.size) + " - " + utils.formatBytes(self.size*2*4) + " (int32)")


    # Shuffle data
    def shuffle(self):
        pairs = []
        for i in range(self.size):
            pairs.append([self.x[i], self.y[i]])
            
        random.shuffle(pairs)
        n_x, n_y = [], []
        for i in range(self.size):
            n_x.append(pairs[i][0])
            n_y.append(pairs[i][1])
            
        self.x = n_x
        self.y = n_y
    
    # Get a batch from the data set
    def getBatch(self, size, i):
        x_batch = []
        y_batch = []
        for i in range(i, i + size):
            x_batch.append(self.x[i%self.size])
            y_batch.append(self.y[i%self.size])
        # print(y_batch)
        return [x_batch, y_batch]
            
    
    # Format dataset to binary inputs and outputs
    def formatToBinary(self):
        bed = BinaryEncoderDecoder()
        
        new_x = []
        new_y = []
        
        # the number of bit needed to store the input and output on binary format
        n_x = len(bed.sntob(self.x_bounds[1])) # Input bit length
        n_y = len(bed.sntob(self.y_bounds[1])) # Output bit length
        
        # convert the input and output pair to array of binary
        for i in range(self.size):
            new_x.append(bed.ntoba(self.x[i],n_x))
            new_y.append(bed.ntoba(self.y[i],n_y))
            
        # set x and y to converted x and y
        self.x = new_x
        self.y = new_y
        
        # set binary upper and lower bounds
        self.x_bounds = [bed.ntoba(self.x_bounds[0], n_x), bed.ntoba(self.x_bounds[1], n_x)]
        self.y_bounds = [bed.ntoba(self.y_bounds[0], n_y), bed.ntoba(self.y_bounds[1], n_y)]
        
        # set the dimension of x and y (elements per element of x and y)
        self.x_dim = n_x
        self.y_dim = n_y
        
        # set etas of x and y (0.5 for binary)
        self.x_eta = []
        self.y_eta = []
        for i in range(n_x):
            self.x_eta.append(0.5)
        for i in range(n_y):
            self.y_eta.append(0.5)
        
        self.size = len(self.x)


    # create a complete dataset for all grid point and also add additional bit for valid flag
    def createDeterministicCompleteDataset(self, controller):
        # create empty array equal to number of grid point
        n_gp = controller.getTotalGridPoints()
        x_all_gp = np.array(range(n_gp))[:,None]
        
        # empty array of control inputs for all grid points
        y_all_gp = np.zeros([n_gp, 2], dtype=np.int32)

        # concatenate empty array of complete dataset
        complete_dataset = np.concatenate((x_all_gp, y_all_gp), axis = 1)
        # concantenate dataset for only winning domain
        valid_dataset = np.concatenate((np.array(self.x)[:,None], np.array(self.y)[:,None]), axis = 1)
        
        # add valid flag for winning domain pair
        valid_flag = np.ones([self.size, 1], dtype=np.int32)
        valid_dataset = np.concatenate((valid_dataset, valid_flag), axis = 1)

        # replace the winning domain value on empty complete dataset with the valid one
        complete_dataset[valid_dataset[:,0]] = valid_dataset

        return complete_dataset


    def createStateSeparateBooleanBinary(self, controller):
        self.x = np.array(list(map(controller.stoss, self.x)))
        state_space_dim = int(controller.state_space_dim)
        X = []
        n_X = []
        for j in range(state_space_dim):
            X.append(self.x[:,j])
            n_X.append(len(bin(int(np.max(X[j]))))-2)

        controller.bit_dim = n_X

        swapped = X[0][:,None]
        for k in range(len(X)-1):
            swapped = np.concatenate((X[k+1][:,None], swapped), axis = 1)
            
        x_int8 = swapped.view(np.uint8)
        x_ordered = np.flip(x_int8,1)
        total_bit = 32
        x_unpacked = np.unpackbits(x_ordered).reshape(-1,total_bit*state_space_dim)

        idx_keep = []
        for k in range(state_space_dim):
            idx_keep.append(np.arange((k+1)*total_bit-n_X[k], (k+1)*total_bit))
            
        reduced = idx_keep[0]
        for k in range(len(X)-1):
            reduced = np.concatenate((reduced, idx_keep[k+1]))

        new_x = (x_unpacked[:,reduced]).tolist()
        return new_x

    # bin to bin
    def addAllGridPointDeterministicAbstractionBoolean(self, controller):

        complete_dataset = self.createDeterministicCompleteDataset(controller)
        
        new_x = []
        new_y = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.y = complete_dataset[:,1] # y_all_gp
        self.size = len(self.x)

        bed = BinaryEncoderDecoder()
        
        n_x = len(bed.sntob(self.x_bounds[1])) # Input bit length
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.x_bounds = [bed.ntoba(self.x_bounds[0], n_x), bed.ntoba(self.x_bounds[1], n_x)]
        
        n_y = len(bed.sntob(self.y_bounds[1])) # Output bit length
        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        self.y_bounds = [bed.ntoba(self.y_bounds[0], n_y), bed.ntoba(self.y_bounds[1], n_y)]

        # convert the input and output pair to array of binary
        # and add the flag bit
        for i in range(self.size):
            new_x.append(bed.ntoba(self.x[i],n_x))
            add_flag = bed.ntoba(self.y[i],n_y)
            add_flag.append(complete_dataset[i,2])
            new_y.append(add_flag) 

        self.x = new_x        
        self.x_dim = n_x
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        self.y = new_y
        self.y_dim = n_y+1
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)


    def addAllGridPointDeterministicSeparateBoolean(self, controller):

        complete_dataset = self.createDeterministicCompleteDataset(controller)
        
        new_x = []
        new_y = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.y = complete_dataset[:,1] # y_all_gp;
        self.size = len(self.x)

        bed = BinaryEncoderDecoder()
        
        self.x = np.array(list(map(controller.stoss, self.x)))
        state_space_dim = int(controller.state_space_dim)
        X = []
        n_X = []
        for j in range(state_space_dim):
            X.append(self.x[:,j])
            n_X.append(len(bin(int(np.max(X[j]))))-2)
            print(np.max(X[j]))

        print(n_X)

        controller.bit_dim = n_X

        swapped = X[0][:,None]
        for k in range(len(X)-1):
            swapped = np.concatenate((X[k+1][:,None], swapped), axis = 1)
            
        x_int8 = swapped.view(np.uint8)
        x_ordered = np.flip(x_int8,1)
        total_bit = 32
        x_unpacked = np.unpackbits(x_ordered).reshape(-1,total_bit*state_space_dim)

        idx_keep = []
        for k in range(state_space_dim):
            idx_keep.append(np.arange((k+1)*total_bit-n_X[k], (k+1)*total_bit))
            
        reduced = idx_keep[0]
        for k in range(len(X)-1):
            reduced = np.concatenate((reduced, idx_keep[k+1]))

        print(x_unpacked[:,reduced].shape)
        new_x = (x_unpacked[:,reduced]).tolist()
        n_x = len(new_x[0])
        print(n_x)

        n_y = len(bed.sntob(self.y_bounds[1])) # Output bit length
        # bounds for y are still the same, but bounds for x are different, just let it be for now 
        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        self.y_bounds = [bed.ntoba(self.y_bounds[0], n_y), bed.ntoba(self.y_bounds[1], n_y)]

        # convert the input and output pair to array of binary
        # and add the flag bit
        for i in range(self.size):
            add_flag = bed.ntoba(self.y[i],n_y)
            add_flag.append(complete_dataset[i,2])
            new_y.append(add_flag) 

        self.x = new_x        
        self.x_dim = n_x
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        self.y = new_y
        self.y_dim = n_y+1
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)


    def createDeterminizingCompleteDataset(self, controller):
        # create empty array equal to number of grid point
        n_gp = controller.getTotalGridPoints()
        x_all_gp = np.array(range(n_gp))[:,None]
        
        cardinality_u = controller.input_total_gp
        # empty array of control inputs for all grid points
        y_all_gp = np.zeros([n_gp, cardinality_u], dtype=np.int32)
        invalid_flag = np.ones([n_gp, 1], dtype=np.int32)
        y_all_gp = np.concatenate((y_all_gp, invalid_flag), axis = 1)

        # concatenate empty array of complete dataset
        complete_dataset = np.concatenate((x_all_gp, y_all_gp), axis = 1)
        
        # concantenate dataset for only winning domain
        valid_y = np.zeros([self.size, cardinality_u+1],  dtype=np.int32)
        for i,admissible_inputs in enumerate(self.y):
            for j,adm_input in enumerate(admissible_inputs):
                valid_y[i, adm_input] = 1

        valid_dataset = np.concatenate((np.array(self.x)[:,None], valid_y), axis = 1)
        
        # replace the winning domain value on empty complete dataset with the valid one
        complete_dataset[valid_dataset[:,0]] = valid_dataset

        return complete_dataset


    # bin to bin
    def addAllGridPointDeterminizingAbstractionBoolean(self, controller):

        complete_dataset = self.createDeterminizingCompleteDataset(controller)
        
        new_x = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.size = len(self.x)

        bed = BinaryEncoderDecoder()

        n_x = len(bed.sntob(self.x_bounds[1])) # Input bit length
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.x_bounds = [bed.ntoba(self.x_bounds[0], n_x), bed.ntoba(self.x_bounds[1], n_x)]
        
        # convert the input and output pair to array of binary
        # and add the flag bit
        for i in range(self.size):
            new_x.append(bed.ntoba(self.x[i],n_x))

        self.x = new_x        
        self.x_dim = n_x
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        self.y = (complete_dataset[:,1:]).tolist()
        self.y_dim = controller.input_total_gp+1
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)

        
    def addAllGridPointDeterminizingSeparateBoolean(self, controller):

        complete_dataset = self.createDeterminizingCompleteDataset(controller)
        
        new_x = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.size = len(self.x)

        bed = BinaryEncoderDecoder()
        
        
        new_x = self.createStateSeparateBooleanBinary(controller)
        n_x = len(new_x[0])
        # print(new_x)
        # print(n_x)

        self.x = new_x        
        self.x_dim = n_x
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        self.y = (complete_dataset[:,1:]).tolist()
        self.y_dim = controller.input_total_gp+1
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)


    # NonDeterministic Helper
    def createDictionary(self, filename):
        file = open(filename + ".scs",'r')
        if(file == None):
            print("Unable to open " + filename + ".scs")
        
        read = file.read()
        read_split = read.split(' \n')
        del read_split[0]
        del read_split[-1]
        
        input_dict = {}
        
        for i in range(len(read_split)):
            split = read_split[i].split(' ')
            u = [int(x) for x in split[1:]]
            input_dict[i+1] = set(u)
            
        return input_dict

    def changeInputsValue(self, controller, input_dict):
        for j in range(len(controller.inputs)):
            for i in range(len(input_dict)):
                if set(controller.inputs[j]) == (input_dict[i+1]):
                    # print(i+1)
                    controller.inputs[j] = i+1
                    break

    def createNonDeterministicCompleteDataset(self, filename, controller):

        input_dict = self.createDictionary(filename)
        self.changeInputsValue(controller, input_dict)

        # get total number of grid point to create the based/skeleton for complete dataset
        n_gp = controller.getTotalGridPoints()
        # create states from 0 to n_gp-1
        x_all_gp = np.array(range(n_gp))[:,None]
        # based control inputs, default value = 0
        y_all_gp = np.zeros([n_gp,1], dtype=np.int32)
        # skeleton complete dataset
        complete_dataset = np.concatenate((x_all_gp, y_all_gp), axis = 1)
        # concat the states and input
        valid_dataset = np.concatenate((np.array(controller.states)[:,None], (np.array(controller.inputs)[:,None])), axis = 1)
        # replace the winning domain value on empty complete dataset with the valid one
        complete_dataset[valid_dataset[:,0]] = valid_dataset
        return complete_dataset


    def addAllGridPointNonDeterministicAbstractionBoolean(self, input_file, controller):
        complete_dataset = self.createNonDeterministicCompleteDataset(input_file, controller)
        
        new_x = []
        new_y = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.y = complete_dataset[:,1] # y_all_gp
        self.size = len(self.x)

        bed = BinaryEncoderDecoder()
        
        n_x = len(bed.sntob(self.x_bounds[1])) # Input bit length
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.x_bounds = [bed.ntoba(self.x_bounds[0], n_x), bed.ntoba(self.x_bounds[1], n_x)]
        
        n_y = len(bed.sntob(self.y_bounds[1])) # Output bit length
        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        self.y_bounds = [bed.ntoba(self.y_bounds[0], n_y), bed.ntoba(self.y_bounds[1], n_y)]

        # convert the input and output pair to array of binary
        # and add the flag bit
        for i in range(self.size):
            new_x.append(bed.ntoba(self.x[i],n_x))
            new_y.append(bed.ntoba(self.y[i],n_y))

        self.x = new_x        
        self.x_dim = n_x
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        self.y = new_y
        self.y_dim = n_y
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)

    
    def addAllGridPointNonDeterministicSeparateBoolean(self, input_file, controller):
        complete_dataset = self.createNonDeterministicCompleteDataset(input_file, controller)
        
        new_x = []
        new_y = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.y = complete_dataset[:,1] # y_all_gp
        self.size = len(self.x)

        bed = BinaryEncoderDecoder()
       
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        new_x = self.createStateSeparateBooleanBinary(controller)
        n_x = len(new_x[0])
        # print(new_x)
        self.x_bounds = [new_x[0], new_x[-1]]

        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        n_y = len(bed.sntob(self.y_bounds[1])) # Output bit length
        self.y_bounds = [bed.ntoba(self.y_bounds[0], n_y), bed.ntoba(self.y_bounds[1], n_y)]

        # convert the input and output pair to array of binary
        # and add the flag bit
        for i in range(self.size):
            new_y.append(bed.ntoba(self.y[i],n_y))

        self.x = new_x        
        self.x_dim = n_x
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        self.y = new_y
        self.y_dim = n_y
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)



    # add all grid point to the winning domain only controller
    def addAllGridPointDeterministic(self, controller, encode_type, var_order):
        self.setEncodeType(encode_type)
        self.setVariableOrder(var_order)

        if(self.encode == EncodeTypes.Regression):
            if(self.order == Ordering.Original):
                self.addAllGridPointDeterministicAbstractionRegression(controller)
            else:
                self.addAllGridPointlDeterministicSeparateRegression(controller)
        elif(self.encode == EncodeTypes.Classification):
            if(self.order == Ordering.Original):
                self.addAllGridPointDeterministicAbstractionClassification(controller)
            else:
                self.addAllGridPointDeterministicSeparateClassification(controller)
        else:
            if(self.order == Ordering.Original):
                self.addAllGridPointDeterministicAbstractionBoolean(controller)
            else:
                self.addAllGridPointDeterministicSeparateBoolean(controller)


    # add all grid point to the winning domain only controller
    def addAllGridPointDeterminizing(self, controller, var_order):
        self.setVariableOrder(var_order)

        if(self.order == Ordering.Original):
            self.addAllGridPointDeterminizingAbstractionBoolean(controller)
        else:
            self.addAllGridPointDeterminizingSeparateBoolean(controller)


    # add all grid point to the winning domain only controller
    def addAllGridPointNonDeterministic(self, input_file, controller, var_order):
        self.setVariableOrder(var_order)

        if(self.order == Ordering.Original):
            self.addAllGridPointNonDeterministicAbstractionBoolean(input_file, controller)
        else:
            self.addAllGridPointNonDeterministicSeparateBoolean(input_file, controller)


    # need to be refactorized # 
    # int to int
    def addAllGridPointDeterministicAbstractionRegression(self, controller):

        complete_dataset = self.createDeterministicCompleteDataset(controller)
        
        new_x = []
        new_y = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.y = complete_dataset[:,1] # y_all_gp
        self.size = len(self.x)

        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        new_x = (self.x[:,None]).tolist()
        self.x = new_x        
        self.x_dim = 1
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        # convert the input and output pair to array of binary
        for i in range(self.size):
            (self.y[i]).append(complete_dataset[i,2])
            new_y.append(add_flag) 
        self.y = new_y
        self.y_dim = 2
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)


    # int to bin
    def addAllGridPointDeterministicAbstractionClassification(self, controller):

        complete_dataset = self.createDeterministicCompleteDataset(controller)
        
        new_x = []
        new_y = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.y = complete_dataset[:,1] # y_all_gp
        self.size = len(self.x)

        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        new_x = (self.x[:,None]).tolist()
        self.x = new_x        
        self.x_dim = 1
        self.x_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)

        # binary for output
        bed = BinaryEncoderDecoder()
        n_y = len(bed.sntob(self.y_bounds[1])) # Output bit length

        # change the bounds to binary
        self.y_bounds = [self.getLowestY(), self.getHighestY()]
        self.y_bounds = [bed.ntoba(self.y_bounds[0], n_y), bed.ntoba(self.y_bounds[1], n_y)]

        # convert the input and output pair to array of binary
        for i in range(self.size):
            add_flag = bed.ntoba(self.y[i],n_y)
            add_flag.append(complete_dataset[i,2])
            new_y.append(add_flag) 

        self.y = new_y
        self.y_dim = n_y+1
        self.y_eta = []
        for i in range(self.y_dim):
            self.y_eta.append(0.5)


    # int to int in separate dimension
    def addAllGridPointDeterministicSeparateRegression(self, controller):

        complete_dataset = self.createDeterministicCompleteDataset(controller)
        
        new_x = []
        new_y = []
        
        self.x = complete_dataset[:,0] # x_all_gp
        self.y = complete_dataset[:,1] # y_all_gp
        self.size = len(self.x)
        
        new_x = np.array(list(map(controller.stoss, self.x)))
        new_y = np.array(list(map(controller.itoii, self.y)))
        
        self.x_bounds = [self.getLowestX(), self.getHighestX()]
        self.y_bounds = [self.getHighestY(), self.getHighestY()]

        # convert the input and output pair to array of binary
        for i in range(self.size):
            (new_y[i]).append(complete_dataset[i,2])
            new_y.append(add_flag)
            
        # set binary upper and lower bounds
        self.x_bounds = [controller.stoss(self.x_bounds[0]), controller.stoss(self.x_bounds[1])]
        self.y_bounds = [controller.itoii(self.y_bounds[0]), controller.itoii(self.y_bounds[1])]    
        
        new_x = (new_x[:,None]).tolist()

        # set x and y to converted x and y
        self.x = new_x        
        self.y = new_y

        # set the dimension of x and y (elements per element of x and y)
        self.x_dim = int(controller.getStateSpaceDim())
        self.y_dim = int(controller.getStateSpaceDim())+1

        # set etas of x and y (0.5 for binary)
        self.x_eta = []
        self.y_eta = []
        for i in range(self.x_dim):
            self.x_eta.append(0.5)
        for i in range(self.y_dim):
            self.y_eta.append(0.5)


    def addAllGridPointDeterministicSeparateClassification(self, controller):
        pass