import sys
sys.path.append("C:\\Users\\Gebruiker\\Dropbox\\SC2NN\\src")
# print(sys.path)

from Importer import Importer
from Exporter import Exporter
from NeuralNetworkManager import NeuralNetworkManager
from StaticController import StaticController
from DataSet import DataSet
from Utilities import Utilities

from NeuralNetworkManager import NNTypes
from NeuralNetworkManager import NNOptimizer
from NeuralNetworkManager import NNActivationFunction

from DataSet import EncodeTypes
from DataSet import Ordering

import matplotlib.pyplot as plt

# Main class from which all functions are called
class COTONN:
    def __init__(self):
        self.version = "2.0"
        
        self.utils = Utilities()
        self.save_path = "./nn/" + self.utils.getTimestamp() + "/"
        
        self.importer = Importer()
        self.exporter = Exporter(self.version)
        self.exporter.setSaveLocation(self.save_path)
        
        self.nnm = NeuralNetworkManager()
        self.nnm.setSaveLocation(self.save_path)
        self.staticController = StaticController()
        
        self.debug_mode = False
        
        self.importer.setDebugMode(False)
        self.nnm.setDebugMode(self.debug_mode)
        
        print("COTONN v" + self.version + "\n")

        self.encode = EncodeTypes.Boolean
        self.var_order = Ordering.PerCoordinate
        self.filename = ""
        
    
    # Clean memory function
    def cleanMemory(self):
        del self.nnm.nn
        del self.nnm.controller
        del self.nnm
        del self.staticController
        del self.exporter
        del self.importer


         # Generate MLP from fullset
    def deterministicMLP(self, filename, layer, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        self.importer.det = True
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate

        
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        fullSet.addAllGridPointDeterministic(self.staticController, self.encode, self.var_order)

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Sigmoid)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.customHiddenLayers(layer)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        
        self.nnm.getDataSize()
        
        # self.nnm.setEpochThreshold(172)
        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()


    # Generate MLP from fullset
    def importDeterministicMLP(self, import_path, filename, layer, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        self.importer.det = True
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
        
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        # fullSet.formatToBinary()
        fullSet.addAllGridPointDeterministic(self.staticController, self.encode, self.var_order)
        
        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        # self.nnm.setActivationFunction(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Sigmoid)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController
         
        # Option to adjust parameters for new training session
        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.customHiddenLayers(layer)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step)
        
        self.nnm.getDataSize()
        
         # Restore Network from saved file:
        self.importer.restoreNetwork(self.nnm, import_path)
      
        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()
        
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        self.nnm.randomCheck(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        self.cleanMemory()


    # Generate MLP from fullset
    def deterministicRectMLP(self, filename, layer_width, layer_height, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        
        self.importer.det = True
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate

        
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        fullSet.addAllGridPointDeterministic(self.staticController, self.encode, self.var_order)

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Sigmoid)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.rectangularHiddenLayers(layer_width, layer_height)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        
        self.nnm.getDataSize()
        
        # self.nnm.setEpochThreshold(172)
        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()


    # Generate MLP from fullset
    def importDeterministicRectMLP(self, import_path, filename, layer_width, layer_height, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        self.importer.det = True
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
        
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        # fullSet.formatToBinary()
        fullSet.addAllGridPointDeterministic(self.staticController, self.encode, self.var_order)
        
        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        # self.nnm.setActivationFunction(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Sigmoid)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController
         
        # Option to adjust parameters for new training session
        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.rectangularHiddenLayers(layer_width, layer_height)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step)
        
        self.nnm.getDataSize()
        
         # Restore Network from saved file:
        self.importer.restoreNetwork(self.nnm, import_path)
      
        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()
        
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        self.nnm.randomCheck(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        self.cleanMemory()


    # Generate MLP from fullset
    def determinizingMLP(self, filename, layer, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        self.importer.det = False
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
   
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        
        fullSet.addAllGridPointDeterminizing(self.staticController, self.var_order)

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Softmax)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.customHiddenLayers(layer)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        self.nnm.getDataSize()
        
        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()


    # Generate MLP from fullset
    def importDeterminizingMLP(self, import_path, filename, layer, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        
        self.importer.det = False
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
   
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        
        fullSet.addAllGridPointDeterminizing(self.staticController, self.var_order)

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Softmax)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.customHiddenLayers(layer)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        self.nnm.getDataSize()
        
        # Restore Network from saved file:
        self.importer.restoreNetwork(self.nnm, import_path)

        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()


    # Generate MLP from fullset
    def determinizingRectMLP(self, filename, layer_width, layer_height, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        self.importer.det = False
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
   
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        
        fullSet.addAllGridPointDeterminizing(self.staticController, self.var_order)

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Softmax)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.rectangularHiddenLayers(layer_width, layer_height)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        self.nnm.getDataSize()
        
        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()


    # Generate MLP from fullset
    def importDeterminizingRectMLP(self, import_path, filename, layer_width, layer_height, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        
        self.importer.det = False
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
   
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        
        fullSet.addAllGridPointDeterminizing(self.staticController, self.var_order)

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Softmax)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.rectangularHiddenLayers(layer_width, layer_height)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        self.nnm.getDataSize()
        
        # Restore Network from saved file:
        self.importer.restoreNetwork(self.nnm, import_path)

        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()


    # Generate MLP from fullset
    def nonDeterministicMLP(self, filename, input_file, layer, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        self.importer.det = False
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
   
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        
        fullSet.addAllGridPointNonDeterministic(input_file, self.staticController, self.var_order)
        self.nnm.non_det = True

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Sigmoid)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.customHiddenLayers(layer)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        self.nnm.getDataSize()
        
        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()


    # Generate MLP from fullset
    def importNonDeterministicMLP(self, import_path, input_file, filename, layer, encode = 0, var_order = 0, learning_rate = 0.01, dropout_rate = 0.0, fitness_threshold = 1.0, batch_size = 1024, display_step = 50, save_option=True):
        
        self.importer.det = False
        self.staticController = self.importer.readStaticController(filename)

        self.filename = filename

        if(encode == 0):
            self.encode = EncodeTypes.Boolean
        if(var_order == 0):
            self.var_order = Ordering.Original
        else:
            self.var_order = Ordering.PerCoordinate
   
        fullSet = DataSet()
        fullSet.readSetFromController(self.staticController)
        
        fullSet.addAllGridPointNonDeterministic(input_file, self.staticController, self.var_order)

        self.nnm.setDebugMode(True)
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)

        self.nnm.setActivationFunctionHidden(NNActivationFunction.Sigmoid)
        self.nnm.setActivationFunctionOutput(NNActivationFunction.Sigmoid)
        self.nnm.setEncodeTypes(self.encode)

        self.nnm.setDataSet(fullSet)
        self.nnm.controller = self.staticController

        self.nnm.setDropoutRate(dropout_rate)
        self.nnm.customHiddenLayers(layer)
        self.nnm.initialize(learning_rate, fitness_threshold, batch_size, display_step, -1, 5000)
        self.nnm.getDataSize()
        
        # Restore Network from saved file:
        self.importer.restoreNetwork(self.nnm, import_path)

        # Train model and visualize performance
        self.nnm.train()
        
        self.nnm.plot()

        # fitness, wrong_states = self.nnm.checkFitness(fullSet)
        self.nnm.randomCheck(fullSet)
        fitness, wrong_states = self.nnm.checkFitnessAllGridPoint(fullSet)
        loosing_states = self.nnm.createLoosingPoints(wrong_states)   

        if(save_option):
            self.exporter.saveNetwork(self.nnm)
            self.exporter.saveWrongStates(wrong_states)
            self.exporter.saveMatlabMLP(self.staticController, self.nnm)
            self.exporter.saveBinary(self.nnm)
            self.exporter.saveTrainingData(self.nnm)
            self.exporter.saveUpdatedAvoid(self.filename, wrong_states, loosing_states)

        self.nnm.close()
        
        self.cleanMemory()
