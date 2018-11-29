
# PlainController class which will hold the controller that can then be accessed in order to read training
# data for the neural network
class StaticController:
    def __init__(self):
        self.state_space_dim = None
        self.state_space_etas = None
        self.state_space_lower_left = None
        self.state_space_upper_right = None

        self.input_space_dim = None
        self.input_space_etas = None
        self.input_space_lower_left = None
        self.input_space_upper_right = None

        self.states = []
        self.inputs = []
        
        self.state_size = 0
        self.input_size = 0

        self.state_no_grid_points = []
        self.input_no_grid_points = []

        self.state_total_gp = 0
        self.input_total_gp = 0
        
        self.x2a_shift = None
        # self.first = []
        self.index_inc = []
        self.index_inc_input = []         # index needed to increase to the next value for the state on dimension i
        
        self.bit_dim = []
        self.con_det = None

    # Getters
    def getStateSpaceDim(self): return self.state_space_dim
    def getStateSpaceEtas(self): return self.state_space_etas
    def getStateSpaceLowerLeft(self): return self.state_space_lower_left
    def getStateSpaceUpperRight(self): return self.state_space_upper_right
    
    def getInputSpaceDim(self): return self.input_space_dim
    def getInputSpaceEtas(self): return self.input_space_etas
    def getInputSpaceLowerLeft(self): return self.input_space_lower_left
    def getInputSpaceUpperRight(self): return self.input_space_upper_right
    
    def getState(self, id): return self.states[id]
    def getInput(self, id): return self.inputs[id]
    
    # Get the input id and state id for a given state id
    def getPairFromState(self, state):
        for i in range(self.state_size):
            if(self.states[i] == state):
                return [self.states[i], self.inputs[i]]
        return None
    
    # Get the input id and state id for a given state index
    def getPairFromIndex(self, id):
        if(id >= 0 and id < self.state_size):
            return [self.states[id], self.inputs[id]]
        return None
        
    # Get the input id corresponding to a given state id
    def getInputFromState(self, state):
        for i in range(self.state_size):
            if(self.states[i]  == state):
                return self.inputs[i]
        print("ID does not correspond to a state in the winning domain.")
        return None
    
    # Get lowest state id contained in the controller
    def getLowestState(self):
        return min(int(s) for s in self.states)
    
    # Get highest state id contained in the controller
    def getHighestState(self):
        return max(int(s) for s in self.states)
    
    # Get lowest input id contained in the controller
    def getLowestInput(self):
        return min(int(i) for i in self.inputs)
    
    # Get highest input id contained in the controller
    def getHighestInput(self):
        return max(int(i) for i in self.inputs)
    
    # Get the size of the controller
    def getSize(self):
        return self.state_size
    
    # Get index of value
    def getIndexOfState(self, state):
        return self.states.index(state)

     # Get total number of grid points
    def getTotalGridPoints(self): return self.state_total_gp
        
    # Setters
    def setStateSpaceDim(self, value): self.state_space_dim = value
    def setStateSpaceEtas(self, value): self.state_space_etas = value
    def setStateSpaceLowerLeft(self, value): self.state_space_lower_left = value
    def setStateSpaceUpperRight(self, value): self.state_space_upper_right = value
    
    def setInputSpaceDim(self, value): self.input_space_dim = value
    def setInputSpaceEtas(self, value): self.input_space_etas = value
    def setInputSpaceLowerLeft(self, value): self.input_space_lower_left = value
    def setInputSpaceUpperRight(self, value): self.input_space_upper_right = value

    def setDeterministic(self, value): self.con_det = value

    def setStateNoGridPoints(self):
        gp_list = []
        state_dim = int(self.state_space_dim)
        for i in range(state_dim):
            ur = float(self.getStateSpaceUpperRight()[i]) 
            ll =  float(self.getStateSpaceLowerLeft()[i])
            eta = float(self.getStateSpaceEtas()[i])
            gp_list.append(int((ur - ll)/eta+1))
        self.state_no_grid_points = gp_list

        total = 1
        for i in range(state_dim):
            self.index_inc.append(total)
            total *= gp_list[i]

        self.state_total_gp = total
    
    def setInputNoGridPoints(self):
        gp_list = []
        for i in range(int(self.input_space_dim)):
            ur = float(self.getInputSpaceUpperRight()[i]) 
            ll =  float(self.getInputSpaceLowerLeft()[i])
            eta = float(self.getInputSpaceEtas()[i])
            gp_list.append(int((ur - ll)/eta+1))
        self.input_no_grid_points = gp_list

        total = 1
        for i in range((int(self.input_space_dim))):
            self.index_inc_input.append(total)
            total *= gp_list[i]

        self.input_total_gp = total

    def setStateInput(self, s, i):
        #self.states.append(int(s))
        #self.inputs.append(int(i))
        self.states.append(s)
        self.inputs.append(i)        

    def setSize(self):
        self.state_size = len(self.states)
        self.input_size = len(self.inputs)

    # helper to change abstraction two separate dimension
    def stoss(self, s):
        dim_s = []
        for k in range(int(self.state_space_dim)-1, 0, -1):
            num = s//self.index_inc[k]
            s = s%self.index_inc[k]
            dim_s.insert(0, num)
        dim_s.insert(0, s)
        return dim_s

    # helper to revert back from two separate dimension to one dimension abstraction
    def sstos(self, ss):
        s = 0
        for k in range(int(self.state_space_dim)):
            s += ss[k]*self.index_inc[k]
        return s 
    
    # helper to change abstraction two separate dimension
    def itoii(self, s):
        dim_s = []
        for k in range(int(self.input_space_dim)-1, 0, -1):
            num = s//self.index_inc_input[k]
            s = s%self.index_inc_input[k]
            dim_s.insert(0, num)
        dim_s.insert(0, s)
        return dim_s

    def sstox(self, ss):
        x = []
        for k in range(int(self.state_space_dim)):
            first_point = float(self.state_space_lower_left[k])
            eta = float(self.state_space_etas[k])
            x.append(first_point+ss[k]*eta)
        return x

    def stox(self, s):
        x = []
        for k in range(int(self.state_space_dim)-1, 0, -1):
            num = s//self.index_inc[k]
            s = s%self.index_inc[k]
            first_point = float(self.state_space_lower_left[k])
            eta = float(self.state_space_etas[k])
            x.insert(0, first_point+num*eta)
        first_point = float(self.state_space_lower_left[0])
        eta = float(self.state_space_etas[0])
        x.insert(0, first_point+s*eta)
        return x

    """
    def istoi(self, dim_is, s):
        s = 0
        for k in range(self.state_space_dim):
            if(dim_is[k] >= no_grid_points[k]):
                return False
            s += dim_is[k]*self.index_inc[k]
        return True
    """

    

    
    
            

    





