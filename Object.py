import numpy as np
class Object:
    def __init__(self, number_of_nodes):
        self.number = number_of_nodes
        self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                         'P', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
        self.standard_deviation = np.random.normal(loc=0, scale=1, size=number_of_nodes)
        self.parameters = []
        self.adaptation = 0
        self.flag = 0
        self.capabilities = list(np.arange(0, self.number))
        for i in range(0, number_of_nodes):
            self.rand = np.random.randint(0, self.number)
            self.parameters.append(self.capabilities.pop(self.rand))
            self.number -= 1
        print(self.parameters)
        # print(self.standard_deviation)
