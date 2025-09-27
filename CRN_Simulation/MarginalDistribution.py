import numpy as np

class MarginalDistribution:

    def __init__(self, parameter_species_name, states, distribution):
        self.parameter_species_name = parameter_species_name
        self.states = states
        self.distribution = np.array(distribution)

        # check eglibility
        if type(self.parameter_species_name) != str:
            raise ValueError("parameter_species_name should be a string.")

        if len(self.states) != len(self.distribution):
            raise ValueError("The length of states and distribution should be the same.")

        if abs(np.sum(self.distribution) - 1) > 1e-6:
            print(self.parameter_species_name, np.sum(self.distribution))
            raise ValueError("The distribution should sum to 1.")

    def adjust_distribution(self, distribution):
        if len(self.distribution) != len(distribution):
            raise ValueError("The length of distribution and states should be the same.")
            return
        else:
            self.distribution = np.array(distribution)