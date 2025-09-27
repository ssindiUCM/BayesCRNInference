
import numpy as np


from CRN_Simulation.MarginalDistribution import MarginalDistribution

class DistributionOfSystems():

    def __init__(self, states, parameter_species_ordering):
        self.states = states # a list of states
        self.species_ordering = parameter_species_ordering
        self.time_list = []
        self.distribution_list = []

    # include distributions at different time points.
    def extend_distributions(self, time_list, distribution_list):
        self.time_list.extend(time_list)
        self.distribution_list.extend(distribution_list)

    def replace_distributions(self, time_list, distribution_list):
        self.time_list = time_list
        self.distribution_list = distribution_list

    ##############################
    # get methods
    ##############################

    def get_states_of_the_element(self, element):
        if element in self.species_ordering.keys():
            return self.states[:,self.species_ordering[element]]
        else:
            raise ValueError('The element is not in the parameter_species_ordering.')

    def expectation(self):
        mean_list = [ ]
        for i in range(len(self.distribution_list)):
            result_i = np.dot(self.distribution_list[i].T, self.states)
            result_dict = {species: result_i[0,species_index] for species, species_index in self.species_ordering.items()}
            mean_list.append(result_dict)

        return self.time_list, mean_list

    def extract_marginal_distributions(self, distribution):
        """

        :return: a dictionary of marginal distributions
        """

        marginal_distributions = {}
        for species, species_index in self.species_ordering.items():
            state_list = np.unique(self.states[:,species_index]).reshape(-1,1)
            distribution_temp = np.zeros((len(state_list), 1))
            for x in state_list:
                index_of_x = np.where(state_list == x)[0]
                distribution_temp[index_of_x] = np.sum(distribution[self.states[:,species_index] == x])
            marginal_distributions.update({species: MarginalDistribution(species, state_list, distribution_temp)})

        return marginal_distributions

    def extract_marginal_distributions_over_time(self):
        """
        :param distribution_list: a list of distributions
        :return: a list of dictionaries of marginal distributions
        """
        marginal_distributions_list = []
        for distribution in self.distribution_list:
            marginal_distributions_list.append(self.extract_marginal_distributions(distribution))
        return marginal_distributions_list