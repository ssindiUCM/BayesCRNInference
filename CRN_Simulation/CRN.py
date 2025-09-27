
# how to write the propensity matrix
#propensities = [
#    lambda mrna, protein, k1 : mrna+protein*k1,
#    lambda mrna, protein, k2 : protein + ...,
#    ...
#] 

import inspect
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import sparse
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed

from CRN_Simulation.DistributionOfSystems import DistributionOfSystems
from CRN_Simulation.MarginalDistribution import MarginalDistribution
from CRN_Simulation.MatrixExponentialKrylov import MatrixExponentialKrylov


class CRN():

    def __init__(self, stoichiometric_matrix, species_names, parameters_names, reaction_names, propensities):
        """
        Create a CRN object

        :param stoichiometric_matrix: numpy array
        :param species_names: list of strings
        :param parameters_names: list of strings
        :param reaction_names: list of strings
        :param propensities:
        """
        self.stoichiometric_matrix = np.array(stoichiometric_matrix) # numpy object

        self.species_names = species_names # list of strings
        self.reaction_names = reaction_names # list of strings
        # save the ordering for further use
        self.species_ordering = { self.species_names[i]:i for i in range(len(species_names)) }
        self.reaction_ordering = { self.reaction_names[i]: i for i in range(len(reaction_names))}

        # TODO update s.m. so that self.species_names labels the rows
        self.parameters_names = parameters_names # list of strings
        self.propensities = propensities # list of functions
        self.propensities_dependencies = [ inspect.getfullargspec(p).args for p in self.propensities ] # [[strings]]

        # define state and parameters
        self.state = np.zeros( [len(self.species_names)] ) # the state is a numpy array
        self.parameters = {} # the set of parameters is just a dictionary

        # variables related to LNA
        self.propensities_derivative_in_state = None

        # variables related to Moment Closure
        self.first_moment_names = ['mu_' + species for species in self.species_names]
        self.second_moment_names = ['y_' + species1 + '_' + species2 for species1 in self.species_names for species2 in self.species_names]
        self.propensity_expectation_approximation_via_moments = None # a list of lambda functions
        self.propensity_times_X_trans_expectation_approximation_via_moments = None # a list of lambda functions




    # incomplete stub for calling the propensity functions on the right parametrs
    # self.propensities[i](*[  self.propensities_dependencies ])

    def _eval(self):
        """
        Evaluate propensity functions on the current state
        """
        out = []
        for prop in self.propensities:
            args = inspect.getfullargspec(prop).args
            input_args = {} # input args of this specific propensity
            for a in args:
                try:
                    input_args[a] = self.parameters[a]
                except KeyError:
                    try:
                        input_args[a] = self.state[self.species_ordering[a]]
                    except KeyError:
                        raise Exception(f'argument {a} is nor a parameter or a species')
            out.append(prop(**input_args))
        return np.array(out)

    def set_state(self, state):
        """Set the state to a specific value

        Args:
            state (Dictionary): an (exhaustive!) dictionary setting the values for each element of the state
        """
        for a in self.species_names:
            self.state[self.species_ordering[a]] = state[a]

    def set_parameters(self, parameters):
        """Set the value of the parameters

        Args:
            parameters (Dictionary): an (exhaustive!) dictionary setting the values for each parameter
        """
        self.parameters = {}
        for p in self.parameters_names:
            self.parameters[p] = parameters[p]

    def update(self, reaction_idx):
        """Update the state accordingly to a given reaction

        Args:
            reaction_idx (int): the index of the firing reaction
        """
        self.state += self.stoichiometric_matrix[:, reaction_idx]

    def get_number_of_reactions(self):
        return len(self.reaction_names)

    def get_number_of_species(self):
        return len(self.species_names)

    def get_number_of_parameters(self):
        return len(self.parameters_names)

    def SSA(self, state, parameters, T0, Tf):
        """

        :param state: a dictionary
        :param parameters: a disctionary
        :param T0: the inital time
        :param Tf: final time
        :return: time, state
        """
        # Initialize time and state variables
        t = T0
        self.set_parameters(parameters)
        self.set_state(state)

        # Store initial state in the output
        time_output = [t]
        state_output = [self.state.copy()]

        while t < Tf:
            # Calculate propensities
            a = self._eval()

            # Generate two random numbers
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Calculate the time until the next reaction occurs
            alpha = np.sum(a)
            if alpha == 0:
                break # no more reactions can occur
            else:
                tau = (1/alpha) * np.log(1/r1)

            # Choose the next reaction
            mu = np.argmax(np.cumsum(a) >= r2*alpha)

            # Update the time and state
            t = t + tau
            if t > Tf:
                break
            self.state = self.state + self.stoichiometric_matrix[:,mu]

            # Store the new state in the output
            time_output.append(t)
            state_output.append(self.state.copy())

        # the last update
        time_output.append(Tf)
        state_output.append(self.state.copy())

        return time_output.copy(), state_output.copy()

    # # paralleld SSA with a list of species and parameters
    # def parallel_SSA(self, list_of_states, list_of_parameters, T0, Tf):
    #
    #     if len(list_of_states) != len(list_of_parameters):
    #         raise Exception('The length of the list of states and parameters are not the same')
    #
    #     results = Parallel(n_jobs=-1)(
    #         delayed(self.SSA)(
    #             state = list_of_states[i],
    #             parameters= list_of_parameters[i],
    #             T0 = T0,
    #             Tf = Tf
    #         ) for i in range(len(list_of_states))
    #     )
    #
    #     return results


    # extract the dynamics of particular species
    def extract_trajectory(self, time_list, state_list, species_name_list):
        # initialization
        species_ordering = {species: count for species, count in zip(species_name_list, range(len(species_name_list)))}
        species_index = [self.species_ordering[species] for species in species_name_list]
        states = np.array(state_list)[:, species_index]
        time_new = [time_list[0]]
        state_new = [states[0,:]]

        for i in range(len(time_list)-2):
            dX = states[i+1,:] - states[i,:]
            if np.any(dX != 0):
                time_new.append(time_list[i+1])
                state_new.append(states[i+1,:])

        # record the final distribution
        time_new.append(time_list[-1])
        state_new.append( [states[-1, :]] )

        return time_new, state_new, species_ordering

    # marginal time average distribution for a given species
    def marginal_time_average_distribution(self, time_list, state_list, species_name):
        if species_name not in self.species_names:
            raise Exception(f'Species {species_name} is not in the model')


        species_index = self.species_ordering[species_name]
        state_list = np.array(state_list)
        max_state = int(np.max(state_list[:,species_index]))
        marginal_time_average = np.zeros(max_state+1)
        for i in range(len(time_list)-1):
            marginal_time_average[int(state_list[i,species_index])] += time_list[i+1] - time_list[i]

        return marginal_time_average / (time_list[-1] - time_list[0])

    ###############################################
    # FSP algorithm
    ###############################################

    def FSP(self, Initial_marginal_distributions, range_of_spcies, parameters, T0, Tf, normalization=False):
        """

        :param Initial_distributions: A list of marginal distributions for each species (assume that the species are independent
        :param range_of_spcies: a df with the range of each species
        :param parameters: a dictionary of parameters
        :param T0: initial time
        :param Tf: final time
        :return:
        """

        # Prepare the initial distribution
        species_ordering, states = self.FSP_prepare_the_state_space(range_of_spcies)
        distribution = DistributionOfSystems(states, species_ordering)
        initial_joint_distribution = self.generate_joint_distribution_from_marginal_distributions(Initial_marginal_distributions, distribution)
        initial_joint_distribution = initial_joint_distribution.reshape(-1,1)
        distribution.extend_distributions([0],[initial_joint_distribution])

        # Construct the A matrix
        A = self.constract_A_matrix_for_CME(distribution, parameters)

        # Solve the CME
        time_list, distribution_list = \
            MatrixExponentialKrylov.exp_AT_x(A, T0, Tf, initial_joint_distribution)

        # normalize the distribution
        if normalization == True:
            distribution_list = [propability / np.sum(propability) for propability in distribution_list]

        # save the result in the distribution
        distribution.extend_distributions(time_list, distribution_list)

        return distribution

    def FSP_prepare_the_state_space(self, range_of_species):
        """

        :param range_of_species: a df with the range of each species
        :return:
        """
        coords_species = [
            np.arange(range_of_species.loc[species, 'min'], range_of_species.loc[species, 'max'] + 1) \
            for species in self.species_ordering.keys()]

        size_of_state_space = np.prod([len(coords) for coords in coords_species])
        if size_of_state_space > 1e6:
            raise Exception(f'The size of state space is {size_of_state_space}, which is too large for FSP algorithm')

        # construct the state
        meshes = np.meshgrid(*coords_species, indexing='ij')
        matrix = np.stack(meshes, axis=-1)
        states = matrix.reshape(-1, matrix.shape[-1])  # each row is a state of hidden species and species

        return self.species_ordering, states

    # generate uniform marginal distributions for every species
    def generate_uniform_marginal_distributions_via_speceis_range(self, range_of_species):
        Marginal_distributions = {}
        # transverse all species
        for species in self.species_names:
            states = list(range(range_of_species.loc[species, 'min'], range_of_species.loc[species, 'max'] + 1))
            uniform_distribution = np.ones(len(states)) / len(states)
            marginal_uniform_distribution = MarginalDistribution(species, states, uniform_distribution)
            Marginal_distributions.update({species: marginal_uniform_distribution})
        return Marginal_distributions

    # generate uniform marginal distributions for every parameter
    def generate_uniform_marginal_distributions_via_parameter_range(self, range_of_parameters, discretization_size_parameters):
        Marginal_distributions = {}
        # transverse all parameters
        for parameter in self.parameters_names:
            states = list(np.linspace(range_of_parameters.loc[parameter, 'min'], range_of_parameters.loc[parameter, 'max'], discretization_size_parameters.loc[parameter, 0]))
            uniform_distribution = np.ones(len(states)) / len(states)
            marginal_uniform_distribution = MarginalDistribution(parameter, states, uniform_distribution)
            Marginal_distributions.update({parameter: marginal_uniform_distribution})
        return Marginal_distributions


    def generate_joint_distribution_from_marginal_distributions(self, Marginal_distributions, distribution):
        """

        :param Marginal_distributions: a dictionary of marginal distributions
        :param distribution: an object of DistributionOfSystems
        :return:
        """
        joint_distribution = np.ones(distribution.states.shape[0])
        for count, state in enumerate(distribution.states):
            for species in self.species_names:
                marginal_distribution = Marginal_distributions[species]
                state_index =   marginal_distribution.states.index(state[self.species_ordering[species]])
                joint_distribution[count] *= marginal_distribution.distribution[state_index]

        return joint_distribution

    def constract_A_matrix_for_CME(self, distribution, parameters):
        """

        :param distribution: an object of DistributionOfSystems
        :param parameters:
        :return:
        """
        # return row_index, column_index, reaction_list, state_column, sign
        column = []
        row = []
        value = []
        self.set_parameters(parameters)

        for state_index, state in enumerate(distribution.states): # transverse all states where the probability outflows
            state_dic = {species: state[distribution.species_ordering[species]] for species in distribution.species_ordering.keys()}
            self.set_state(state_dic)
            propensities = self._eval() # a list of propensities

            # probability outflows
            column.append(state_index)
            row.append(state_index)
            value.append(-np.sum(propensities))

            # probability inflows to other states
            for reaction_index, propensity in enumerate(propensities):
                state_after_reaction = state + self.stoichiometric_matrix[:, reaction_index]
                state_after_reaction_index = np.where(np.all(distribution.states == state_after_reaction, axis=1))[0]
                if len(state_after_reaction_index) > 0:
                    column.append(state_index)
                    row.append(state_after_reaction_index[0])
                    value.append(propensity)

        number_of_states = distribution.states.shape[0]
        A = sparse.coo_matrix((value, (row, column)), shape=(number_of_states, number_of_states)).tocsr()

        return A


    ###############################################
    # Linear Noise Approximation
    ###############################################

    def LNA(self, initial_state, parameters, T0, Tf, output_time_points, atol=1e-6, rtol=1e-3):
        """
        Linear Noise Approximation

        :param initial_state: a dictionary of initial state
        :param parameters: a dictionary of parameters
        :param derivative_propensities: a lambda function of the derivative of propensities in x
        :param T0: initial time
        :param Tf: final time
        :param output_time_points: a list of time points for output
        :return: time, state
        """
        mean_output = np.zeros([len(self.species_names), len(output_time_points)])
        cov_output = []

        # set the parameters and derivatives of the propensities
        self.set_parameters(parameters)
        # self.set_derivative_of_propensities_in_state(derivative_propensities)
        if self.propensities_derivative_in_state is None:
            raise Exception('The derivative of propensities is not implemented yet')
        x0_state = [initial_state[species] for species in self.species_names]
        x0_cov = np.zeros( len(self.species_names) * len(self.species_names) )
        x0 = np.concatenate([x0_state, x0_cov])


        # solve the ODE
        sol = solve_ivp(self.drift_term_of_LNA, [T0, Tf], x0, t_eval=output_time_points, rtol=rtol, atol=atol)

        # extract the mean and covariance
        mean_output = sol.y[:len(self.species_names), :]
        cov_output = sol.y[len(self.species_names):, :].T.reshape(-1, len(self.species_names), len(self.species_names))
        # for i in range(self.get_number_of_species()):
        #     mean_output[i,:] = sol.y[i,:]
        # for t in range(len(sol.t)):
        #     cov = sol.y[self.get_number_of_species():, t]
        #     cov_output.append(cov.reshape(len(self.species_names), len(self.species_names)))

        return {"time": sol.t, "mean": mean_output, "covariance": cov_output}


    # set the derivative of propensities
    def set_derivative_of_propensities_in_state(self, propensities_derivative_in_state):
        self.propensities_derivative_in_state = propensities_derivative_in_state

    def _eval_propensities_derivative_in_state(self):
        """
        Evaluate the derivative of propensities in state
        """
        out = []
        for prop in self.propensities_derivative_in_state:
            args = inspect.getfullargspec(prop).args
            input_args = {}
            for a in args:
                if a in self.species_names:
                    input_args[a] = self.state[self.species_ordering[a]]
                else:
                    input_args[a] = self.parameters[a]
            out.append(prop(**input_args))
        return np.array(out)

    # define the ode system LNA
    def drift_term_of_LNA(self, t, x):
        # the notations follow the paper Golightly & Sherlock, Statistics and Computing, 2019

        # x is the state: a vector contain the state first n element and the remaining n^2 element is the covariance matrix
        # split the state
        n = self.get_number_of_species()
        state = x[:n]
        covariance_matrix = x[n:].reshape(n,n)

        #update the state
        self.set_state({species: x[self.species_ordering[species]] for species in self.species_names})

        # deterministic part
        propensities = self._eval()
        deterministic_part = np.dot(self.stoichiometric_matrix, propensities)

        # stochastic part
        derivative_propensities = self._eval_propensities_derivative_in_state()
        F = np.dot(self.stoichiometric_matrix, derivative_propensities)
        diagonal_propensities = np.diag(propensities)
        beta = np.dot(np.dot(self.stoichiometric_matrix, diagonal_propensities), self.stoichiometric_matrix.T)
        stochastic_part = np.dot(F, covariance_matrix) + covariance_matrix.dot(F.T) + beta

        # return the drift term
        return np.concatenate([deterministic_part, stochastic_part.reshape(-1)])


    ###############################################
    # Moment Closure
    # Notations follow the paper David Schnoerr, Sanguinetti, Grima, 2015
    # The ODE is developed for the mean and covariance to avoid scale issues
    ###############################################

    def Moment_Closure(self, initial_state, parameters, T0, Tf, output_time_points, atol=1e-6, rtol=1e-3):
        """
        Moment Closure

        :param initial_state: a dictionary of initial state
        :param parameters: a dictionary of parameters
        :param E_lambda_approximation: a list of lambda functions of the expected propensities
        :param E_lambda_times_X_trans_approximation: a list of lambda functions of the expected propensities
        :param T0: initial time
        :param Tf: final time
        :param output_time_points: a list of time points for output
        :return: time, moments
        """

        # set the parameters and lambda functions
        self.set_parameters(parameters)
        # self.set_propensity_expectation_approximation_via_moments(E_lambda_approximation)
        # self.set_propensity_times_X_trans_expectation_approximation_via_moments(E_lambda_times_X_trans_approximation)
        if self.propensity_expectation_approximation_via_moments is None:
            raise Exception('The approximation of expected propensities is not implemented yet')
        if self.propensity_times_X_trans_expectation_approximation_via_moments is None:
            raise Exception('The approximation of expected propensities times X transposed is not implemented yet')
        x0_state = np.array([initial_state[species] for species in self.species_names])
        # x0_second_moment = np.diag(x0_state**2)
        # x0 = np.concatenate([x0_state, x0_second_moment.reshape(-1)])
        x0_cov = np.zeros([len(self.species_names), len(self.species_names)])
        x0 = np.concatenate([x0_state, x0_cov.reshape(-1)])

        # solve the ODE
        sol = solve_ivp(self.drift_term_of_Moment_Closure, [T0, Tf], x0, t_eval=output_time_points, rtol=rtol, atol=atol)

        # extract the moments
        mean_output = sol.y[:self.get_number_of_species(), :]
        cov_output = sol.y[self.get_number_of_species():, :].T.reshape(-1, len(self.species_names), len(self.species_names))
        # cov_output = cov_output - np.einsum('ij,ik->ijk', mean_output.T, mean_output.T)
        # cov_output = []
        # for t in range(len(sol.t)):
        #     cov = sol.y[self.get_number_of_species():, t]
        #     cov = cov.reshape(len(self.species_names), len(self.species_names)) - np.outer(mean_output[:,t], mean_output[:,t])
        #     cov_output.append(cov)

        return {"time": sol.t, "mean": mean_output, "covariance": cov_output}




    def set_propensity_expectation_approximation_via_moments(self, E_lambda_approximation):
        # check the length of the approximation
        if len(E_lambda_approximation) != self.get_number_of_reactions():
            raise Exception(f'The length of the approximation of expected propensities ({len(E_lambda_approximation)}) is not consistent with the number of reactions ({self.get_number_of_reactions()}).')
        # check the arguments of the lambda functions
        valid_arguments = self.parameters_names + self.first_moment_names + self.second_moment_names
        for p in E_lambda_approximation:
            for arg in inspect.getfullargspec(p).args:
                if arg not in valid_arguments:
                    raise Exception(f'The argument {arg} is not in the list of moments')

        self.propensity_expectation_approximation_via_moments = E_lambda_approximation

    def set_propensity_times_X_trans_expectation_approximation_via_moments(self, E_lambda_times_X_trans_approximation):
        # check the length of the approximation
        if len(E_lambda_times_X_trans_approximation) != self.get_number_of_reactions():
            raise Exception(f'The length of E_lambda_times_X_trans_approximation ({len(E_lambda_times_X_trans_approximation)}) is not consistent with the number of reactions ({self.get_number_of_reactions()}).')
        # check the arguments of the lambda functions
        valid_arguments = self.parameters_names + self.first_moment_names + self.second_moment_names
        for p in E_lambda_times_X_trans_approximation:
            for arg in inspect.getfullargspec(p).args:
                if arg not in valid_arguments:
                    raise Exception(f'The argument {arg} is not in the list of moments')
        self.propensity_times_X_trans_expectation_approximation_via_moments = E_lambda_times_X_trans_approximation

    def set_moment_values_dict(self, mu, sigma):
        """
        Return a dictionary of moments

        :param mu: a vector of the first moments
        :param y: a matrix of the second moments
        :return: a dictionary
        """

        if type(mu) != np.ndarray:
            raise Exception('The first moment is not a numpy array')
        if type(sigma) != np.ndarray:
            raise Exception('The second moment is not a numpy array')
        if mu.shape[0] != len(self.species_names):
            raise Exception('The length of the first moment is not consistent with the number of species')
        if sigma.shape[0] != len(self.species_names) or sigma.shape[1] != len(self.species_names):
            raise Exception('The shape of the covariance is not consistent with the number of species')
        # if np.array_equal(y, y.T) == False:
        #     print(y)
        #     raise Exception('The second moment is not symmetric')

        moment_values = {}
        for i, species in enumerate(self.species_names):
            moment_values['mu_' + species] = mu[i]
            for j, species2 in enumerate(self.species_names):
                moment_values['y_' + species + '_' + species2] = sigma[i,j] + mu[i]*mu[j]

        return moment_values

    def _eval_propensity_expectation_approximation_via_moments(self, mu, sigma):
        all_arg_dict = self.set_moment_values_dict(mu, sigma)
        all_arg_dict.update(self.parameters)
        out = []
        for prop in self.propensity_expectation_approximation_via_moments:
            args = inspect.getfullargspec(prop).args
            input_args = {a: all_arg_dict[a] for a in args}
            out.append(prop(**input_args))

        return np.array(out)

    def _eval_propensity_tims_X_trans_expectation_approximation_via_moments(self, mu, sigma):
        all_arg_dict = self.set_moment_values_dict(mu, sigma)
        all_arg_dict.update(self.parameters)
        out = []
        for prop in self.propensity_times_X_trans_expectation_approximation_via_moments:
            args = inspect.getfullargspec(prop).args
            input_args = {a: all_arg_dict[a] for a in args}
            out.append(prop(**input_args))

        return np.array(out)

    # define the ode system of Moment Closure
    def drift_term_of_Moment_Closure(self, t, x):
        # x is the state: a vector contain the state first n element and the remaining n^2 element is the matrix of the second moments

        # split the state
        n = self.get_number_of_species()
        mu = x[:n]
        sigma = x[n:].reshape(n,n)

        # first moment part
        E_propensities = self._eval_propensity_expectation_approximation_via_moments(mu, sigma)
        first_moment_part = np.dot(self.stoichiometric_matrix, E_propensities)

        # second moment part
        E_prop_times_X_trans = self._eval_propensity_tims_X_trans_expectation_approximation_via_moments(mu, sigma)
        F1 = np.dot(self.stoichiometric_matrix, E_prop_times_X_trans)
        diagonal_propensities = np.diag(E_propensities)
        F2 = self.stoichiometric_matrix.dot(np.dot(diagonal_propensities, self.stoichiometric_matrix.T))
        # F2 = np.zeros([n,n])
        # for j in range(self.get_number_of_reactions()):
        #     F2 = F2 + np.outer(self.stoichiometric_matrix[:,j], self.stoichiometric_matrix[:,j].T) * E_propensities[j]
        second_moment_part = F1 + F1.T + F2 - np.outer(first_moment_part, mu) - np.outer(mu, first_moment_part)

        # return the drift term
        return np.concatenate([first_moment_part, second_moment_part.reshape(-1)])




    ###############################################
    # Plotting
    ###############################################


    # plot the result
    def plot_trajectories(self, time_list, state_output):
        state_output = np.array(state_output)
        rows = math.ceil(self.get_number_of_species()/2)
        columns = 2
        fig, axs = plt.subplots(rows, columns, figsize=(columns*3,rows*3))
        fig.subplots_adjust(wspace=0.5, hspace=1)

        for i, ax in enumerate(axs.flatten()):
            if i >= self.get_number_of_species():
                break
            ax.step(time_list, state_output[:, i], where='post')
            ax.set_xlabel('Time')
            ax.set_ylabel('Copy number of ' + self.species_names[i])

if __name__ == '__main__':
    
    # central dogma

    propensities = [
        lambda kr : kr,
        lambda kp, mrna : kp*mrna,
        lambda gr, mrna : gr*mrna,
        lambda gp, protein : gp*protein  
    ]

    stoichiometric_matrix = np.array([
        [1, 0, -1, 0],
        [0, 1, 0, -1]
    ])

    rn = CRN(
        stoichiometric_matrix=stoichiometric_matrix,
        species_names = ['mrna', 'protein'],
        parameters_names = ['kr', 'kp', 'gr', 'gp'],
        reaction_names= ['mRNA birth', 'Protein birth', 'mRNA degradation', 'Protein degradation'],
        propensities = propensities )

    rn.set_state({'mrna' : 10, 'protein' : 3})
    rn.set_parameters({'kr': 1, 'gr': 2, 'kp' : 1, 'gp' : 1})

    print(rn.state)
    print(rn._eval())
    rn.update(3)
    print(rn.state)
    print(rn._eval())
    rn.update(2)
    print(rn.state)
    print(rn._eval())

    print('')

    # test SSA
    time_list, state_list = rn.SSA({'mrna' : 10, 'protein' : 3}, {'kr': 1, 'gr': 2, 'kp' : 1, 'gp' : 1}, 0, 1)
    print(time_list)
    print(state_list)

    rn.plot_trajectories(time_list, state_list)
