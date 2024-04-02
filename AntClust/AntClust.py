from Ant import Ant
from concurrent.futures import ThreadPoolExecutor
import random as rng
import numpy as np
import logging

# IMPLEMENTME: make a threaded version of initialize_ants
# IMPLEMENTME: make a threaded version of meet
# threading in the case of antclust is not a trivial problem.
# The similarity function is covered as an object and used by
# __similarity(). If the user definable similarity function is not
# thread save it will fail. Threads will as such not be easy to use
# within sophisticated similarity functions that utilize external
# librarys.
# However if store_computed_similaritys = False and the similarity
# object is thread save ant clust should be as well.
# Then the GIL will be the next point of slow down...
# IMPLEMENTME: inside constructor build a check that ensures
# len(datasetfatures) == len(similarity_function_per_feature)
# maybe use len(ant.genetics) == len(similarity_function_per_feature)?
# IMPLEMENTME:
# Improve logging
# https://docs.python.org/3.8/library/logging.html

# Idea for reassign:
# It would be also possible to make a probailistic method.
# this would be as folows:
# During the nest shrink, which is done before, it can easily be
# filtered out which ants belong to which nest.
# Then compare the ants that have no nest not to all other ants but
# instead to n ants taken from each nest.
# So lets say take randomly 50% int(nest_size*0.5) ants from each nest
# and compare the ant only to those. This reduces the load by 50%.
#


class AntClust:
    """implements the AntClust algorithm logic"""

    def __init__(self,
                 similarity_function_per_feature,
                 ruleset,
                 store_computed_similaritys=True,
                 alpha_ant_meeting_iterations=150,
                 betta_template_init_meetings=0.5,
                 nest_shrink_prop=0.2,
                 nest_removal_prop=0.3,
                 print_status=True,
                 dropout = False,
                 visualization=False,
                 dynamic_template_adaptation=False):
        """
        dataset: the data set as N dimensional array or list

        similarity_function_per_feature: array which holds the similarity
            classes for every feature inside the data set

        ruleset: rule class that implements the rule set

        store_computed_similaritys: will store every computed similarity
        (which is not subject to change) inside a dictionary to prevent
        recomputing. Increases memory load (needs a dictionary of size
        (N-1)*N/2, where N=numdataframes inside the data set)
        but saves computational cost

        alpha_ant_meeting_iterations: based on this the meetings for phase 0
        will be calculated by 0.5*alpha_ant_meeting_iterations*Num_data_points

        betta_template_init_meetings: used to calculate how many other ants
        a single ant will meet to initialize their template.
        betta_template_init_meetings
        is a percent value of alpha_ant_meeting_iterations.
        Template meetings = betta*alpha
        These meetings are meetings with randomly selected ants.

        nest_shrink_prop: this value will be used for the probabilistic nest
        fitness calculation in nest_shrink.

        nest_removal_prop: the value is a floating point percent value giving
        the fitness threshold for deleting a nest with a smaller fitness.
        """
        # variables
        self.print_status = print_status
        self.similarity_functions = similarity_function_per_feature
        self.rules = ruleset
        # save ant objects
        self.ants = []
        # stores the computed labels for each data vector
        self.labels_ = []

        # AntClust parameters

        # alpha used to calculate the meetings per ant in phase 1
        # and template initialization meetings
        self.alpha_ant_meeting_iterations = alpha_ant_meeting_iterations

        # calc template init meetings
        self.template_init_meetings = int(betta_template_init_meetings *
                                          alpha_ant_meeting_iterations)

        # used to calculate the nest deletion property in phase 2
        self.nest_shrink_prop = nest_shrink_prop
        self.nest_removal_prop = nest_removal_prop

        # store computed similarity's
        self.store_computed_similaritys = store_computed_similaritys
        self.saved_similaritys = {}

        # dropout: similar to the idea in machine learning, deactivate some ants during the process to make it more robust
        self.dropout = dropout
        self.dropout_prob = 0.3

        # visualization
        self.visualization = visualization
        self.cluster_evolution = [] 
        self.rule_applied = []

        # dynamic template adaptation: apply a sigmoid to ant's template based on age to encourage ants to meet be more acceptable in 
        # younger age 
        self.dynamic_template_adaptation = dynamic_template_adaptation


    def set_log_level(self, loglevel):
        """
        sets the log level.
        loglevel: string ('DEBUG, INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        LOG_LEVELS = {
            'CRITICAL': logging.CRITICAL,
            'ERROR': logging.ERROR,
            'WARNING': logging.WARNING,
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG
        }

        logging.basicConfig(level=LOG_LEVELS[loglevel])

    def __initialize_ants(self, dataset):
        """
        Initialize_ants with their data frame and return a list of the ants
        """
        # initialize ants
        ant_list = []
        for i in range(len(dataset)):
            ant_list.append(Ant(dataset[i], i))

        # initialize template of the ants
        # each ant will meet with template_init_meetings other ants and update
        # its threshold
        # the threshold update can be achieved by calling the acceptance
        # function with the two ants
        for ant_i in range(len(ant_list)):
            for j in range(self.template_init_meetings):
                # ensure ant does not meet itself
                rnd_ant = rng.randint(0, len(ant_list) - 1)
                while ant_i == rnd_ant:
                    rnd_ant = rng.randint(0, len(ant_list) - 1)

                # let teh ants meet
                self.__acceptance(ant_list[ant_i], ant_list[rnd_ant])

        return ant_list

    def __similarity(self, ant_i, ant_j):
        """Calculate the distance for every feature inside ant genetics"""

        # if enabled, check if similarity was already computed
        # compute the unique key, i.e. smaller index first
        if self.store_computed_similaritys:
            if ant_i.index < ant_j.index:
                dict_key = str(ant_i.index) + ':' + \
                    str(ant_j.index)
            else:
                dict_key = str(ant_j.index) + ':' + \
                    str(ant_i.index)

            # check if key inside dict
            if dict_key in self.saved_similaritys:
                return self.saved_similaritys[dict_key]

        # if no similarity's will be stored or it is not computed already
        # compute it
        sim = 0

        # calculate and add up sim for every feature with its sim_function
        for i in range(len(self.similarity_functions)):
            # extract data vectors
            d_i = ant_i.gene[i]
            d_j = ant_j.gene[i]

            # calculate similarity
            sim += self.similarity_functions[i].similarity(d_i, d_j)

        # normalize between the feature i.e between zero and one
        sim = sim / len(self.similarity_functions)

        # store sim if enabled
        if self.store_computed_similaritys:
            self.saved_similaritys[dict_key] = sim

        return sim

    def __acceptance(self, ant_i, ant_j):
        """Evaluates if ants accept each other or not"""

        # get similarity
        similarity = self.__similarity(ant_i, ant_j)

        # update template of the ants
        # FIXME: should this be updated
        # before or after the acceptance check below?
        ant_i.update_template(similarity)
        ant_j.update_template(similarity)
        ant_i_template = ant_i.template
        ant_j_template = ant_j.template
        # check the similarity against ants templates and return acceptance
        
        # if dynamic template adaptation is set to true, younger ants will have reduced template
        if self.dynamic_template_adaptation:
            ant_j_template = ant_j_template * self.sigmoid(ant_i.get_age())
            ant_i_template = ant_j_template * self.sigmoid(ant_j.get_age())

        if (similarity > ant_i_template) and (similarity > ant_j_template):
            return True
        return False

    def get_acceptance(self, ant_i, ant_j):
        """
        Returns true if the ants accept each other false if not.
        """
        return self.__acceptance(ant_i, ant_j)

    def __meet(self):
        """let ants meet"""
        number_of_ants = len(self.ants)
        iterations_left = self.meeting_iterations
        while iterations_left:
            if self.dropout and rng.random() >= self.dropout_prob and iterations_left <= self.meeting_iterations//2:
                drop_count = rng.randint(number_of_ants//8, number_of_ants//6)
                drop_indices = [rng.randint(0, number_of_ants - 1) for _ in range(drop_count)]
                for index in drop_indices:
                    # drop ant
                    self.ants[index].reset()
                if self.print_status:
                    print("----DROPOUT----")
                    print(f"Ants reseted: {drop_indices}")
            # print meetings?
            if self.print_status:
                if iterations_left % (int(self.meeting_iterations) * 0.1) == 0:
                    print('left meetings ', iterations_left, '/',
                          self.meeting_iterations)
            iterations_left = iterations_left - 1
            ant0_index = rng.randint(0, number_of_ants - 1)
            ant1_index = rng.randint(0, number_of_ants - 1)
            while ant0_index == ant1_index:
                ant1_index = rng.randint(0, number_of_ants - 1)

            # apply rules to ants
            ant_i = self.ants[ant0_index]
            ant_j = self.ants[ant1_index]
            if self.visualization:
                rule = self.rules.apply_rules(ant_i, ant_j, self)
                self.rule_applied.append(rule)
                self.__cluster_label_assignment()
                self.cluster_evolution.append(self.get_clusters())
            else:
                self.rules.apply_rules(ant_i, ant_j, self)

    def __nest_shrink(self):
        """
        Probabilistic nest shrink method
        P_del(nest) = (1 - v) * mean(M⁺_nest) + v * N_nest/N
        Where P_del(nest) is the probability to delete the nest labeled "nest"
        mean(M⁺_nest) the mean of the integration prediction of the ants inside
        the nest, N_nest the number of ants in that particular nest and
        N the number of all ants.
        v=0.2 was tested by Labroche et al. and argued for the best partitioning
        This function also normalize cluster numbers to an ascending row,
        necessary as other wise every cluster has a random label and this is
        not readable.
        """

        # save everything: dict['label'] = [sum_M⁺, [ants]]
        label_dict = {}

        # find all nests and their propertys
        for ant in self.ants:
            if ant.label != -1:
                if ant.label in label_dict:
                    # update M⁺
                    label_dict[ant.label][0] += ant.m_p
                    # update sum_ants
                    label_dict[ant.label][1].append(ant)

                else:
                    # create new label
                    label_dict[ant.label] = [ant.m_p, [ant]]

        # calculate P_del(nest)
        P_del_dict = {}

        for label in label_dict:
            P_del_dict[label] = (1 - self.nest_shrink_prop) * \
                (label_dict[label][0]/len(label_dict[label][1]))
            + self.nest_shrink_prop * \
                (len(label_dict[label][1])/len(self.ants))

        # delete unintegrated nest's
        for label in label_dict:
            # delete the nest?
            if P_del_dict[label] < self.nest_removal_prop:
                # all ants inside the nest loosing their label
                for ant in label_dict[label][1]:
                    ant.label = -1
                # remove nest
                del P_del_dict[label]

        # normalize cluster numbers to an ascending row
        label_counter = 0
        # go trough all the left nests and assign new labels
        for label in P_del_dict:
            # assign the new label to all ants
            for ant in label_dict[label][1]:
                ant.label = label_counter
            # update counter
            label_counter += 1

        # Debug Output
        logging.debug('PDelDict' + str(P_del_dict))

        dic2 = dict(sorted(P_del_dict.items(), key=lambda x: x[1]))
        logging.debug('Sorted PropbDeletion dictionary' + str(dic2))

    def __cluster_label_assignment(self):
        """makes an array holding the label, by index, for each data vector"""
        # make cluster array
        self.labels_ = np.zeros(len(self.ants), dtype=int)
        for i in range(len(self.ants)):
            self.labels_[i] = self.ants[i].label

    def get_clusters(self):
        """Returns the label, by index, for each data vector"""
        return self.labels_

    def __reassign(self):
        """reassign all ants that have no nest to most similar nest"""
        # Steps:
        # go trough all ants and find the ones with no labe = nest
        # compare this ants to all other ants and find the most similar one
        # assign the ant to the same label as the most similar ant

        # do we find an ant with no label?
        for ant0_index in range(len(self.ants)):
            if self.ants[ant0_index].label == -1:  # ant has no label
                # compute max similarity to other ants
                max_sim = -1
                ant_max_sim_index = 0
                for ant1_index in range(len(self.ants)):
                    if ant0_index != ant1_index:  # ensure no self reference
                        if self.ants[ant1_index].get_label() != -1:  # e. labe
                            sim = self.__similarity(self.ants[ant0_index],
                                                    self.ants[ant1_index])
                            if sim > max_sim:
                                max_sim = sim
                                ant_max_sim_index = ant1_index
                # most similar ant found, assign her label to the no label Ant
                self.ants[ant0_index].set_label(
                    self.ants[ant_max_sim_index].get_label())

                logging.debug('lable: ant' + str(ant0_index) + ' <- ant_' +
                              str(ant_max_sim_index))

    def __find_clusters(self):
        """will run AntClus algorithm on the given dataset"""
        # Steps:
        # simulate/meet ants with iterations=n
        # delete nests i.e. call nest_shrink()
        # re-assign ants with no nest i.e. call reassign()

        # meetings
        if self.print_status:
            print('AntClust: phase 1 of 3 -> meeting ants')
        self.__meet()

        # delet nests with P x n (P << 1)
        if self.print_status:
            print('AntClust: phase 2 of 3 -> shrink nests')
        self.__nest_shrink()

        # reassign ants and make sure every ant has a colony
        if self.print_status:
            print('AntClust: phase 3 of 3 -> reassign ants')
        self.__reassign()

        # save clusters to a index readable array
        self.__cluster_label_assignment()

    def fit(self, X):
        """Find Clusters in data set X"""
        
        # initialize all the ants with the given data set
        self.ants = self.__initialize_ants(X)

        # number of random meetings per ant for phase 1
        self.meeting_iterations = int(0.5 * self.alpha_ant_meeting_iterations *
                                      len(self.ants))
        # run phases
        self.__find_clusters()
        return self.cluster_evolution, self.rule_applied
