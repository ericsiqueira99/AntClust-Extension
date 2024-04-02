import sys
sys.path.append('../AntClust')
import rules

class dynamic_rules(rules.rule_interface):
    def __init__(self, rule_list):
        # list of all created labels
        self.labels = []

        # label counter, will be increased for every new label creation
        self.label_counter = 0

        # list of rules for dynamic rule creation
        self.rule_list = rule_list

    def apply_rules(self, ant_i, ant_j, AntClust):
        # get the ants acceptance
        acceptance = AntClust.get_acceptance(ant_i, ant_j)
        for rule in self.rule_list:
            rule_method = getattr(self, rule)
            rule_applied = rule_method(ant_i, ant_j, AntClust, acceptance)
            if rule_applied != None:
                return rule_applied
        return "R6"

    def increase_estimator_variable(self, x, alpha=0.2):
        return (1 - alpha) * x + alpha

    def decrease_estimator_variable(self, x, alpha=0.2):
        return (1 - alpha) * x
    
    def get_mp_cluster_mean(self, AntClust, label):
        cluster_sum = 0
        cluster_len = 0
        for a in AntClust.ants:
            if a.get_label() == label:
                cluster_sum += a.get_m_p()
                cluster_len += 1 
        return cluster_sum/cluster_len
    

    """
    CONDITIONS --------------------------------------------------
    """
    # Both ants have no label and accept
    def R1_condition(self,ant_i, ant_j, AntClust, acceptance):
        return (ant_i.get_label() == -1) and (ant_j.get_label() == -1) and acceptance

    # One ant has no label the other does and accept
    def R2_condition(self,ant_i, ant_j, AntClust, acceptance):
        return ((ant_i.get_label() == -1) and (ant_j.get_label() != -1) and acceptance) or ((ant_j.get_label() == -1) and (ant_i.get_label() != -1) and acceptance)

    # Both ants have same label and accept
    def R3_condition(self,ant_i, ant_j, AntClust, acceptance):
        return (ant_i.get_label() == ant_j.get_label()) and (ant_i.get_label() != -1) and (ant_j.get_label() != -1) and acceptance

    # Both ants have same label and reject
    def R4_condition(self,ant_i, ant_j, AntClust, acceptance):
        return (ant_i.get_label() == ant_j.get_label()) and (ant_i.get_label() != -1) and (ant_j.get_label() != -1) and acceptance == False

    # Ants have different label and reject.
    def R5_condition(self,ant_i, ant_j, AntClust, acceptance):
        return (ant_i.get_label() != ant_j.get_label()) and acceptance

    """
    CONSEQUENCES --------------------------------------------------
    """

    # create new label and assign both ants
    def R1(self,ant_i, ant_j, AntClust,acceptance):
        if self.R1_condition(ant_i, ant_j, AntClust, acceptance):
            new_label = self.label_counter
            self.labels.append(new_label)
            self.label_counter += 1  # update label counter
            # assign label to ants
            ant_i.set_label(new_label)
            ant_j.set_label(new_label)
            return "R1"
    
    # assign label from labeled ant to unlebeled ant
    def R2(self,ant_i, ant_j, AntClust,acceptance):
        if self.R2_condition(ant_i, ant_j, AntClust, acceptance):
            if ant_i.get_label() != -1:
                ant_j.set_label(ant_i.get_label())
            if ant_j.get_label() != -1:
                ant_i.set_label(ant_j.get_label())
            return "R2"
    
    # Increase ants estimators M and M+
    def R3(self,ant_i, ant_j, AntClust,acceptance):
        if self.R3_condition(ant_i, ant_j, AntClust, acceptance):
        # increase their meeting estimators m and m_p
            # m:
            ant_i.set_m(self.increase_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.increase_estimator_variable(ant_j.get_m()))

            # m_p
            ant_i.set_m_p(self.increase_estimator_variable(ant_i.get_m_p()))
            ant_j.set_m_p(self.increase_estimator_variable(ant_j.get_m_p()))
            return "R3"
    
    # Increase ants estimators M and if one of the ants has 
    # M+ higher than clsuter avarage, boost the other ant M+ increase
    def R3_BOOST(self,ant_i, ant_j, AntClust,acceptance):
        if self.R3_condition(ant_i, ant_j, AntClust, acceptance):
            mp_cluster_mean = self.get_mp_cluster_mean(AntClust,ant_j.get_label())
            if ant_i.get_m_p() >= mp_cluster_mean and ant_j.get_m_p() < mp_cluster_mean:
                # boost ant_j M+
                # m_p
                ant_i.set_m_p(self.increase_estimator_variable(ant_i.get_m_p()))
                ant_j.set_m_p(self.increase_estimator_variable(ant_j.get_m_p(),alpha=0.4))
            elif ant_j.get_m_p() >= mp_cluster_mean and ant_i.get_m_p() < mp_cluster_mean:
                # boost ant_i M+
                # m_p
                ant_i.set_m_p(self.increase_estimator_variable(ant_i.get_m_p(),alpha=0.4))
                ant_j.set_m_p(self.increase_estimator_variable(ant_j.get_m_p()))
            else:
                # m_p
                ant_i.set_m_p(self.increase_estimator_variable(ant_i.get_m_p()))
                ant_j.set_m_p(self.increase_estimator_variable(ant_j.get_m_p()))
            # m:
            ant_i.set_m(self.increase_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.increase_estimator_variable(ant_j.get_m()))
            return "R3_BOOST"

    # Increase M of both ants, decreasse M+, ant with smaller M+ loses label
    def R4(self,ant_i, ant_j, AntClust,acceptance):
        if self.R4_condition(ant_i, ant_j, AntClust, acceptance):
            # increase m_j,i and decrease m_p_i,j
            # m:
            ant_i.set_m(self.increase_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.increase_estimator_variable(ant_j.get_m()))

            # m_p:
            ant_i.set_m_p(self.decrease_estimator_variable(ant_i.get_m_p()))
            ant_j.set_m_p(self.decrease_estimator_variable(ant_j.get_m_p()))

            # The ant x (x=i, x=j) which posseses the worst integration
            # in the nest loses its label
            # m_p is the estimator how well accepted the ant is inside her nest
            m_p_i = ant_i.get_m_p()
            m_p_j = ant_j.get_m_p()

            # ant_i looses its label i.e. is not as good accepted as ant_j
            if m_p_i <= m_p_j:
                ant_i.set_label(-1)
                ant_i.set_m(0)
                ant_i.set_m_p(0)
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))

            # ant_j looses its label i.e. is not as good accepted as ant_i
            elif m_p_j < m_p_i:
                ant_j.set_label(-1)
                ant_j.set_m(0)
                ant_j.set_m_p(0)
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))
            return "R4"
        
    # Increase M, decrease M+, younger ant loses label
    def R4_YOUNG(self,ant_i, ant_j, AntClust,acceptance):
        if self.R4_condition(ant_i, ant_j, AntClust, acceptance):
            # increase m_j,i and decrease m_p_i,j
            # m:
            ant_i.set_m(self.increase_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.increase_estimator_variable(ant_j.get_m()))

            # m_p:
            ant_i.set_m_p(self.decrease_estimator_variable(ant_i.get_m_p()))
            ant_j.set_m_p(self.decrease_estimator_variable(ant_j.get_m_p()))

            # The youngest ant x (x=i, x=j) loses its label
            # m_p is the estimator how well accepted the ant is inside her nest
            age_i = ant_i.get_age()
            age_j = ant_j.get_age()

            # ant_i looses its label i.e. is not as good accepted as ant_j
            if age_i <= age_j:
                ant_i.set_label(-1)
                ant_i.set_m(0)
                ant_i.set_m_p(0)
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))

            # ant_j looses its label i.e. is not as good accepted as ant_i
            elif age_j < age_i:
                ant_j.set_label(-1)
                ant_j.set_m(0)
                ant_j.set_m_p(0)
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))
            return "R4_YOUNG"

    # Increase M, decrease M+, ant with higher stability loses label
    def R4_STABILITY(self,ant_i, ant_j, AntClust,acceptance):
        if self.R4_condition(ant_i, ant_j, AntClust, acceptance):
            # increase m_j,i and decrease m_p_i,j
            # m:
            ant_i.set_m(self.increase_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.increase_estimator_variable(ant_j.get_m()))

            # m_p:
            ant_i.set_m_p(self.decrease_estimator_variable(ant_i.get_m_p()))
            ant_j.set_m_p(self.decrease_estimator_variable(ant_j.get_m_p()))

            # The oldest ant x (x=i, x=j) loses its label
            # m_p is the estimator how well accepted the ant is inside her nest
            stability_i = ant_i.get_stability()
            stability_j = ant_j.get_stability()

            # ant_i looses its label i.e. is not as good accepted as ant_j
            if stability_i >= stability_j:
                ant_i.set_label(-1)
                ant_i.set_m(0)
                ant_i.set_m_p(0)
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))


            # ant_j looses its label i.e. is not as good accepted as ant_i
            elif stability_j > stability_i:
                ant_j.set_label(-1)
                ant_j.set_m(0)
                ant_j.set_m_p(0)
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))
            return "R4_STABILITY"
        
    # Increase M, decrease M+, older ant loses label
    def R4_OLD(self,ant_i, ant_j, AntClust,acceptance):
        if self.R4_condition(ant_i, ant_j, AntClust, acceptance):
            # increase m_j,i and decrease m_p_i,j
            # m:
            ant_i.set_m(self.increase_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.increase_estimator_variable(ant_j.get_m()))

            # m_p:
            ant_i.set_m_p(self.decrease_estimator_variable(ant_i.get_m_p()))
            ant_j.set_m_p(self.decrease_estimator_variable(ant_j.get_m_p()))

            # The oldest ant x (x=i, x=j) loses its label
            # m_p is the estimator how well accepted the ant is inside her nest
            age_i = ant_i.get_age()
            age_j = ant_j.get_age()

            # ant_i looses its label i.e. is not as good accepted as ant_j
            if age_i >= age_j:
                ant_i.set_label(-1)
                ant_i.set_m(0)
                ant_i.set_m_p(0)
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))

            # ant_j looses its label i.e. is not as good accepted as ant_i
            elif age_j > age_i:
                ant_j.set_label(-1)
                ant_j.set_m(0)
                ant_j.set_m_p(0)
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))
            return "R4_OLD"
        
    # Decrease M estimator and assign ant with smaller M to the other ant's cluster
    def R5(self,ant_i, ant_j, AntClust,acceptance):
        if self.R5_condition(ant_i, ant_j, AntClust, acceptance):
            # m:
            ant_i.set_m(self.decrease_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.decrease_estimator_variable(ant_j.get_m()))

            # M:
            if ant_i.get_m() < ant_j.get_m():
                # ant_i is inside the smaller cluster and will be absorbed
                ant_i.set_label(ant_j.get_label())
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))

            else:
                # ant_j is inside the smaller cluster and will be absorbed
                ant_j.set_label(ant_i.get_label())
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))
            return "R5"

    # Decrease M estimator and assign younger ant with to the other ant's cluster
    def R5_YOUNG(self,ant_i, ant_j, AntClust,acceptance):
        if self.R5_condition(ant_i, ant_j, AntClust, acceptance):
            # m:
            ant_i.set_m(self.decrease_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.decrease_estimator_variable(ant_j.get_m()))

            # M:
            if ant_i.get_age() < ant_j.get_age():
                # ant_i is inside the smaller cluster and will be absorbed
                ant_i.set_label(ant_j.get_label())
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))
            else:
                # ant_j is inside the smaller cluster and will be absorbed
                ant_j.set_label(ant_i.get_label())
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))
            return "R5_OLD"

    # Decrease M estimator and assign older ant with to the other ant's cluster
    def R5_OLD(self,ant_i, ant_j, AntClust,acceptance):
        if self.R5_condition(ant_i, ant_j, AntClust, acceptance):
            # m:
            ant_i.set_m(self.decrease_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.decrease_estimator_variable(ant_j.get_m()))

            # M:
            if ant_i.get_age() > ant_j.get_age():
                # ant_i is inside the smaller cluster and will be absorbed
                ant_i.set_label(ant_j.get_label())
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))
            else:
                # ant_j is inside the smaller cluster and will be absorbed
                ant_j.set_label(ant_i.get_label())
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))
            return "R5_OLD"
        
    # If both ant's M+ estimator are higher than average, merge two clusters, otherwise apply R5
    def R5_MERGE(self,ant_i, ant_j, AntClust,acceptance):
        if self.R5_condition(ant_i, ant_j, AntClust, acceptance):
            avg_m_p_i = self.get_mp_cluster_mean(AntClust,ant_i.get_label())
            avg_m_p_j = self.get_mp_cluster_mean(AntClust,ant_j.get_label())
            if (ant_i.get_m_p() >= avg_m_p_i) and (ant_j.get_m_p() >= avg_m_p_j):
                # reduce cluster numbers
                # remove label from self.labels
                self.labels = [num for num in self.labels if num != ant_j.get_label()]
                self.label_counter -= 1  # update label counter

                # merge clusters: set all ant with label from ant_j to label of ant_i
                for ant in AntClust.ants:
                    if ant.get_label() == ant_j.get_label():
                        ant.set_label(ant_i.get_label())
                        ant.set_stability(self.decrease_estimator_variable(ant.get_stability()))  
            return "R5_MERGE"
        
    # If both ant's M+ estimator are lower than average, create new cluster with the two ants
    def R5_NEW(self,ant_i, ant_j, AntClust,acceptance):
        if self.R5_condition(ant_i, ant_j, AntClust, acceptance):
            avg_m_p_i = self.get_mp_cluster_mean(AntClust,ant_i.get_label())
            avg_m_p_j = self.get_mp_cluster_mean(AntClust,ant_j.get_label())
            if (ant_i.get_m_p() < avg_m_p_i) and (ant_j.get_m_p() < avg_m_p_j):
                # create new clusters: set both ants with new label
                new_label = self.label_counter
                self.labels.append(new_label)
                self.label_counter += 1  # update label counter
                ant_i.set_label(new_label)
                ant_j.set_label(new_label)
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))  
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))  
            return "R5_NEW"


    # Decrease M estimator and assign ant with lower stability to the other ant's cluster
    def R5_STABILITY(self,ant_i, ant_j, AntClust,acceptance):
        if self.R5_condition(ant_i, ant_j, AntClust, acceptance):
            # m:
            ant_i.set_m(self.decrease_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.decrease_estimator_variable(ant_j.get_m()))

            # M:
            if ant_i.get_stability() < ant_j.get_stability():
                # ant_i is inside the smaller cluster and will be absorbed
                ant_i.set_label(ant_j.get_label())
                ant_i.set_stability(self.decrease_estimator_variable(ant_i.get_stability()))  

            else:
                # ant_j is inside the smaller cluster and will be absorbed
                ant_j.set_label(ant_i.get_label())         
                ant_j.set_stability(self.decrease_estimator_variable(ant_j.get_stability()))  
            return "R5_STABILITY"
