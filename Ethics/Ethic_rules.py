# Informal rule Interface
# The function apply_rules will need to be implemented 
# and will be called by AntClust to perform the rules.
class rule_interface:
    def apply_rules(self, ant_i, ant_j, AntClust):
        """
        Will apply all rules the rules inside the ruleset of this class.
        ant_i: one ant
        ant_j: another ant
        AntClust: reference to the AntClust instance

        You will have acess to the following things:
            - Ant attributes:
                - get_label(): which is thhe label for the cluster this ant
                  belongs to. If not yet assigned it will be -1
                - get_m(): m is an estimator which is reflecting how sucessfull
                  this ant is during its meetings with other ants, is an
                  estimator of the size of the nest
                - get_m_p: m_p measures how well accpeted this ant is inside
                   his nest

            - AntClust attributes:
                - get_acceptance(ant_i, ant_j): true or false depending if
                  the ant accept each other or not
        """
        raise NotImplementedError

class ethical_rules(rule_interface):
    def __init__(self):
        # list of all created labels
        self.labels = []

        # label counter, will be increased for every new label creation
        self.label_counter = 0

    def apply_rules(self, ant_i, ant_j, AntClust):
        # get the ants acceptance
        acceptance = AntClust.get_acceptance(ant_i, ant_j)
        # R1
        if acceptance:
            ant_i.set_m(self.increase_estimator_variable(ant_i.get_m()))
            ant_j.set_m(self.increase_estimator_variable(ant_j.get_m()))
        # R2
        else:
            if ant_i.get_m() > ant_j.get_m():
                ant_j.set_label(ant_i.get_label())
                ant_j.set_m(self.decrease_estimator_variable(ant_j.get_m()))
            else:
                ant_j.set_label(ant_i.get_label())
                ant_j.set_m(self.decrease_estimator_variable(ant_j.get_m()))

    def increase_estimator_variable(self, x, alpha=0.2):
        return (1 - alpha) * x + alpha

    def decrease_estimator_variable(self, x, alpha=0.2):
        return (1 - alpha) * x
