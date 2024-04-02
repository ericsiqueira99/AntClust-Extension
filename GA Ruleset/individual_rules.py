"""
CONDITIONS
"""
# Both ants have no label and accept
def R1_condition(ant_i,ant_j,acceptance):
    return (ant_i.get_label() == -1) and (ant_j.get_label() == -1) and acceptance

# One ant has no label the other does and accept
def R2_condition(ant_i,ant_j,acceptance):
    return ((ant_i.get_label() == -1) and (ant_j.get_label() != -1) and acceptance) or ((ant_j.get_label() == -1) and (ant_i.get_label() != -1) and acceptance)

# Both ants have same label and accept
def R3_condition(ant_i,ant_j,acceptance):
    return (ant_i.get_label() == ant_j.get_label()) and (ant_i.get_label() != -1) and (ant_j.get_label() != -1) and acceptance

# Both ants have same label and reject
def R4_condition(ant_i,ant_j,acceptance):
    return (ant_i.get_label() == ant_j.get_label()) and (ant_i.get_label() != -1) and (ant_j.get_label() != -1) and acceptance == False

# Ants have different label and reject.
def R5_condition(ant_i,ant_j,acceptance):
    return (ant_i.get_label() != ant_j.get_label()) and acceptance

"""
CONSEQUENCE
"""

# create new label and assign both ants
def R1():
    new_label = self.label_counter
    self.labels.append(new_label)
    self.label_counter += 1  # update label counter

    # assign label to ants
    ant_i.set_label(new_label)
    ant_j.set_label(new_label)