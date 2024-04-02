"""
File for unit tests for AntClust
"""
import AntClust
from importlib import reload
from distance_classes import similarity_1d, similarity_euclid2d
import rules 
import Ant
reload(AntClust)

data = [
    [0.1, [1, 1]],
    [0.2, [1, 2]],
    [0.11, [2, 1]],
    [0.13, [2, 2]],
    [0.9, [8, 9]],
    [0.98, [9, 9]],
    [0.87, [9, 10]],
    [0.7, [10, 9]],
]

f_sim = [similarity_1d(0, 1), similarity_euclid2d(0, 14)]
rule = rules.labroche_rules()
ec = rules.labroche_carvalho()
ant_clust = AntClust.AntClust(f_sim, rule)

# test age
def test_age_penalty():
    ant_clust.fit(data)
    label = ant_clust.get_clusters()
    old_ant = Ant.Ant(data[0],0)
    new_ant = Ant.Ant(data[0],1)
    old_ant.set_age(10)
    old_ant.set_label(label[1])
    new_ant.set_age(2)
    new_ant.set_label(label[0])
    ec.apply_rules(old_ant,new_ant,ant_clust)
    assert(new_ant.get_label() == old_ant.get_label() and new_ant.get_label() == label[1])


# test M+ boost 
def test_m_p_boost():
    val = 0.5
    assert(ec.increase_estimator_variable(val) < ec.increase_estimator_variable(val,alpha=0.4))

# test cluster avg
def test_cluster_avg():
    ant_clust.fit(data)
    label = ant_clust.get_clusters()
    assert(ec.get_mp_cluster_mean(ant_clust,label[0]))

def test_dropout():
    ant_clust_dropout = AntClust.AntClust(f_sim, rule, dropout=True, dropout_prob=0.3)
    ant_clust_dropout.fit(data)

if __name__ == "__main__":
    test_dropout()