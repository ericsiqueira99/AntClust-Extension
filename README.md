# ANTCLUST EXTENSION

This project is a extension of the work done by Winfried Gero Oed and Parisa Memarmoshrefi
[0], whihc is a Python build of ANTCLUST, a clustering algorithm based on the chemical recognition system of ants which was developed by Nicolas Labroche et al. [1]. The idea is to explore the interface and apply it to more data types, as well as test the implementation of bio-inspired methods with the goal of improve AntClust performance.

# New Data types
Developed new similarity functions for the clustering of different data types (categorical, numerical, time series, spatial, textual data) and tested it against classical clustering methods.

# New Ruleset
Rulesets define the behaviour of the AntClust algorithm, and is a delicate combination of condition and consequences. Work was done to delevop new rules and a way to combine different rules dynamically.

# Hybrid Approaches
Bio-inspired methods such as Genetic Algorithm and Particle Swarm Optimization were applied with the goal of searching for a optimal ruleset, and fidnign the best hypeaparameter for the AntClust implementation.

# Collective Ethical Decision Making
An attempt of applying the AntClust algorithm to generate a collective decision for ethics, unfortunately not applied due to lack of ethically encoded dataset.

# Sources
[0] https://gitlab.com/Winnus/antclust

[1] https://www.researchgate.net/publication/2551128_A_New_Clustering_Algorithm_Based_on_the_Chemical_Recognition_System_of_Ants