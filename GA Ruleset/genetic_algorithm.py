import sys
from collections import defaultdict
import random
import numpy as np 
from itertools import product    
sys.path.append('../AntClust')

# Define a defaultdict with a list as the default value
class GA:    
    RULESETS = {"R1_condition": ["R1"],
                "R2_condition": ["R2"],
                "R3_condition": ["R3","R3_BOOST"],
                "R4_condition": ["R4","R4_YOUNG","R4_OLD","R4_STABILITY"],
                "R5_condition": ["R5","R5_YOUNG","R5_OLD","R5_MERGE","R5_NEW","R5_STABILITY"]}

    def __init__(self, fitness_function, pop_size=10, generations=20, mutation_rate = 0.2, criteria=0.8):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = self.generate_individuals(pop_size)
        self.fitness_function = fitness_function
        self.criteria = criteria

    # def generate_individuals(self, size):
    #     ind_list = []
    #     for _ in range(size):
    #         result = []
    #         for key in self.RULESETS.keys():
    #             result.append((key, random.choice(self.RULESETS[key])))
    #         ind_list.append(result)
    #     return ind_list
    
    def generate_individuals(self, size):
        all_combinations = list(product(*self.RULESETS.values()))
        size = min(size,len(all_combinations))
        indices = random.sample(range(len(all_combinations)), size)
        return [(list(all_combinations[i]),random.choice([True, False]),random.choice([True, False])) for i in indices]
    
    # Tournament selection of parents
    def tournament_selection(self, fitness_values, tournament_size=3):
        selected = []
        for _ in range(2):  # Select 2 parents
            tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            selected.append(self.population[tournament_indices[np.argmax(tournament_fitness)]])
        return selected
    
    # Perform crossover between two individuals
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(2, len(parent1))
        child = (parent1[0][:crossover_point] + parent2[0][crossover_point:], random.choice([parent1[1],parent2[1]]),random.choice([parent1[2],parent2[2]]))
        return child
    
    # Mutate child based on mutation rate
    def mutation(self, child):
        for i,key in enumerate(self.RULESETS.keys()):
            if np.random.rand() < self.mutation_rate:
                child[0][i] = random.choice(self.RULESETS[key])
        return child
    
    def print_individual(self, individual):
        print(f"{[i for i in individual[0]]}, Dropout: {individual[1]}, Dynamic template: {individual[2]}")

    def print_population(self):
        for ind in self.population:
            self.print_individual(ind)
    # Run the algorithm
    def run_genetic_algorithm(self):
        best_solution = None
        best_fitness = 0
        top_fitness = []
        patience = 0
        for i in range(self.generations):
            print(f"Generation {i}")
            fitness_values = [self.fitness_function(ind[0],ind[1],ind[2]) for ind in self.population]
            for i in range(self.pop_size // 2):
                parent1, parent2 = self.tournament_selection(fitness_values)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                self.population[i] = child
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > best_fitness:
                best_solution = self.population[best_idx]
                best_fitness = fitness_values[best_idx]
            top_fitness.append(best_fitness)
            # if best_fitness >= self.criteria or patience > 3:
            #     break
            # if len(top_fitness) > 2:
            #     if top_fitness[-1] == top_fitness[-2]:
            #         patience +=1
            #     else:
            #         patience = 0
        return top_fitness, best_solution, best_fitness


# def fun():
#     pass

# test = GA(fun)
# ind = test.population[0]
# print(ind[0])
# print(test.mutation(ind))