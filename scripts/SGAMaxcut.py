import math
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randrange, uniform, randint
import itertools
import time
import scipy.stats
import numpy.random as npr
# import numpy.random.Generator as rgen

# Quantum statics
Sqrt_2 = math.sqrt(2.0)
Had = np.array([[1, 1], [1, -1]]) / Sqrt_2
Zero = np.array([[1], [0]])
Plus = np.matmul(Had, Zero)

rng = npr.default_rng() 

class SGAAlgParams:
    'Instance variables for running QGA'

    def __init__(self, pop_size, genome_len, max_gens, crossover_rate, mutations, debug = False):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.mutations = mutations
        self.debug = debug
        self.max_gens = max_gens
        self.crossover_rate = crossover_rate


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# Randomly put in 0/1 states
def init_pop(params):
    # data[individual][gene]
    init_pop = np.empty([params.pop_size, params.genome_len], dtype=int)
    for indiv in range(params.pop_size):
        for gene in range(params.genome_len):
            if uniform(0,1) < 0.5:
                init_pop[indiv][gene] = 0
            else:
                init_pop[indiv][gene] = 1

    return init_pop


# Returns (fits[], idx_best_fit)
def fitness_evaluation(population, G):
    fits = np.array([individual_fit(chromosome, G) for chromosome in population])
    argmx = np.argmax(fits)
    return (fits, argmx)


# Assumes G's edges start at 0
def individual_fit(chromosome, G):
    if len(chromosome) != len(G):
        raise ValueError('Chromosome length is {0} but graph has {1} nodes'.format(len(chromosome), G.order()))

    cut_val = 0
    cut = []

    for idx_i in G:
        for idx_j in G[idx_i]:
            if idx_j > idx_i: # avoid repeats
                # print(idx_i, idx_j)
                if chromosome[idx_i] != chromosome[idx_j]:
                    # print('Yessir!')
                    cut_val += G[idx_i][idx_j]['weight']
                    cut.append((idx_i, idx_j))
    return cut_val


def mutate(pop, rate_pop, rate_individual, best_idx):   
    chrom_len = len(pop[0][0])
    mutated = np.zeros([len(pop), chrom_len], dtype=int)
    mutated[best_idx] = pop[best_idx][0]

    for idx_chrom in range(len(pop)):
        if idx_chrom != best_idx:
            if uniform(0, 1) <= rate_pop:
                for idx_gene in range(chrom_len):
                    gene = pop[idx_chrom][0][idx_gene]
                    if uniform(0, 1) <= rate_individual:
                        mutated[idx_chrom][idx_gene] = 1 - gene # negation
                    else:
                        mutated[idx_chrom][idx_gene] = gene
            else:
                mutated[idx_chrom] = pop[idx_chrom][0]

    return mutated
  

def print_gen_stats(gen, fitnesses):
    mean = np.mean(fitnesses[0])
    std = np.std(fitnesses[0])
    print('{0}, {1}, {2}, {3}, {4}'.format(gen, fitnesses[0][fitnesses[1]], mean, std, std/math.sqrt(mean)))


def wheel_p_selection(pop, fitnesses):
    fitness_total = sum(abs(fitnesses))
    popsize = len(pop)
    selection_probs = [abs(fit) / fitness_total for fit in fitnesses]

    return [(pop[idx], fitnesses[idx]) for idx in rng.choice(popsize, popsize, p=selection_probs)]

#########################################################
# TOURNAMENT SELECTION OPERATOR                         #
#########################################################
def select_p_tournament(fits):
    u1 = randint(0, len(fits) - 1)
    u2 = u1
    while (u2 == u1):
        u2 = randint(0, len(fits) -1)
    
    return u1 if fits[u1][1] >= fits[u2][1] else u2

########################################################
# ONE-POINT CROSSOVER                                  #
########################################################  
# crossover_rate: setup crossover rate
def mating(pop, fitnesses, crossover_rate):
    crossover_point = 0
    chrom_len = len(pop[0])
    parent1 = select_p_tournament(pop)
    parent2 = select_p_tournament(pop)
    if uniform(0, 1) <= crossover_rate:
        crossover_point = randint(0, chrom_len)
    
    child = []

    for j in range(chrom_len):
        if j <= crossover_point:
            child.append(pop[parent1][j])
        else:
            child.append(pop[parent2][j])
    
    return child
  
def crossover(parents, fitnesses, crossover_rate):
    kids = [mating(parents, fitnesses, crossover_rate) for _ in range(len(parents) - 1)]
    kids.append(parents[np.argmax(fitnesses)])
    return kids


def fit_avg(pop, G):
    fits = [individual_fit(chromosome, G) for chromosome in pop]
    return np.mean(fits)


def plot(stats):
    stats = [(np.mean(fitnesses[0]), np.std(fitnesses[0]), fitnesses[0][fitnesses[1]]) for fitnesses in stats]

    with open('sga-single-run.csv', 'a') as csvfile:
        for stat in stats:
            # [i]: max, avg
            csvfile.write('{},{}\n'.format(stat[2], stat[0]))   

    
    plt.subplot(1,2,1)
    plt.plot(range(len(stats)), [x[0] for x in stats])
    plt.title('means')

    # plt.subplot(1,2,2)
    # plt.plot(range(len(stats)), [x[1] for x in stats])
    # plt.title('std.s')

    plt.subplot(1,2,2)
    plt.plot(range(len(stats)), [x[2] for x in stats])
    plt.title('maxess')

    plt.show()    

# graph: graph/network
# params: GA-specific params, such as mutation rates
# DEFINITELY OVERWRITING POPULATIONS IN EACH GENERATION!
# 
# out: max-cut (approx)
def run_sga(graph, params):
    generation = 0
    pop = init_pop(params)
    fitnesses = [fitness_evaluation(pop, graph)]

    if params.debug:
        print('gen, max, avg, std, SE')

    while (generation < params.max_gens):
        # print('####Running gen {0}'.format(generation))

        parents = wheel_p_selection(pop, fitnesses[generation][0])
        kids = crossover(parents, fitnesses[generation][0], params.crossover_rate)
        pop = mutate(kids, params.mutations[0], params.mutations[1], fitnesses[generation][1])
        fitness = fitness_evaluation(pop, graph)
        
        fitnesses.append(fitness)

        if params.debug:
            print_gen_stats(generation, fitnesses[generation])        

        generation = generation + 1

    best_cut = ''.join(str(int(po)) for po in pop[fitnesses[generation][1]])

    if params.debug:
        plot(fitnesses)
    
    return (*fitnesses[generation], best_cut)

        