import math
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randrange, uniform
import itertools
import time
import scipy.stats

# Quantum statics
Sqrt_2 = math.sqrt(2.0)
Had = np.array([[1, 1], [1, -1]]) / Sqrt_2
Zero = np.array([[1], [0]])
Plus = np.matmul(Had, Zero)

class Qubit:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def a(self):
        return self.alpha

    def b(self):
        return self.beta

    def a_squared(self):
        return pow(abs(self.alpha), 2)

    def b_squared(self):
        return pow(abs(self.beta), 2)

    def to_np(self):
        return [[self.alpha], [self.beta]]

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

def rotation_matrix(angle):
    rot = np.empty([2, 2])
    rot[0, 0] = math.cos(angle)
    rot[0, 1] = -math.sin(angle)
    rot[1, 0] = math.sin(angle)
    rot[1, 1] = math.cos(angle)
    return rot

def rotate_qubit(qubit, angle):
    rot = rotation_matrix(angle)
    prod = np.matmul(rot, qubit)
    return Qubit(prod[0].item(), prod[1].item())


def init_pop(params):
    # data[generation][individual][gene]
    init_pop = np.empty([params.pop_size, params.genome_len], dtype=Qubit)
    for indiv in range(params.pop_size):
        for gene in range(params.genome_len):
            init_pop[indiv, gene] = rotate_qubit(Plus, math.radians(uniform(0, 90)))

    return init_pop


def measure(q_chromosomes, c_chromosomes=None, running_best=None): # expects q_chromosomes as `Qubit`s
    pop = np.zeros([len(q_chromosomes), len(q_chromosomes[0])])
    # for idx, chromosome in [x for x in enumerate(q_chromosomes) if x != running_best]:
    for idx, chromosome in enumerate(q_chromosomes):
        for idx2, gene in enumerate(chromosome):
            # Consider using Lea here
            r = uniform(0,1)
            pop[idx][idx2] = r < gene.a_squared()
    if running_best:
        pop[running_best] = c_chromosomes[running_best]
    return np.array(pop)

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

d_angle = 0.001 * math.pi
# d_angle = 0.0785398163

def rotate_towards_best(q_pop, fits, idx_best):
    new_q_pop = np.zeros([len(q_pop), len(q_pop[0])], dtype=Qubit)
    best_fit = fits[idx_best]
    best_q_chrom = q_pop[idx_best]
    # iterate q chroms
    for idx, q_chrom in enumerate(q_pop):
        if fits[idx] < best_fit and idx_best != idx:
            rotated_chrom = []
            # Change each gene individually
            for g_idx, gene in enumerate(q_chrom):
                gene_best = best_q_chrom[g_idx]
                A = np.array([[gene_best.a(), gene.a()], [gene_best.b(), gene.b()]])
                det = np.linalg.det(A)
                sgn = 1 if det <= 0 else -1
                # if idx_best == idx:
                #     print('idx_best == idx, {0}'.format(det))

                theta = d_angle * sgn
                rotated = rotate_qubit(gene.to_np(), theta)
                rotated_chrom.append(rotated)
            new_q_pop[idx] = rotated_chrom
        else:
            new_q_pop[idx] = q_pop[idx]

    return np.array(new_q_pop)

def mutate(q_pop, rate_pop, rate_individual, idx_best):   
    mutated = np.zeros([len(q_pop), len(q_pop[0])], dtype=Qubit)

    for idx_chrom in range(len(q_pop)):
    # for idx_chrom in range(len(q_pop)):
        r = uniform(0,1)
        if r <= rate_pop:
            for idx_gene in range(len(q_pop[0])):
                r = uniform(0,1)
                if r <= rate_individual:
                    gene = q_pop[idx_chrom][idx_gene]
                    mutated[idx_chrom][idx_gene] = Qubit(gene.b(), gene.a())
                else:
                    mutated[idx_chrom][idx_gene] = q_pop[idx_chrom][idx_gene]
        else:
            mutated[idx_chrom] = q_pop[idx_chrom]

    # Preserve the best
    mutated[idx_best] = q_pop[idx_best]

    return mutated
            

class AlgParams:
    'Instance variables for running QGA'

    def __init__(self, pop_size, genome_len, max_gens, mutations, debug = False):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.mutations = mutations
        self.debug = debug
        self.max_gens = max_gens

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

def print_gen_stats(gen, fitnesses):
    mean = np.mean(fitnesses[0])
    std = np.std(fitnesses[0])
    print('{0}, {1}, {2}, {3}, {4}'.format(gen, fitnesses[0][fitnesses[1]], mean, std, std/math.sqrt(mean)))

def plot(fitnesses):
    stats = [(np.mean(fits[0]), np.std(fits[0]), fits[0][fits[1]]) for fits in fitnesses]

    with open('qga-single-run.csv', 'a') as csvfile:
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
def run_qga(graph, params):
    generation = 0
    q_pop = init_pop(params)  
    pop = measure(q_pop)
    fitnesses = [fitness_evaluation(pop, graph)]

    if params.debug:
        print('gen, max, avg, std, SE')

    while (generation < params.max_gens):

        q_pop = rotate_towards_best(q_pop, *fitnesses[generation])
        q_pop = mutate(q_pop, params.mutations[0], params.mutations[1], fitnesses[generation][1])
        pop = measure(q_pop, pop, fitnesses[generation][1])
        fitnesses.append(fitness_evaluation(pop, graph))

        if params.debug:
            print_gen_stats(generation, fitnesses[generation])        

        generation = generation + 1

    if params.debug:
        plot(fitnesses) 
            
    best_cut = ''.join(str(int(po)) for po in pop[fitnesses[generation][1]])
    return (*fitnesses[generation], best_cut)

