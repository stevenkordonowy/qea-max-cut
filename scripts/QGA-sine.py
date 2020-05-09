#########################################################
#                                                       #
#       QUANTUM GENETIC ALGORITHM (24.05.2016)          #
#                                                       #
#               R. Lahoz-Beltra                         #
#               S. Kordonowy                            #
#                                                       #
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND   #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY #
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  #
# THE SOFWTARE CAN BE USED BY ANYONE SOLELY FOR THE     #
# PURPOSES OF EDUCATION AND RESEARCH.
#                                                       #
# https://figshare.com/articles/Quantum_Genetic_Algorithm_QGA_/3398062/2         #
#                                                       #
#########################################################
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randrange, uniform

#########################################################
# GLOBAL VARIABLES                                      #
#########################################################
QuBitZero = np.array([[1], [0]])
QuBitOne = np.array([[0], [1]])
# Hadamard gate
sqrt2 = math.sqrt(2.0)
Had = 1/sqrt2 * np.array([[1, 1], [1, -1]])
ket_plus = np.matmul(Had, QuBitZero)
ket_minus = np.matmul(Had,QuBitOne)
# Had_gate = Qobj(Had)

#########################################################
# ALGORITHM PARAMETERS                                  #
#########################################################
N = 50                  # Define here the population size
Genome = 4              # Define here the chromosome length
generation_max = 450    # Define here the maximum number of generations/iterations

#########################################################
# QUANTUM POPULATION INITIALIZATION                     #
#########################################################


def rotation_matrix(angle):
    rot = np.empty([2, 2])
    rot[0, 0] = math.cos(angle)
    rot[0, 1] = -math.sin(angle)
    rot[1, 0] = math.sin(angle)
    rot[1, 1] = math.cos(angle)
    return rot

def Init_population(params):
    # qpv: quantum chromosome (or population vector, QPV)
    qpv = np.empty([params.pop_size, params.genome_len, 2])

    # Rotation Q-gate
    # Initial population array (individual x chromosome)
    for i in range(1, params.pop_size):
        for j in range(1, params.genome_len):
            rand_angle = math.radians(uniform(0, 90))
            rotation = rotation_matrix(rand_angle)
            rotated_qubit = np.matmul(rotation, ket_plus)
            qpv[i, j, 0] = rotated_qubit[0] ** 2
            qpv[i, j, 1] = rotated_qubit[1] ** 2

            # print("rotated_qubit={0}, qpv[i, j]={1}".format(rotated_qubit, qpv[i, j]))

    return AlgData(params, qpv)

# p_alpha: probability of finding qubit in alpha state  
def Measure(alg_data, p_alpha):
    for i in range(1, alg_data.params.pop_size):
        # print()
        for j in range(1, alg_data.params.genome_len):
            if p_alpha <= alg_data.qpv[i, j, 0]:
                alg_data.classical_chromosome[i, j] = 0
            else:
                alg_data.classical_chromosome[i, j] = 1
            # print(chromosome[i,j]," ",end="")

#########################################################
# FITNESS EVALUATION                                    #
#########################################################


def Fitness_evaluation(generation, alg_data):
    fitness_total = 0
    fitness_average = 0
    # for i in range(1, alg_data.params.pop_size):
    #     alg_data.fitness[i] = 0

#########################################################
# Define your problem in this section. For instance:    #
#                                                       #
# Let f(x)=abs(x-5/2+sin(x)) be a function that takes   #
# values in the range 0<=x<=15. Within this range f(x)  #
# has a maximum value at x=11 (binary is equal to 1011) #
#########################################################
    for i in range(1, alg_data.params.pop_size):
        x = 0
        for j in range(1, alg_data.params.genome_len):
            # translate from binary to decimal value
            # print("alg_data.classical_chromosome[i, j]={0},\nalg_data.params.genome_len-j-1={1},\npow(2, alg_data.params.genome_len-j-1={2}\n".format(alg_data.classical_chromosome[i, j], alg_data.params.genome_len-j-1, pow(2, alg_data.params.genome_len-j-1)))
            x += alg_data.classical_chromosome[i, j] * pow(2, alg_data.params.genome_len-j-1)
            # replaces the value of x in the function f(x)
            y = np.fabs((x-5)/(2+np.sin(x)))
            # the fitness value is calculated below: (Note that in this example is multiplied by a scale value, e.g. 100)
            alg_data.fitness[generation][i] = y*100
#########################################################

        fitness_total = fitness_total+alg_data.fitness[generation][i]
    fitness_average = fitness_total/N
    the_best_chrom = 0
    fitness_max = alg_data.fitness[generation][1]
    for i in range(1, alg_data.params.pop_size):
        if alg_data.fitness[generation][i] >= fitness_max:
            fitness_max = alg_data.fitness[generation][i]
            the_best_chrom = i


    alg_data.best_chrom[generation] = the_best_chrom
    alg_data.gen_fit[generation] = fitness_average

#########################################################
# QUANTUM ROTATION GATE                                 #
#########################################################


def rotation(alg_data, generation, best_chrom_idx):
    # rot=np.empty([2,2])
    # Lookup table of the rotation angle
    for i in range(1, alg_data.params.pop_size):
        for j in range(1, alg_data.params.genome_len):
            first_true = False
            second_true = False
            if alg_data.fitness[generation][i] < alg_data.fitness[generation][best_chrom_idx]:
              # if chromosome[i,j]==0 and alg_data.classical_chromosome[best_chrom[generation],j]==0:
                if alg_data.classical_chromosome[i, j] == 0 and alg_data.classical_chromosome[best_chrom_idx, j] == 1:
                    # Define the rotation angle: delta_theta
                    #    SAK: This might be a good place to update ie calculating a diff angle each time
                    delta_theta = 0.0785398163
                    rot = rotation_matrix(delta_theta)
                    alg_data.nqpv[i, j, 0] = (
                        rot[0, 0]*alg_data.qpv[i, j, 0])+(rot[0, 1]*alg_data.qpv[i, j, 1])
                    alg_data.nqpv[i, j, 1] = (
                        rot[1, 0]*alg_data.qpv[i, j, 0])+(rot[1, 1]*alg_data.qpv[i, j, 1])
                    alg_data.qpv[i, j, 0] = round(alg_data.nqpv[i, j, 0], 2)
                    alg_data.qpv[i, j, 1] = round(1-alg_data.nqpv[i, j, 0], 2)
                    first_true = True

                # SAK: this is seemingly never true...
                if alg_data.classical_chromosome[i, j] == 1 and alg_data.classical_chromosome[best_chrom_idx, j] == 0:
                    # Define the rotation angle: delta_theta (e.g. -0.0785398163)
                    delta_theta = -0.0785398163
                    rot = rotation_matrix(delta_theta)
                    alg_data.nqpv[i, j, 0] = (
                        rot[0, 0]*alg_data.qpv[i, j, 0])+(rot[0, 1]*alg_data.qpv[i, j, 1])
                    alg_data.nqpv[i, j, 1] = (
                        rot[1, 0]*alg_data.qpv[i, j, 0])+(rot[1, 1]*alg_data.qpv[i, j, 1])
                    alg_data.qpv[i, j, 0] = round(alg_data.nqpv[i, j, 0], 2)
                    alg_data.qpv[i, j, 1] = round(1-alg_data.nqpv[i, j, 0], 2)
                    second_true = True
                #    print("OHYA")
            #    if (first_true == second_true == True):
            #         print("all true!!!!")
            #    else:
            #        print("first={0}, second={1}".format(first_true, second_true))
              # if chromosome[i,j]==1 and chromosome[best_chrom[generation],j]==1:

#########################################################
# X-PAULI QUANTUM MUTATION GATE                         #
#########################################################
# pop_mutation_rate: mutation rate in the population
# mutation_rate: probability of a mutation of a bit


def mutation(alg_data, mutation_params):
    for i in range(1, alg_data.params.pop_size):
        up = np.random.randint(1, 101)/100
        if up <= mutation_params[0]:
            for j in range(1, alg_data.params.genome_len):
                um = np.random.randint(1, 101)/100
                if um <= mutation_params[1]:
                    alg_data.nqpv[i, j, 0] = alg_data.qpv[i, j, 1]
                    alg_data.nqpv[i, j, 1] = alg_data.qpv[i, j, 0]
                else:
                    alg_data.nqpv[i, j, 0] = alg_data.qpv[i, j, 0]
                    alg_data.nqpv[i, j, 1] = alg_data.qpv[i, j, 1]
        else:
            for j in range(1, alg_data.params.genome_len):
                alg_data.nqpv[i, j, 0] = alg_data.qpv[i, j, 0]
                alg_data.nqpv[i, j, 1] = alg_data.qpv[i, j, 1]
    for i in range(1, alg_data.params.pop_size):
        for j in range(1, alg_data.params.genome_len):
            alg_data.qpv[i, j, 0] = alg_data.nqpv[i, j, 0]
            alg_data.qpv[i, j, 1] = alg_data.nqpv[i, j, 1]


def write_to_file(alg_data):
    f = open("output.dat", "a")
    for idx, best_fit in enumerate(alg_data.gen_fit):
        f.write(str(idx)+" "+str(best_fit)+"\n")
    f.write(" \n")
    f.close()


#########################################################
# PERFORMANCE GRAPH                                     #
#########################################################
# Read the Docs in http://matplotlib.org/1.4.1/index.html


def plot_Output(gen_fit):
    x = range(0, generation_max)
    y = gen_fit
    plt.plot(x, y)
    plt.xlabel('Generation')
    plt.ylabel('Fitness average')
    plt.xlim(0.0, 450.0)
    # plt.ylim(0.0, 600.0)
    plt.show()


class AlgParams:
    'Instance variables for running QGA'

    def __init__(self, pop_size, genome_len, mutations):
        self.pop_size = pop_size
        self.genome_len = genome_len
        self.mutations = mutations

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class AlgData:

    def __init__(self, params, qpv):
        self.params = params

        # qpv: quantum chromosome (or population vector, QPV)
        self.qpv = qpv

        self.nqpv = np.zeros([params.pop_size, params.genome_len, 2])

        # chromosome: classical chromosome
        self.classical_chromosome = np.zeros([params.pop_size, params.genome_len], dtype=np.int)

        self.fitness = np.zeros([generation_max, params.pop_size])

        self.best_chrom = np.zeros([generation_max], dtype=np.int)

        self.gen_fit = np.zeros([generation_max])

    
    # def __str__(self):
    #     return str(self.__class__) + ": " + str(self.__dict__)


########################################################
# MAIN PROGRAM                                         #
########################################################
def Q_GA(params):
    generation = 0
    alg_data = Init_population(params)
    Measure(alg_data, 0.5)
    Fitness_evaluation(generation, alg_data)
    while (generation < generation_max - 1):
        rotation(alg_data, generation, alg_data.best_chrom[generation])
        mutation(alg_data, params.mutations)
        generation = generation+1
        Measure(alg_data, 0.5)
        Fitness_evaluation(generation, alg_data)

    return alg_data


# if os.path.isfile("output.dat"):
#     os.remove("output.dat")


def main():
    print("do calculations")
    # popSize=N+1
    # genomeLength=Genome+1

    # print ("""QUANTUM GENETIC ALGORITHM""")
    # # input("Press Enter to continue...")
    # for _trial in range(10):
    #     mutations = [0.01, 0.001] # pop_mutation_rate, mutation_rate
    #     params = AlgParams(popSize, genomeLength, mutations)
    #     result = Q_GA(params)
    #     print_result(result)


if __name__ == "__main__":
    main()
