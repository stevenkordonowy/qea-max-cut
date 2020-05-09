import unittest
import numpy as np
import pprint
import csv  
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from random import randrange, uniform
import time, timeit
from convertgraphs import *
from QGAMaxcut import *
from SGAMaxcut import *
import networkx as nx
import matplotlib.pyplot as plt
# import dwave_networkx as dnx
# import dimod
import urllib.request

def generate_rand_cut(n):
    N = 2 ** n
    r = randrange(N)
    bitstring = format(r, '0{0}b'.format(n))
    return bitstring


def plot(stats):
    stats = [(np.mean(fitnesses[0]), np.std(fitnesses[0]), fitnesses[0][fitnesses[1]]) for fitnesses in stats]
    
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

def complete_graph(size):
    G = nx.complete_graph(size)
    # order = 250
    # G = nx.complete_graph(order)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G

def ring(size):
    G = nx.cycle_graph(size)
    # order = 250
    # G = nx.complete_graph(order)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G

class TestQGA(unittest.TestCase):

    def xtest_plot_qaoa(self):
        for size in range(3,8):
            data = []
            with open('{0}-range.csv'.format(size), 'r') as csvfile:
                graph = csv.reader(csvfile, delimiter=',')
                for stuff in graph:
                    # We want the nodes of our graph to start at 0
                    data.append((float(stuff[0]), float(stuff[1]), float(stuff[2])))
            
            data = np.array(data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:,0], data[:,1], data[:,2] / size, c='r', marker='o')

            ax.set_xlabel('gammas')
            ax.set_ylabel('betas')
            ax.set_zlabel('% happy vertices')

            plt.savefig('size-{0}.png'.format(size), bbox_inches='tight')
            # plt.show()

    def xtest_convert_graphs(self):
        g1 = read_graph('G1')
        self.assertEqual(g1.order(), 800, 'Should have 800 nodes')
        self.assertTrue(0 in g1, 'Should start at 0')
        self.assertEqual(g1.size(), 19176, 'Should have 19176 edges')
    
    def xxxtest_running_graph(self):
        graph = read_graph('G1')
        popSize=N+1
        genomeLength=4+1 # was orig 4 so could do {0     15}=2^4

        # params = AlgParams(popSize, genomeLength, [0.1020408163265306,0.8979591836734693])
        params = AlgParams(popSize, genomeLength, 5, [0.1,0.5])


    def xtest_init_pop(self):
        params = AlgParams(5, 11, 5, [0.1,0.5])
        pop = init_pop(params)
        # for individual in pop:
        for chrom in pop:
            for gene in chrom:
                self.assertTrue(math.isclose(gene.a_squared() + gene.b_squared(), 1.0), '{0} + {1} should be 1'.format(gene.a_squared(), gene.b_squared()))

    def xtest_measure(self):
        params = AlgParams(5, 11, [0.1,0.5])
        q_pop = init_pop(params)[0]
        # print(q_pop)
        # print()
        pop = measure(q_pop)
        # for idx, measurement in enumerate(pop):
        #     for idx2, gene in enumerate(measurement):
                # print('qubit={0}, measurement={1}'.format(q_pop[idx][idx2], gene))

    def xtest_individual_fit(self):
        # cut = generate_rand_cut(5)
        cut = '10011'
        # print(cut)
        G = nx.Graph()
        G.add_edge(0,1,weight=1.0)
        G.add_edge(0,2,weight=1.0)
        G.add_edge(1,2,weight=1.0)
        G.add_edge(2,3,weight=1.0)
        G.add_edge(3,4,weight=1.0)
        # nx.draw(G)
        # plt.show()
        fit = individual_fit(cut, G)
        # print(fit)
        self.assertEqual(fit[0], 3, 'Fit should be 3')
        self.assertEqual(fit[1], [(0,1),(0,2),(2,3)], "Cut should be edges (0, 1), (0, 2), (2, 3)")

    def xtest_pop_fit_even_ring(self):
        G = read_graph_into_array('ring-10')

        pop_size = 50
        chrom_size = len(G)

        params = AlgParams(pop_size, chrom_size, 5, [0.1,0.5])
        q_pop = init_pop(params)
        pop = measure(q_pop)
        best_gene = fitness_evaluation(pop, G)
        # best_gene[0][best_gene[1]] should be 10 for ring-10
        # self.assertEqual(best_gene[0][best_gene[1]], 4, 'Best should be 4')


        # print('best_gene:')
        # print(pop[best_gene[0]])
        # print('producing:')
        # print(best_gene[1])
        # best = determine_best(q_pop)
        # print('best:')
        # print(best)

    @unittest.skip("plotting")
    def test_pop_fit_g1(self):
        G = read_graph_into_array('g1')
        
        pop_size = 50
        chrom_size = len(G)

        running_best = (0.0,)
        
        # for pop_mut_rate in np.linspace(0,1,10):
        #     for ind_mut_rate in np.linspace(0,1,10):

        params = AlgParams(pop_size, chrom_size, 5, [0.25, 0.25])
        q_pop = init_pop(params)
        chrom = '00000010100000100100001010101111000000011001000000000010001010000011010000001001100010011001100111110000000001101000101000001001011000010000110100010001001000000000001000000010000100010000010011010000000000000010001100100100100001010010100111101000111101111000111100100000000000110100000110000001011111001111001100010001010001000111100000111110010000000010000000001100110010100011111010011010000110010101011000101100001000100010001010000000001010001000000101011010010110010110100101001111001111011011000000100010111000100111011000011001100010101010001001100101000000100010100000001111000010001001100101001110010000101111000010000011010001010000010011100001011011100011000100110101110000001110000001101000011000010001000000000011000101011011111001100100001011011010101001101001001010000101011000001001'
        cut_val = individual_fit(chrom, G)

        self.assertEqual(cut_val, 8956, 'gotta be equal')

        # best = determine_best(q_pop)
        # print('best:')
        print('allllll done!')
        print(running_best)


    @unittest.skip("Speed baby")
    def test_simple_run_ring(self):
        G = read_graph('ring-9')
        pop_size = 100
        chrom_size = G.order() 
        params = AlgParams(pop_size, 5, chrom_size, [0.1, 0.1])

        result = run_qga(G, params)
        print('Final ring-9 results:')
        print(result)

        # self.assertGreaterEqual(max(result.gen_fit), 400, "Should get >= 400")

    @unittest.skip("Speed baby")
    def test_simple_run(self):
        G = read_graph_into_array('g1')
        # G = nx.complete_graph(1000)
        # G = nx.gnp_random_graph(800, 0.25)
        # order = 250
        # G = nx.complete_graph(order)
        # for edge in G.edges():
            # G[edge[0]][edge[1]]['weight'] = 1.0

        pop_size = 50
        chrom_size = len(G) 
        
        running_best = (0.0,)

        max_trials = 3
        times_ms = []
        for _trial in range(max_trials):
            print('Trial {0}'.format(_trial))
            params = AlgParams(pop_size, chrom_size, 25, [0.25, 0.25])
            t1 = time.time()
            result = run_qga(G, params)
            t2 = time.time()
            times_ms.append((t2 - t1) * 1000)
            if result[0][result[1]] > running_best[0]:
                running_best = (result[0][result[1]], result[2])
                print('New best: {0}'.format(running_best[0]))
        # plot(result)
        print('Overall best: {0}'.format(running_best[0]))
        print('cut:{0}'.format(running_best[1]))
        avg_time = np.mean(times_ms)
        std_time = np.std(times_ms)
        max_time = max(times_ms)
        min_time = min(times_ms)
        print('Timings (ms)')
        print('avg={0}, std={1}, max={2}, min={3}'.format(avg_time, std_time, max_time, min_time))

    
    @unittest.skip("Speed baby")
    def test_individual_fit_complete_graphs(self):
        for size in [2*x for x in range(2,100)]:
            G = nx.complete_graph(size)
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1.0

            pop_size = 50
            chrom_size = len(G) 
            params = SGAAlgParams(pop_size, chrom_size, 5, 0.75, [0.5, 0.5])
            q_pop = init_pop(params)
            pop = measure(q_pop)
            chrom = ''.join(['01' for x in range(int(size / 2))])
            fit = individual_fit(chrom, G)

            # result = run_qga(G, params)
            # print('{0}, {1}'.format(size, fit))
            exp = pow(size, 2) / 4
            self.assertGreaterEqual(fit, exp, "Should get >= {0}".format(exp))

    @unittest.skip("Speed baby")
    def test_individual_fit_ring_graphs(self):
        for size in [2*x for x in range(2,100)]:
            G = nx.cycle_graph(size)
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1.0

            pop_size = 50
            chrom_size = len(G) 
            params = AlgParams(pop_size, chrom_size, 5, [0.5, 0.5])
            q_pop = init_pop(params)
            pop = measure(q_pop)
            chrom = ''.join(['01' for x in range(int(size / 2))])
            fit = individual_fit(chrom, G)

            # result = run_qga(G, params)
            # print('{0}, {1}'.format(size, fit))
            self.assertGreaterEqual(fit, size, "Should get >= {0}".format(size))

    # @unittest.skip("running qga")
    def xtest_simple_run_SGA(self):
        print('SGA')
        max_trials = 1
        max_gens = 250
        pop_size = 50 
        
        for g in range(1, 2):
            print('###Starting graph G{0}'.format(g))
            G = read_graph_into_array('G{0}'.format(g))
            pop_size = 50
            chrom_size = len(G) 
            
            bests = []
            times_ms = []
            for _trial in range(max_trials):
                # print('Trial {0}'.format(_trial))
                params = SGAAlgParams(pop_size, chrom_size, max_gens, 0.75, [0.1, 0.1], True)
                t1 = time.time()
                result = run_sga(G, params)
                t2 = time.time()
                times_ms.append((t2 - t1) * 1000)
                bests.append(result[0][result[1]])
                # print('Best: {0}'.format(result[0][result[1]]))

            avg_best = np.mean(bests)
            std_best = np.std(bests)
            max_best = max(bests)
            min_best = min(bests)
            print('avg={0}, std={1}, max={2}, min={3}'.format(avg_best, std_best, max_best, min_best))
            
            # with open('sga-results.csv', 'a') as csvfile:
            #     csvfile.write('{},{},{},{}\n'.format(g, avg_best, std_best, max_best))
        
    
    # graphs = []
    def xtest_compare_g1(self):
        qga = []
        sga = []
        with open('qga-single-run.csv', 'r') as qga_file:
            data = csv.reader(qga_file, delimiter=',')
            for line in data:
                qga.append((float(line[0]), float(line[1])))
        qga = np.array(qga)

        with open('sga-single-run.csv', 'r') as sga_file:
            data = csv.reader(sga_file, delimiter=',')
            for line in data:
                sga.append((float(line[0]), float(line[1])))
        sga = np.array(sga)

        fig = plt.figure()
        p1 = fig.add_subplot(1,2,1)
        p1.plot(range(200), qga[:200,0], label='QEA')
        p1.plot(range(200), sga[:200,0], label='SEA')
        p1.set_xlabel('Generation')
        p1.set_ylabel('Fit')
        plt.title('Best')
        handles, labels = p1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')


        p2 = fig.add_subplot(1,2,2)
        p2.plot(range(200), qga[:200,1])
        p2.plot(range(200), sga[:200,1])
        p2.set_xlabel('Generation')
        p2.set_ylabel('Fit')
        plt.title('Average')
        plt.show()

    def test_compare_data(self):
        pos = [1,2,3,4,5,14,15,16,17,22,23,24,25,26,35,36,37,38,43,44,45,46,47,48,49,50,51,52,53,54,55,58,60,63]
        negs = [6,7,8,9,10,11,12,13,18,19,20,21,27,28,29,30,31,32,33,34,39,40,41,42,56,57,59,61,62,64,65,66]

        sga_res_pos = []
        sga_res_neg = []
        with open('sga-results.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for line in data:
                size = int(line[0])
                if size in pos:
                    # (size, max, avg)
                    sga_res_pos.append((size, float(line[3]), float(line[1])))
                else:
                    sga_res_neg.append((size, float(line[3]), float(line[1])))
        sga_res_pos = np.array(sga_res_pos)
        sga_res_neg = np.array(sga_res_neg)

        qga_res_pos = []
        qga_res_neg = []
        with open('qga-results.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for line in data:
                size = int(line[0])
                if size in pos:
                    # (size, max, avg)
                    qga_res_pos.append((size, float(line[3]), float(line[1])))
                else:
                    qga_res_neg.append((size, float(line[3]), float(line[1])))

        qga_res_pos = np.array(qga_res_pos)
        qga_res_neg = np.array(qga_res_neg)

        memetics_res_pos = []
        memetics_res_neg = []
        with open('memetic-results.csv', 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            for line in data:
                size = int(line[0])
                if size in pos:
                    # (size, max, avg)
                    memetics_res_pos.append((size, float(line[1]), float(line[2])))
                else:
                    memetics_res_neg.append((size, float(line[1]), float(line[2])))
        memetics_res_pos = np.array(memetics_res_pos)
        memetics_res_neg = np.array(memetics_res_neg)

        fig = plt.figure()
        p1 = fig.add_subplot(2,2,1)
        p1.scatter(qga_res_pos[:,0], qga_res_pos[:,1], label='QEA')
        p1.scatter(sga_res_pos[:,0], sga_res_pos[:,1], label='SEA')
        p1.scatter(memetics_res_pos[:,0], memetics_res_pos[:,1], label='WH')
        p1.set_xlabel('Graph instance')
        p1.set_ylabel('Cut value')
        plt.title('(a)')
        handles, labels = p1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')

        p2 = fig.add_subplot(2,2,2)
        p2.scatter(qga_res_pos[:,0], qga_res_pos[:,2])
        p2.scatter(sga_res_pos[:,0], sga_res_pos[:, 2])
        p2.scatter(memetics_res_pos[:,0], memetics_res_pos[:,1])
        p2.set_xlabel('Graph instance')
        p2.set_ylabel('Cut value')
        plt.title('(b)')

        p3 = fig.add_subplot(2,2,3)
        p3.scatter(qga_res_neg[:,0], qga_res_neg[:,1])
        p3.scatter(sga_res_neg[:,0], sga_res_neg[:,1])
        p3.scatter(memetics_res_neg[:,0], memetics_res_neg[:,1])
        p3.set_xlabel('Graph instance')
        p3.set_ylabel('Cut value')
        plt.title('(c)')

        p4 = fig.add_subplot(2,2,4)
        p4.scatter(qga_res_neg[:,0], qga_res_neg[:,2])
        p4.scatter(sga_res_neg[:,0], sga_res_neg[:, 2])
        p4.scatter(memetics_res_neg[:,0], memetics_res_neg[:,1])
        p4.set_xlabel('Graph instance')
        p4.set_ylabel('Cut value')
        plt.title('(d)')

        # plt.savefig('pos-weights.png'.format(size), bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()   


    def xtest_run_full_qga(self):
        print('qga')
        max_trials = 1
        max_gens = 500
        pop_size = 50

        for g in range(1, 2):
            print('###Starting graph G{0}'.format(g))
            G = read_graph_into_array('g{0}'.format(g))
            chrom_size = len(G) 
            
            bests = []
            times_ms = []
            for _trial in range(max_trials):
                print('Trial {0}'.format(_trial))
                params = AlgParams(pop_size, chrom_size, max_gens, [0.1, 0.1], True)
                t1 = time.time()
                result = run_qga(G, params)
                t2 = time.time()
                times_ms.append((t2 - t1) * 1000)
                bests.append(result[0][result[1]])
                print('Best: {0}'.format(result[0][result[1]]))

            avg_best = np.mean(bests)
            std_best = np.std(bests)
            max_best = max(bests)
            min_best = min(bests)
            print('avg={0}, std={1}, max={2}, min={3}'.format(avg_best, std_best, max_best, min_best))
            avg_time = np.mean(times_ms)
            std_time = np.std(times_ms)
            max_time = max(times_ms)
            min_time = min(times_ms)
            print('Timings (ms)')
            print('avg={0}, std={1}, max={2}, min={3}'.format(avg_time, std_time, max_time, min_time))
            with open('qga-results.csv', 'a') as csvfile:
                csvfile.write('{},{},{},{}\n'.format(g, avg_best, std_best, max_best))

# def xtest_find_negative_weighted_graphs(self):
#         pos = []
#         negs = []
#         for g in range(1, 67):
#             neg = False
#             G = read_graph_into_array('g{0}'.format(g))
#             for i1 in G:
#                 for i2 in G[i1]:
#                     if G[i1][i2]['weight'] < 0:
#                         neg = True
#             if neg:
#                 negs.append(g)
#             else:
#                 pos.append(g)
        
#         print('Pos:{0}'.format(','.join(str(g) for g in pos)))
#         print('Neg:{0}'.format(','.join(str(g) for g in negs)))
    # def test_download_graphs(self):
    #     for g in range(3, 67):
    #         url = 'https://web.stanford.edu/~yyye/yyye/Gset/G{0}'.format(g)
    #         file_name = 'G{0}.txt'.format(g)
    #         # Download the file from `url` and save it locally under `file_name`:
    #         urllib.request.urlretrieve(url, file_name)


if __name__ == '__main__':
    unittest.main()