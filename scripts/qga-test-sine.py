import unittest
import numpy as np
from QGA import *
import pprint
import csv  
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from random import randrange, uniform
import time, timeit

class TestQGA(unittest.TestCase):

    def test_simple_run(self):
        popSize=N+1
        genomeLength=Genome+1

        # params = AlgParams(popSize, genomeLength, [0.1020408163265306,0.8979591836734693])
        params = AlgParams(popSize, genomeLength, [0.1,0.5])
        t0 = time.time()
        result = Q_GA(params)
        t1 = time.time()
        print("max={0}, time={1}fck".format(max(result.gen_fit), t1-t0))
        # new_parms = AlgParams(popSize, genomeLength, [0.1,0.5])
        # t = timeit.timeit(lambda: Q_GA(new_parms), number=3)
        # print("sss t={0}, t_avg={1}".format(t, t/3))
        # plot_Output(result.gen_fit)
        self.assertGreaterEqual(max(result.gen_fit), 400, "Should get >= 400")

    def test_simple_run_500(self):
        popSize=N+1
        genomeLength=Genome+1
        trials = 10

        maxes = []
        for _ in range(trials):
            gamma = uniform(0,1)
            beta = uniform(0,1)
            params = AlgParams(popSize, genomeLength, [gamma,beta])
            result = Q_GA(params)
            # plot_Output(result.gen_fit)
            maxes.append(max(result.gen_fit))
        self.assertGreaterEqual(max(maxes), 500, "Should get >= 500 at least once")


    def XXXtest_sum(self):
        if os.path.isfile("output.dat"):
            os.remove("output.dat")
        popSize=N+1
        genomeLength=Genome+1

        print("pop_mut_rate, ind_mut_rate, max, avg")

        maxes = []
        to_write = []
        # for pop_mut_rate in np.linspace(0,1,50):
            # for ind_mut_rate in np.linspace(0,1,50):
                # trials = []
        mx = 0
        avga = 0
        trials = 5
        for _trial in range(trials):
            # pop_mut_rate = ran
            params = AlgParams(popSize, genomeLength, [0.1020408163265306,0.8979591836734693])
            result = Q_GA(params)
            # print_result(result)
            m = max(result.gen_fit)
            avg = np.mean(result.gen_fit)
            maxes.append(m)
            # trials.append((pop_mut_rate, ind_mut_rate, m, avg))
            mx = m if m > mx else mx
            avga += avg
            # print("{0}, {1}, {2}, {3}".format(pop_mut_rate, ind_mut_rate, m, avg))
            # print("{0},{1} trial {2}".format(0.1020408163265306,0.8979591836734693, _trial))
            to_write.append((0.1020408163265306,0.8979591836734693, m, avg))
        print("{0}, {1}, {2}, {3}".format(0.1020408163265306,0.8979591836734693, mx, avga / trials))
        # to_write.append((0.1020408163265306,0.8979591836734693, mx, avga / 3))
        # print("{0}, {1}, {2}, {4}".format(pop_mut_rate, ind_mut_rate, max(trials), np.mean(trials)))

        with open("output.dat", "w") as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['pop_mut_rate','ind_mut_rate','max','avg'])
            for row in to_write:
                csv_out.writerow(row)
            f.write(" \n")
            f.close()
        self.assertGreaterEqual(max(maxes), 525, "Should get >= 525 at least once")

    def XXXtest_display(self):
        df = pd.read_csv('output.dat', header=0)
        # my_data = np.readtext(open('output.cp.dat'))
        print(df)
        # df = df.drop(df.columns[[2]], axis=1)
        # df = df.drop(df.index[0])
        # print()
        print("overall max: {0}".format(max(df['max'])))
        fig = plt.figure()
        ax = Axes3D(fig)
        surf = ax.scatter(df['pop_mut_rate'], df['ind_mut_rate'], df['max'])
        ax.set_xlabel('pop_mut_rate')
        ax.set_ylabel('ind_mut_rate')
        ax.set_zlabel('avg')
        # ax.set_zlim(-1.01, 1.01)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title('Original Code')
        plt.show()

if __name__ == '__main__':
    unittest.main()