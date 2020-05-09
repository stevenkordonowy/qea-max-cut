import unittest
import numpy as np
from QGA import *
import pprint
import csv  
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class TestQGA(unittest.TestCase):

    def test_simple_run(self):
        popSize=N+1
        genomeLength=Genome+1

        params = AlgParams(popSize, genomeLength, [0.1020408163265306,0.8979591836734693])
        result = Q_GA(params)
        self.assertGreaterEqual(max(result.gen_fit), 400, "Should get >= 400 at least once")

    def Xtest_simple_run_500(self):
        popSize=N+1
        genomeLength=Genome+1

        maxes = []
        for _ in range(5):
            params = AlgParams(popSize, genomeLength, [0.1020408163265306,0.8979591836734693])
            result = Q_GA(params)
            maxes.append(max(result.gen_fit))
        self.assertGreaterEqual(max(maxes), 500, "Should get >= 500 at least once")


    # def test_sum(self):
    #     print("pop_mut_rate, ind_mut_rate, max, avg")

    #     maxes = []
    #     to_write = []
    #     for pop_mut_rate in np.linspace(0,1,50):
    #         # for ind_mut_rate in np.linspace(0,1,50):
    #             # trials = []
    #             mx = 0
    #             avga = 0
    #             trials = 5
    #             for _trial in range(trials):
    #                 params = AlgParams(popSize, genomeLength, [0.1020408163265306,0.8979591836734693])
    #                 result = Q_GA(params)
    #                 # print_result(result)
    #                 m = max(result.gen_fit)
    #                 avg = np.mean(result.gen_fit)
    #                 maxes.append(m)
    #                 # trials.append((pop_mut_rate, ind_mut_rate, m, avg))
    #                 mx = m if m > mx else mx
    #                 avga += avg
    #                 # print("{0}, {1}, {2}, {3}".format(pop_mut_rate, ind_mut_rate, m, avg))
    #                 print("{0},{1} trial {2}".format(0.1020408163265306,0.8979591836734693, _trial))
    #             print("{0}, {1}, {2}, {3}".format(0.1020408163265306,0.8979591836734693, mx, avga / trials))
    #             to_write.append((0.1020408163265306,0.8979591836734693, mx, avga / 3))
    #             # print("{0}, {1}, {2}, {4}".format(pop_mut_rate, ind_mut_rate, max(trials), np.mean(trials)))

    # #         with open("output.dat", "w") as f:
    # #             csv_out = csv.writer(f)
    # #             csv_out.writerow("pop_mut_rate, ind_mut_rate, max, avg")
    # #             for row in to_write:
    # #                 csv_out.writerow(row)
    # #             f.write(" \n")
    # #             f.close()
    #     self.assertGreaterEqual(max(maxes), 525, "Should get >= 525 at least once")

    # def testing(self):
    #     df = pd.read_csv('output.dat')
    #     # my_data = np.readtext(open('output.cp.dat'))
    #     # print(df)
    #     # df = df.drop(df.columns[[2]], axis=1)
    #     # df = df.drop(df.index[0])
    #     # print()
    #     print("overall max: {0}".format(max(df['max'])))
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     surf = ax.scatter(df['pop_mut_rate'], df['ind_mut_rate'], df['max'])
    #     ax.set_xlabel('pop_mut_rate')
    #     ax.set_ylabel('ind_mut_rate')
    #     ax.set_zlabel('avg')
    #     # ax.set_zlim(-1.01, 1.01)
    #     # fig.colorbar(surf, shrink=0.5, aspect=5)
    #     plt.title('Original Code')
    #     plt.show()
        # with open('long.csv') as f:
        #     lines = (line for line in f if not line.startswith('#'))
        #     FH = np.loadtxt(f, delimiter=',')
    #     # data = np.loadtxt('long.out', skiprows=1)
    #     # x=data[:,0]
    #     # y=data[:,3]
    #     # print("max:{0}".format(max(y)))
    #     # plt.plot(x,y)
    #     # plt.xlabel('Generation')
    #     # plt.ylabel('Fitness average')
    #     # plt.xlim(0.0, 550.0)
    #     # plt.show()

if __name__ == '__main__':
    unittest.main()