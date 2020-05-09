import numpy as np
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
import pyquil.api as api
from convertgraphs import *
import networkx

qvm_connection = api.QVMConnection()


# square_ring = [(0,1),(1,2),(2,3),(3,0)]
# G = read_graph('ring-9')
# G = read_graph('g1')
# G = networkx.gnp_random_graph(25, 0.75)
G = networkx.complete_graph(6)


steps = 2
inst = maxcut_qaoa(graph=G, steps=steps)
betas, gammas = inst.get_angles() # might be a clean way to optimize angles... we shall see


t = np.hstack((betas, gammas))
param_prog = inst.get_parameterized_program()
prog = param_prog(t)
wf = qvm_connection.wavefunction(prog)
wf = wf.amplitudes

exps = {state_index : abs(np.conj(wf[state_index])*wf[state_index]) for state_index in range(inst.nstates)}
exps = {k: v for k, v in sorted(exps.items(), key=lambda item: abs(item[1]))}
argmax = max(exps, key=lambda key: exps[key])
mx = exps[argmax]
print('max:{0}, cut:{1}, idx:{2}'.format(mx, inst.states[argmax], argmax))
for state_index in range(inst.nstates):
    print(inst.states[state_index], exps[state_index])
    