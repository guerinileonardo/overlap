#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimates the state overlap of a random pair of states using both a noisy
implementation of the Bell-basis circuit (via the LPDO paradigm) and
the quasiprob. method.

"""

import numpy as np
import time
from tools import bb_block, dep_site, eff_meas, kraus_dep, overlap, \
    pauli6_frame, random_lpdo, renorm, rotate_lpdo, trace  
import matplotlib.pyplot as plt

############## general settings #####################################

m = 3 # number of qubits in each state
N = 2*m # number of qubits in the circuit
pd = 2 # dimension of the particles (for now, it has to be 2)

instance_num = 1 # number of instances
sample_num = 5000 # initial number of samples
sample_increment = 5000 # increment on the number of samples 

ov_bound = 1e-1 # lower bound for the overlap between the random states
noise_unit = 0.005 # noise in the implementation of the unitary
noise_meas = 0.01 # noise in the implementation of the measurement

svd_tol = 1e-8 # bound for discarding singular values in the truncation
bond_bound = 32 # max dimension admitted in the SVD decomposition
kraus_bound = 72 # upper bound on the kraus dimension between ket and bra
trace_bound = 1e-6

error_bound = 0.05 # max error for the weighted average of our method 
abs_av_quasi = 1 # just a kickstart value, greater than error_bound
abs_av_noisy = 0 # just a kickstart value
rel_av_quasi = 1 # just a kickstart value, greater than error_bound
rel_av_noisy = 0 # just a kickstart value
sample_max = 5000

column_num = 25 # number of columns in the histograms

tac = time.perf_counter() # set counter

################# starters ########################################

frame, pinvT = pauli6_frame_tn()

kraus_unit = kraus_dep(noise_unit)
kraus_meas = kraus_dep(noise_meas)

# list of relevant effects in a Z measurement on each qubit of circuit 3
eff_indices = []
for i in range(2**(N)):
    a = np.zeros(N, int)
    bits = bin(i)[2:]
    for j in range(len(bits)):
        a[N-len(bits)+j] = int(bits[j])
    for k in range(int(m/2)+1):
        if np.dot(a[:m], a[m:]) == 2*k:
            eff_indices.append(a)

############################### Bell-basis circuit

cnot_num = [0, 1, 10, 27, 52, 85, 126, 232]

psi = []
phi = []
E = []
P_psi = []
P_phi = []
prob_noisy = []
dep_fid_mean = []
cnot_fid_mean = []
final_fid_list = []
for instance in range(instance_num):
    print('Implementing instance', instance+1, 'of', instance_num)
    psi.append(random_lpdo(m, pd))
    phi.append(rotate_lpdo(psi[instance], 
                          (np.mod(instance, 3)+1)*np.pi/8,
                          cnot_on=False))
    ov = overlap(psi[instance], phi[instance])
    E.append(ov)
    psi_dep, _, _ = dep_site(psi[instance], np.arange(m), kraus_meas,
                             svd_tol, kraus_bound)
    psi_dep = renorm(psi_dep) # renormalising
    phi_dep, _, _ = dep_site(phi[instance], np.arange(m), kraus_meas,
                             svd_tol, kraus_bound)
    phi_dep = renorm(phi_dep) # renormalising
    P_psi.append([])
    P_phi.append([])
    for index in range(len(frame)**m):
        label = list(np.base_repr(index, len(frame)))
        while len(label) < m:
            label.insert(0, 0)
        effect = []
        for i in range(m):
            effect.append(frame[int(label[i])])
        # measuring psi
        P_psi[instance].append(eff_meas(psi_dep, effect))
        # measuring phi
        P_phi[instance].append(eff_meas(phi_dep, effect))

    # implementing the noisy circuit
    mps = psi[instance] + phi[instance]
    dep_fid_sum = 0
    dep_fid_total = 1
    cnot_fid_sum = 0
    cnot_fid_total = 1
    for qubit in range(m):
        mps, dep_fid_sum, dep_fid_total, cnot_fid_sum, cnot_fid_total = \
            bb_block(mps, qubit, kraus_unit, svd_tol, kraus_bound,
                     bond_bound, dep_fid_sum, dep_fid_total,
                     cnot_fid_sum, cnot_fid_total)
    mps, dep_fid_sum, dep_fid_total = dep_site(mps, np.arange(N), kraus_meas,
                                               svd_tol, kraus_bound,
                                               dep_fid_sum, dep_fid_total)
    tr = 0
    while abs(1-tr) > trace_bound:
        tr = trace(mps)
        for j in range(pd):
            for k in range(mps[0][j].shape[0]):
                mps[0][j][k] = mps[0][j][k]/(tr)**0.5
    dep_fid_mean.append(dep_fid_sum/(cnot_num[m]+1))
    cnot_fid_mean.append(cnot_fid_sum/(cnot_num[m]))
    final_fid_list.append(dep_fid_total*cnot_fid_total)
    temp = 0 
    for index in range(len(eff_indices)):
        effect = []
        for i in range(N):
            effect.append((np.array([[1, 0], [0, 1]]) + \
                          (-1)**eff_indices[index][i]*\
                          np.array([[1, 0], [0, -1]]))/2)
        temp += eff_meas(mps, effect)
    prob_noisy.append(temp)
    # for subsequent sampling
    psi[instance], _, _ = dep_site(psi[instance], np.arange(m), kraus_meas,
                                   svd_tol, kraus_bound)
    psi[instance] = renorm(psi[instance]) # renormalising
    phi[instance], _, _ = dep_site(phi[instance], np.arange(m), kraus_meas,
                                   svd_tol, kraus_bound)
    phi[instance] = renorm(phi[instance]) # renormalising
tic = time.perf_counter()
clock = (tic - tac)/60
print('%s instances implemented in ' % instance_num + str(round(clock, 2)) +
      ' minutes')
print('final fidelity', dep_fid_total*cnot_fid_total,
      dep_fid_total, cnot_fid_total)
np.savez('./%sQ_%sinst_bounds%s_%s_ov%s' %(m, instance_num, kraus_bound,
                                        bond_bound,
                                        int(100*sum(E)/instance_num)),
         E=E, P_psi=P_psi, P_phi=P_phi, prob_noisy=prob_noisy,
         dep_fid_mean=dep_fid_mean, cnot_fid_mean=cnot_fid_mean,
         final_fid_list=final_fid_list)

sample_num = sample_max - sample_increment 
while abs_av_quasi > error_bound:
    E_noisy = []
    E_quasi = []
    error_noisy = []
    error_quasi = []
    relative_noisy = []
    relative_quasi = []
    sample_num += sample_increment
    for instance in range(instance_num):
        E_noisy.append([])
        E_quasi.append([])
        print('Sampling from instance', instance+1, 'of', instance_num)
        for _ in range(sample_num):
            # sampling from the noisy circuit
            E_noisy[instance].append(np.random.choice(2, 1, 
                    p=[1-prob_noisy[instance], prob_noisy[instance]])[0])
            # sampling for the quasi prob. method
            k = np.random.choice(len(frame)**m, 1, p=P_psi[instance])[0]
            l = np.random.choice(len(frame)**m, 1, p=P_phi[instance])[0]
            k = list(np.base_repr(k, len(frame)))
            l = list(np.base_repr(l, len(frame)))
            while len(k)<m:
                k.insert(0, 0)
            while len(l)<m:
                l.insert(0, 0)
            prob = 1
            for i in range(m):
                prob *= pinvT[int(k[i])][int(l[i])]
            E_quasi[instance].append(prob)
        error_noisy.append(
                abs(E[instance] - 2*sum(E_noisy[instance])/sample_num+1))
        error_quasi.append(
                abs(E[instance] - sum(E_quasi[instance])/sample_num))
        relative_noisy.append(
                abs(E[instance] - 2*sum(E_noisy[instance])/sample_num+1)\
                /E[instance])
        relative_quasi.append(
                abs(E[instance] - sum(E_quasi[instance])/sample_num)\
                /E[instance])
    abs_av_quasi = sum(error_quasi)/instance_num
    rel_av_quasi = sum(relative_quasi)/instance_num
    abs_av_noisy = sum(error_noisy)/instance_num
    rel_av_noisy = sum(relative_noisy)/instance_num
    toc = tic
    tic = time.perf_counter()
    clock = (tic - toc)/60 # time overlapse in minutes
    print('%sK samples from each instance obtained in ' % int(sample_num/1000)\
          + str(round(clock, 2)) + ' minutes')

    ########################## histogram
    columns = np.linspace(0,
                          max(max(error_quasi), max(error_noisy)),
                          column_num+1)
    plt.style.use('default')
    n3 = plt.hist([error_quasi, error_noisy], columns, 
                  label=['quasiprob.', 'Bell-basis circ.'], color=['r', 'b'])
    plt.legend(loc='upper right')
    plt.xlabel('Absolute error \n Averages: ' + 
               str(round(abs_av_quasi, 4)) + ', ' + 
               str(round(abs_av_noisy, 4)))
    plt.ylabel('Number of instances')
    plt.title(str(m) + '-qubit states, ' + str(instance_num) +
              ' instances, %sK samples, av. ov. %s \n bounds: (' \
              %(int(sample_num/1000), round(sum(E)/instance_num, 2)) +
              str(int(kraus_bound)) + ', ' + str(int(bond_bound)) +
              '), %s gate noise, %s meas. noise' %(noise_unit, noise_meas) +
              '\n dep. fid. %s, cnot fid. %s, final fid. %s' \
              %(round(dep_fid_total, 4), round(cnot_fid_total, 4),
                round(dep_fid_total*cnot_fid_total, 4)))
    plt.savefig('/home/leo/Documents/Python/TN/swap_test/histograms/locpur/%sQ_%sinst_%sKsamp_ov%s_bounds%s-%s_abs' \
                %(m, instance_num, int(sample_num/1000),
                  int(100*sum(E)/len(E)), kraus_bound,
                  bond_bound), bbox_inches='tight')
    plt.show()

tic = time.perf_counter()
clock = (tic - tac)/60 # time overlapse in minutes
print('Total time overlapse: ' + str(round(clock, 2)) + ' minutes')
error = []
error.append(E)
error.append(error_quasi)
error.append(error_noisy)
error.append(relative_quasi)
error.append(relative_noisy)
error = np.array(error)
np.save('./%sQ_%sinst_%sKsamp_bounds%s-%s' %(m, instance_num, int(sample_num/1000),
                              kraus_bound, bond_bound), error)
