#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Leonardo Guerini

Estimates the state overlap of a random pair of states using both a noisy
implementation of the improves swap test circuit and the quasiprob. method.

"""

import numpy as np
import time
from TN_tools_locpur4 import pauli6_frame_tn, random_mps, overlap4, eff_meas, \
    dep_site, kraus_dep, rotate_mps, renorm, losalamos_block, meas1
from copy import deepcopy
from ncon import ncon
import argparse
import os

if __name__=="__main__":
        # Add argparser
        parser = argparse.ArgumentParser(description='Set parameters')
        parser.add_argument('--instance', default=0)
        args = parser.parse_args()

        instance_number = int(args.instance)
        m = 2 # number of qubits in each particle
        N = 2*m # number of qubits in the circuit
        pd = 2 # dimension of the particles
        
        instance_num = 1 # number of instances
        sample_num = 300 # initial number of samples
        sample_increment = 100 # increment on the number of samples 
        
        ov_bound = 1e-1 # lower bound for the overlap between the random states
        noise_unit = 0.005 # noise in the implementation of the unitary
        noise_meas = 0.01 # noise in the implementation of the measurement
        
        svd_tol = 1e-8 # bound for discarding singular values in the truncation
        kraus_bound = 140 # upper bound on the kraus dimension between ket and bra
        bond_bound = 48 # max dimension admitted in the SVD decomposition
        trace_bound = 1e-6
        
        error_bound = 0.1 # max error for the weighted average of our method 
        abs_av_quasi = 1 # just a kickstart value, greater than error_bound
        abs_av_noisy = 0 # just a kickstart value
        rel_av_quasi = 1 # just a kickstart value, greater than error_bound
        rel_av_noisy = 0 # just a kickstart value
        sample_max = 1000
        
        column_num = 25 # number of columns in the histograms
        
        ####################################################### starters
        
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
        
        cincio = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                           [(1-1j)/2, (-1+1j)/2]])
        I = np.array([[1, 0], [0, 1]])
        Z = np.array([[1, 0], [0, -1]])

        tac = time.perf_counter() # set counter
        psi = []
        phi = []
        angle = np.pi/12
        E = []
        P_psi = []
        P_phi = []
        max_bond_dim = []
        probIST = []
        dep_fid_mean = []
        cnot_fid_mean = []
        final_fid_list = []
        for instance in range(instance_num):
            print('Implementing instance', instance+1, 'of', instance_num)
            psi.append(random_mps(m, pd, 2))
            phi.append(rotate_mps(psi[instance], 
                                  angle,
                                  cnot_on=False))
            ov = overlap4(psi[-1], phi[-1])
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
        
            ################ implementing the noisy improved swap test circuit
            bd = 1
            dep_fid_sum = 0
            dep_fid_total = 1
            cnot_fid_sum = 0
            cnot_fid_total = 1
            # ancilla ket 0 mps
            m1 = np.zeros((1, bd), complex)
            m1[0][0] = 1
            mps00 = []
            mps00.append(np.zeros((pd, 1, bd), complex))
            mps00[0][0] = m1
            kraus = kraus_dep(0)
            mps20 = []
            mps20.append(np.zeros((pd, 4, mps00[0].shape[1], mps00[0].shape[2]),
                                complex))
            # input state
            mps20[0] = ncon((mps00[0], kraus), ([1, -3, -4], [-2, 1, -1]))
            mpsIST = deepcopy(mps20+psi[instance]+phi[instance])
            # first Cincio gate
            mpsIST[0] = ncon((cincio, mpsIST[0]),
                            ([-1, 1],  [1, -2, -3, -4]))
            for q in range(1, m+1):
                mpsIST, dep_fid_sum, dep_fid_total, cnot_fid_sum, cnot_fid_total = \
                    losalamos_block(mpsIST, q, kraus_unit, svd_tol, kraus_bound,
                                     bond_bound, dep_fid_sum, dep_fid_total,
                                     cnot_fid_sum, cnot_fid_total)
            # last Cincio gate
            mpsIST[0] = ncon((cincio.conjugate().transpose(), mpsIST[0]),
                          ([-1, 1],  [1, -2, -3, -4]))
            probIST.append(meas1(mpsIST, 0, (I+Z)/2))
            opt_noisy2 = abs(ov-2*probIST[instance]+1)
            print('IST error:', opt_noisy2)
            print('Depolarising fidelity:', dep_fid_total)
            print('CNOT fidelity:', cnot_fid_total)
            print('Circuit final fidelity:', dep_fid_total*cnot_fid_total)
           
        tic = time.perf_counter()
        clock = (tic - tac)/60
        print('%s instances implemented in ' % instance_num + str(round(clock, 2)) +
              ' minutes')

        ################################################### sampling
        sample_num = sample_num - sample_increment
        while abs_av_quasi > error_bound:
            E_IST = []
            error_IST = []
            error_quasi = []
            sample_num += sample_increment
            for instance in range(instance_num):
                E_IST.append([])
                print('Sampling from instance', instance+1, 'of', instance_num)
                for _ in range(sample_num):
                    # sampling from the noisy IST circuit
                    E_IST[instance].append(np.random.choice(2, 1, 
                            p=[1-probIST[instance], probIST[instance]])[0])
                    # sampling for the quasi prob. method
                summ = 0
                Qpsi = np.zeros(len(frame)**m)
                Qphi = np.zeros(len(frame)**m)
                k = list(np.random.choice(len(frame)**m, sample_num, p=P_psi[instance]))
                l = list(np.random.choice(len(frame)**m, sample_num, p=P_phi[instance]))
                for i in range(len(frame)**m):
                    Qpsi[i] = k.count(i)/sample_num
                    Qphi[i] = l.count(i)/sample_num
                    for j in range(len(frame)**m):
                        ii = list(np.base_repr(i, len(frame)))
                        jj = list(np.base_repr(j, len(frame)))
                        while len(ii)<m:
                            ii.insert(0, 0)
                        while len(jj)<m:
                            jj.insert(0, 0)
                        prob = 1
                        for a in range(m):
                            prob *= pinvT[int(ii[a])][int(jj[a])]
                        summ += (k.count(i)/sample_num)*prob*(l.count(j)/sample_num)
                error_IST.append(
                        abs(E[instance] - 2*sum(E_IST[instance])/sample_num+1))
                error_quasi.append(abs(E[instance] - summ))
            abs_av_quasi = sum(error_quasi)/instance_num
            abs_av_la = sum(error_IST)/instance_num
            toc = tic
            tic = time.perf_counter()
            clock = (tic - toc)/60 # time overlapse in minutes
            print('%sK samples from each instance obtained in ' % int(sample_num/1000)\
                  + str(round(clock, 2)) + ' minutes')
        
        tuc = time.perf_counter()
        clock = (tuc - tac)/60 # time overlapse in minutes
        print('Total time overlapse: ' + str(round(clock, 2)) + ' minutes')
        
        if not os.path.exists('./IST_data'):
            os.mkdir('./IST_data')
        data = []
        data.append(error_quasi)
        data.append(error_IST)
        data.append([dep_fid_total, cnot_fid_total, dep_fid_total*cnot_fid_total])
        data = np.array(data)
        np.save(f'./IST_data/ist_{m}qubits_{sample_num}samples_{str(int(kraus_bound)) + str(int(bond_bound))}bounds_{instance_number}', data)
