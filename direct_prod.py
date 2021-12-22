#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates a random pair of pure product states in LPDO form and
compares the estimation of their overlap by the quasiprob. method
and the method by Elben et. al.
"""

import numpy as np
import qutip as qt
import time
import itertools as it
from tools import comp_frame, hamming, kraus_dep, overlap, pauli6_frame, \
    povm_meas, random_prod_lpdo, random_rotation_site, rotate_lpdo
import argparse
import os

if __name__=="__main__":
        # Add argparser
        parser = argparse.ArgumentParser(description='Set parameters')
        parser.add_argument('--instance', default=0)
        args = parser.parse_args()

        instance_number = int(args.instance)

        pd = 2 # dimension of the particles (for now, it has to be 2)
        error_bound = 0.05
        n_instances = 1 # number of instances
        N_U = 100 # number of unitaries
        N_M = 100
        m_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        ov_bound = 1e-1 # lower bound for the overlap between the random states
        noise_unit = 0.005 # noise in the implementation of the unitary
        noise_meas = 0.01 # noise in the implementation of the measurement

        svd_tol = 1e-8
        kraus_bound = 4

        column_num = 25 # number of columns in the histograms

        tac = time.perf_counter() # set counter
        print(f"Instance {instance_number}")
        
        ################# starters ########################################

        # hadamard = np.zeros((2, 2), complex)
        hadamard = [[1, 1], [1, -1]] / np.sqrt(2)
        U = []
        U.append(hadamard)
        U.append(np.array([[1, 0], [0, 1j]]))

        frame_comp = comp_frame()
        frame_pauli, pinvT = pauli6_frame()
        kraus_meas = kraus_dep(noise_meas)
        error_inn = []
        error_quasi = []
        for m in m_range:
            print(m, 'qubits')
            # creates all operators of the comp. basis on m qubits
            effect = []
            for index in range(2**m):
                label = list(np.base_repr(index, len(frame_comp)))
                while len(label) < m:
                    label.insert(0, 0)
                effect.append([])
                for i in range(m):
                    effect[index].append(frame_comp[int(label[i])])
            for instance in range(n_instances):
                fid = 0
                # creates random mps
                psi = random_prod_lpdo(m, pd)
                # making sure the fidelity is not too small
                phi = rotate_lpdo(psi, np.pi/6)
                ov = overlap(psi, phi)

                ######################## innsbruck method (numerator only)
                # applies a random local unitary on psi
                for _ in range(N_U):
                    for q in range(m):
                        UU = np.array(qt.rand_unitary_haar(2))
                        psi = random_rotation_site(psi, q, UU)
                        phi = random_rotation_site(phi, q, UU)
                    # measuring psi_U
                    pr = povm_meas(psi, frame_comp)
                    Ppsi_comp = np.reshape(pr, (1, 2**m))[0]
                    pr = povm_meas(phi, frame_comp)
                    Pphi_comp = np.reshape(pr, (1, 2**m))[0]
                    s1 = np.random.choice(2**m, N_M, p=Ppsi_comp)
                    t1 = np.random.choice(2**m, N_M, p=Pphi_comp)
                    for s, ss in it.product(s1, t1):
                        j0 = list(np.binary_repr(s, m))
                        j1 = list(np.binary_repr(ss, m))
                        fid += (-2)**(-hamming(j0, j1))
                error_inn.append(abs(ov-(2**m)*fid/(N_U*N_M**2)))

                ###################### quasiprob method
                pr = povm_meas(psi, frame_pauli)
                Ppsi_pauli = np.reshape(pr, (1, 6**m))[0]
                pr = povm_meas(phi, frame_pauli)
                Pphi_pauli = np.reshape(pr, (1, 6**m))[0]
                summ0 = 0
                k = list(np.random.choice(len(frame_pauli)**m, N_M*N_U, p=Ppsi_pauli))
                l = list(np.random.choice(len(frame_pauli)**m, N_M*N_U, p=Pphi_pauli))
                for k0 in k:
                    for l0 in l:
                        kk = list(np.base_repr(k0, len(frame_pauli)))
                        ll = list(np.base_repr(l0, len(frame_pauli)))
                        while len(kk)<m:
                            kk.insert(0, 0)
                        while len(ll)<m:
                            ll.insert(0, 0)
                        tprod = 1
                        for i in range(m):
                            tprod *= pinvT[int(kk[i])][int(ll[i])]
                        summ0 += tprod
                error_quasi.append(abs(ov-summ0/((N_M*N_U)**2)))

        if not os.path.exists('./direct_data_prod'):
            os.mkdir('./direct_data_prod')
        data = []
        data.append(error_inn)
        data.append(error_quasi)
        data = np.array(data)
        np.save(f'./direct_data_prod/prod_qubits_{"_".join([str(q) for q in m_range])}_{instance_number}inst_nu{N_U}_nm{N_M}', data)

