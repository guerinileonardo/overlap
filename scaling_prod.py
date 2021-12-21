#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:31:25 2021

@author: leo
"""

import numpy as np
import time
from tools import pauli6_frame_tn, random_prod_mps, overlap4, povm_meas, \
    rotate_mps, comp_frame
import argparse
import os

if __name__=="__main__":
        # Add argparser
        parser = argparse.ArgumentParser(description='Set parameters')
        parser.add_argument('--instance', default=0)
        args = parser.parse_args()
        instance_number = int(args.instance)
        m_range = [1, 2, 3, 4, 5, 6, 7, 8] # number of swapped qubits
        pd = 2 # dimension of the particles
        ninstances = 10 # average over this number of instances 
        sample_init = [0, 100, 100, 100, 100, 100, 200, 700, 1500]
        sample_increment = 50
        error_bound = 0.05 # max error for the weighted average of our method 
        nrep = 5
        column_num = 25 # number of columns in the histograms
        
        ################# starters ########################################
        
        frame_pauli, pinvT = pauli6_frame_tn()
        frame_comp = comp_frame()
        tac = time.perf_counter() # set counter
        
        sample_final_quasi = []
        quasi = []
        for m in m_range:
            print(m)
            ############################################## quasiprob method
            av_err = 1
            nsamples = sample_init[m] - sample_increment
            while av_err > error_bound:
                nsamples += sample_increment
                err = []
                for _ in range(nrep):
                    psi = random_prod_mps(m, pd)
                    phi = rotate_mps(psi, np.pi/6)
                    ov = overlap4(psi, phi)
                    pr = povm_meas(psi, frame_pauli)
                    Ppsi_pauli = np.reshape(pr, (1, 6**m))[0]
                    pr = povm_meas(phi, frame_pauli)
                    Pphi_pauli = np.reshape(pr, (1, 6**m))[0]            
                    summ = 0
                    k = list(np.random.choice(len(frame_pauli)**m, nsamples, p=Ppsi_pauli))
                    l = list(np.random.choice(len(frame_pauli)**m, nsamples, p=Pphi_pauli))
                    for k0 in k:
                        for l0 in l:
                            kk = list(np.base_repr(k0, len(frame_pauli)))
                            ll = list(np.base_repr(l0, len(frame_pauli)))
                            while len(kk) < m:
                                kk.insert(0, 0)
                            while len(ll) < m:
                                ll.insert(0, 0)
                            tprod = 1
                            for i in range(m):
                                tprod *= pinvT[int(kk[i])][int(ll[i])]
                            summ += tprod
                    err.append(abs(ov-summ/((nsamples)**2))/ninstances)
                av_err = np.mean(err)
                st_dev = np.std(err)
                print('quasi samples', nsamples, 'st dev', st_dev, 'av', av_err)
            quasi.append([nsamples, av_err, st_dev])
            sample_final_quasi.append(nsamples)

        if not os.path.exists('./scaling_prod'):
            os.mkdir('./scaling_prod')
        sample_final = np.array([quasi, sample_final_quasi])
        np.save(f'./scaling_prod/prod_scaling_{"_".join([str(q) for q in m_range])}_{instance_number}inst_{nrep}rep', sample_final)

