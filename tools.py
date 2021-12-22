#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toolbox for noisy circuit simulation and overlap estimation.
"""

import numpy as np
import scipy as sp
import qutip as qt
from ncon import ncon
from tenpy.linalg.svd_robust import svd
import itertools as it


I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# COPY tensor
copy = np.zeros((2, 2, 2), complex)
copy[0][0][0] = 1
copy[1][1][1] = 1

# XOR tensor
xor = np.zeros((2, 2, 2), complex)
xor[0][0][0] = 1
xor[0][1][1] = 1
xor[1][1][0] = 1
xor[1][0][1] = 1

# hadamard gate
hadamard = np.zeros((2, 2), complex)
hadamard = [[1, 1], [1, -1]]/np.sqrt(2)

# cincio gate
cincio = np.array(qt.phasegate(np.pi/4).dag()*qt.snot())


def bb_block(lpdo, q, kraus_unit, svd_tol, kraus_bound, bond_bound,
                     dep_fid_sum=0, dep_fid_total=1,
                     cnot_fid_sum=0, cnot_fid_total=1):
    """
    Returns a block of gates that perform a noisy SWAP between qubits q and
    q+m in the circuit, that is, between the q-th qubit of each input state.
    The block involves only first-neighbour CNOTs and Hadamard gates.
    """
    m = int(len(lpdo)/2)
    # checking the labels of qubits involved in the first neighbours reduction
    ind = first_neighbours_indices(q, q+m)
    # writing the lpdo in canonical form
    lpdo = qr_left(lpdo, min(ind[0]))
    lpdo = qr_right(lpdo, max(ind[0]))
    prev = min(ind[0])
    for label in ind:
        # writing the lpdo in canonical form
        if prev < min(label):
            lpdo = qr_left(lpdo, min(label), prev)
            prev = min(label)
        elif prev > min(label):
            lpdo = qr_right(lpdo, max(label), prev+1)
            prev = min(label)
        # modeling the noise in the unitary gate
        lpdo, dep_fid_sum, dep_fid_total = \
          dep_site(lpdo, label, kraus_unit, svd_tol, kraus_bound,
                   dep_fid_sum, dep_fid_total)
        # performing a cnot: contraction of copy on control and xor on target
        lpdo, cnot_fid_sum, cnot_fid_total = \
          cnot(lpdo, label, svd_tol, bond_bound, cnot_fid_sum,
               cnot_fid_total)
    # performing a Hadamard
    lpdo[q] = ncon((hadamard, lpdo[q]),
                  ([-1, 1],  [1, -2, -3, -4]))
    return lpdo, dep_fid_sum, dep_fid_total, cnot_fid_sum, cnot_fid_total


def comp_frame(rcond=1e-15):
    """
    Returns the 1-qubit computational basis. 
    """
    comp = []
    comp.append([[1, 0], [0, 0]])
    comp.append([[0, 0], [0, 1]])
    comp = np.array(comp)
    return comp


def cnot(lpdo, label, svd_tol=1e-10, bond_bound=1024, cnot_fid_sum=0,
         cnot_fid_total=1):
    """
    Implements a CNOT in the neighbour sites of 'lpdo' defined in 
    label=[control, target]. 
    """
    pd = 2
    cnot_fid_sum += 1
    lpdo_temp = lpdo[:min(label[0], label[1])]
    if label[0] < label[1]:
        temp = ncon((copy, xor, lpdo[label[0]], lpdo[label[1]]),
                    ([-1, 1, 2], [-3, 1, 3], [2, -2, -5, 4], [3, -4, 4, -6]))
        tempp = np.zeros((temp.shape[0]*temp.shape[1]*temp.shape[4], 
                          temp.shape[2]*temp.shape[3]*temp.shape[5]), complex)
        for i0, i1, i2, i3, i4, i5 in it.product(range(2),
          range(temp.shape[1]), range(2), range(temp.shape[3]),
          range(temp.shape[4]), range(temp.shape[5])):
            tempp[i0*temp.shape[1]*temp.shape[4] + i1*temp.shape[4]+i4]\
              [i2*temp.shape[3]*temp.shape[5] + i3*temp.shape[5]+i5] = \
                                          temp[i0][i1][i2][i3][i4][i5]
        U, S, V = svd(tempp)
        for i in range(S.shape[0]):
            if S[i] > svd_tol:
                dim_trunc = i+1
        if dim_trunc > bond_bound:
            dim_trunc = bond_bound
            S2 = S**2
            cnot_fid_sum += -(sum((S2)[dim_trunc:])/sum(S2))
            cnot_fid_total *= 1-(sum((S2)[dim_trunc:])/sum(S2))
        # S is merged with U
        U = ((U.transpose()[:dim_trunc]).transpose())\
            .dot(np.diag(S[:dim_trunc]))
        V = V[:dim_trunc]
        lpdo_temp.append(np.zeros((pd, temp.shape[1], temp.shape[4], dim_trunc),
                                 complex))
        for i0, i1, i4, ii in it.product(range(temp.shape[0]),
          range(temp.shape[1]), range(temp.shape[4]), range(dim_trunc)):
            lpdo_temp[label[0]][i0][i1][i4][ii] = \
              U[i0*temp.shape[1]*temp.shape[4]+i1*temp.shape[4]+i4][ii]
        lpdo_temp.append(np.zeros((pd, temp.shape[3], dim_trunc, temp.shape[5]),
                                 complex))
        for i2, i3, i5, ii in it.product(range(temp.shape[2]), \
          range(temp.shape[3]), range(temp.shape[5]), range(dim_trunc)):
            lpdo_temp[label[1]][i2][i3][ii][i5] = \
              V[ii][i2*temp.shape[3]*temp.shape[5]+i3*temp.shape[5]+i5]
    elif label[1] < label[0]:
        temp = ncon((xor, copy, lpdo[label[1]], lpdo[label[0]]),
                    ([-1, 1, 2], [-3, 1, 3], [2, -2, -5, 4], [3, -4, 4, -6]))
        tempp = np.zeros((temp.shape[0]*temp.shape[1]*temp.shape[4], 
                          temp.shape[2]*temp.shape[3]*temp.shape[5]), complex)
        for i0, i1, i2, i3, i4, i5 in it.product(range(pd),
          range(temp.shape[1]), range(2), range(temp.shape[3]),
          range(temp.shape[4]), range(temp.shape[5])):
            tempp[i0*temp.shape[1]*temp.shape[4]+i1*temp.shape[4]+i4]\
              [i2*temp.shape[3]*temp.shape[5]+i3*temp.shape[5]+i5] = \
              temp[i0][i1][i2][i3][i4][i5]
        U, S, V = svd(tempp)
        for i in range(S.shape[0]):
            if S[i] > svd_tol:
                dim_trunc = i+1
        if dim_trunc > bond_bound:
            dim_trunc = bond_bound
            S2 = S**2
            cnot_fid_sum += -(sum((S2)[dim_trunc:])/sum(S2))
            cnot_fid_total *= 1-(sum((S2)[dim_trunc:])/sum(S2))
        # S is merged with U
        U = ((U.transpose()[:dim_trunc]).transpose())\
            .dot(np.diag(S[:dim_trunc]))
        V = V[:dim_trunc]
        lpdo_temp.append(np.zeros((pd, temp.shape[1], temp.shape[4], dim_trunc),
                                 complex))
        for i0, i1, i4, ii in it.product(range(temp.shape[0]), \
          range(temp.shape[1]), range(temp.shape[4]), range(dim_trunc)):
            lpdo_temp[label[1]][i0][i1][i4][ii] = \
              U[i0*temp.shape[1]*temp.shape[4]+i1*temp.shape[4]+i4][ii]
        lpdo_temp.append(np.zeros((pd, temp.shape[3], dim_trunc, temp.shape[5]),
                                 complex))
        for i2, i3, i5, ii in it.product(range(temp.shape[2]),
          range(temp.shape[3]), range(temp.shape[5]), range(dim_trunc)):
            lpdo_temp[label[0]][i2][i3][ii][i5] = \
              V[ii][i2*temp.shape[3]*temp.shape[5]+i3*temp.shape[5]+i5]
    for site in range(max(label[0], label[1])+1, len(lpdo)):
        lpdo_temp.append(lpdo[site])
    return lpdo_temp, cnot_fid_sum, cnot_fid_total


def dep_site(lpdo, site_list, kraus, svd_tol=1e-10, kraus_bound=4,
             dep_fid_sum=0, dep_fid_total=1):
    """
    Returns the LPDO mixed with white noise on the sites in site_list. 
    """
    temp = []
    dep_fid_sum += 1
    for i in range(len(lpdo)):
        if i in site_list:
            tens = ncon((kraus, lpdo[i]),
                        ([-3, -1, 1], [1, -2, -4, -5]))
            # reshaping
            tens2 = np.zeros((tens.shape[0]*tens.shape[3]*tens.shape[4], 
                          tens.shape[1]*tens.shape[2]), complex)
            for i0, i1, i2, i3, i4 in it.product(range(tens.shape[0]), 
                                                 range(tens.shape[1]),
                                                 range(tens.shape[2]), 
                                                 range(tens.shape[3]),
                                                 range(tens.shape[4])):
                tens2[i0*tens.shape[3]*tens.shape[4] + i3*tens.shape[4]+i4]\
                    [i1*tens.shape[2] + i2] = tens[i0][i1][i2][i3][i4]
            # kraus dimension control via SVD
            U, S, V = svd(tens2)
            for k in range(S.shape[0]):
                if S[k] > svd_tol:
                    dim_trunc = k+1
            if dim_trunc > kraus_bound:
                dim_trunc = kraus_bound
                S2 = S**2
                dep_fid_sum -= sum((S2)[dim_trunc:])/sum(S2)
                dep_fid_total *= 1-(sum((S2)[dim_trunc:])/sum(S2))
            U = (U.transpose()[:dim_trunc]).transpose()\
              .dot(np.diag(S[:dim_trunc]))
            temp.append(np.zeros((tens.shape[0], dim_trunc,
                                  tens.shape[3], tens.shape[4]),
                                    complex))
            for j0, jj, j3, j4 in it.product(range(tens.shape[0]),
                                             range(dim_trunc),
                                             range(tens.shape[3]),
                                             range(tens.shape[4])):
                temp[i][j0][jj][j3][j4] = \
                  U[j0*tens.shape[3]*tens.shape[4]+j3*tens.shape[4]+j4][jj]           
        else:
            temp.append(lpdo[i])
    return temp, dep_fid_sum, dep_fid_total


def eff_meas(mpo, effect, imag_tol=1e-12):
    '''
    Calculates trace(rho*effect), where rho is a state in LPDO form and
    effect is a list of single-site effects being measured at each site.
    '''
    N = len(mpo)
    q = []
    for i in range(N):
        q.append(ncon((effect[i], mpo[i], mpo[i].conjugate()),
                      ([2, 1], [1, 3, -1, -3], [2, 3, -2, -4])))
    ind_con = []
    for i in range(N-1):
        ind_con.append([2*i+1, 2*i+2, 2*i+3, 2*i+4])
    ind_con.append([2*N-1, 2*N, 1, 2])
    pr = ncon(tuple(q), tuple(ind_con))
    if abs(np.imag(pr)) < imag_tol:
        pr = np.real(pr)
    return pr


def first_neighbours_indices(control, target):
    '''
    Returns the indices involved in the decomposition of a CNOT between qubits
    'control' and 'target' into first-neighbours CNOTs.
    '''
    ind = []
    distance = abs(control-target)
    if control < target:
        for i in range(distance-1):
            ind.append([control+i+1, control+i])
            ind.append([control+i, control+i+1])
        ind.append([control+distance-1, target])
        for i in range(distance-2, -1, -1):
            ind.append([control+i, control+i+1])
            ind.append([control+i+1, control+i])
    elif control > target:
        for i in range(distance-1):
            ind.append([control-i-1, control-i])
            ind.append([control-i, control-i-1])
        ind.append([control-distance+1, target])
        for i in range(distance-2, -1, -1):
            ind.append([control-i, control-i-1])
            ind.append([control-i-1, control-i])
    return(ind)


def hamming(a, b):
    """
    Hamming distance between the lists a and b of 0's and 1's.
    """
    a = np.array(a, int)
    b = np.array(b, int)
    return float(sum(np.mod(a-b, 2)))


def ist_block(lpdo, q, kraus_unit, svd_tol, kraus_bound, bond_bound,
                    dep_fid_sum=0, dep_fid_total=1,
                    cnot_fid_sum=0, cnot_fid_total=1):
    '''
    Returns a block of gates that perform a noisy SWAP between qubits q and
    q+m in the circuit, that is, between the q-th qubit of each input state.
    The block involves only first-neighbour CNOTs and Hadamard gates.
    '''
    m = int((len(lpdo)-1)/2)
    # cnot(q, q+m)
    ind = first_neighbours_indices(q, q+m)
    # writing the lpdo in canonical form
    lpdo = qr_left(lpdo, min(ind[0]))
    lpdo = qr_right(lpdo, max(ind[0]))
    for label in ind:
        lpdo = qr_left(lpdo, min(label))
        lpdo = qr_right(lpdo, max(label))
        # modeling the noise in the unitary gate
        lpdo, dep_fid_sum, dep_fid_total = \
          dep_site(lpdo, label, kraus_unit, svd_tol, kraus_bound,
                   dep_fid_sum, dep_fid_total)
        # performing a cnot: contraction of copy on control and xor on target
        lpdo, cnot_fid_sum, cnot_fid_total = \
          cnot(lpdo, label, svd_tol, bond_bound, cnot_fid_sum,
               cnot_fid_total)
    # cnot(q+m, 0)
    ind = first_neighbours_indices(q+m, 0)
    for label in ind:
        lpdo = qr_left(lpdo, min(label))
        lpdo = qr_right(lpdo, max(label))
        # modeling the noise in the unitary gate
        lpdo, dep_fid_sum, dep_fid_total = \
          dep_site(lpdo, label, kraus_unit, svd_tol, kraus_bound,
                   dep_fid_sum, dep_fid_total)
        # performing a cnot: contraction of copy on control and xor on target
        lpdo, cnot_fid_sum, cnot_fid_total = \
          cnot(lpdo, label, svd_tol, bond_bound, cnot_fid_sum,
               cnot_fid_total)
    # Cincio dagger on qubit 0
    lpdo[0] = ncon((cincio.conjugate().transpose(), lpdo[0]),
                    ([-1, 1],  [1, -2, -3, -4]))
    # cnot(0, q)
    ind = first_neighbours_indices(0, q)
    for label in ind:
        lpdo = qr_left(lpdo, min(label))
        lpdo = qr_right(lpdo, max(label))
        # modeling the noise in the unitary gate
        lpdo, dep_fid_sum, dep_fid_total = \
          dep_site(lpdo, label, kraus_unit, svd_tol, kraus_bound,
                   dep_fid_sum, dep_fid_total)
        # performing a cnot: contraction of copy on control and xor on target
        lpdo, cnot_fid_sum, cnot_fid_total = \
          cnot(lpdo, label, svd_tol, bond_bound, cnot_fid_sum,
               cnot_fid_total)
    # Cincio on qubit 0
    lpdo[0] = ncon((cincio, lpdo[0]),
                    ([-1, 1],  [1, -2, -3, -4]))
    # cnot(q+m, 0)
    ind = first_neighbours_indices(q+m, 0)
    for label in ind:
        lpdo = qr_left(lpdo, min(label))
        lpdo = qr_right(lpdo, max(label))
        # modeling the noise in the unitary gate
        lpdo, dep_fid_sum, dep_fid_total = \
          dep_site(lpdo, label, kraus_unit, svd_tol, kraus_bound,
                   dep_fid_sum, dep_fid_total)
        # performing a cnot: contraction of copy on control and xor on target
        lpdo, cnot_fid_sum, cnot_fid_total = \
          cnot(lpdo, label, svd_tol, bond_bound, cnot_fid_sum,
               cnot_fid_total)
    return lpdo, dep_fid_sum, dep_fid_total, cnot_fid_sum, cnot_fid_total


def kraus_dep(noise):
    """
    Defines the Kraus operators of the depolarising channel of a given noise 
    degree.
    """
    Dep = np.zeros((4, 2, 2), complex)
    Dep[0] = np.sqrt((4- 3*noise)/4)*I
    Dep[1] = np.sqrt(noise/4)*X
    Dep[2] = np.sqrt(noise/4)*Y
    Dep[3] = np.sqrt(noise/4)*Z
    return Dep


def meas(lpdo, site, effect, imag_tol=1e-12):
    """
    Calculates trace(rho*effect), where rho is a state in LPDO form and
    'effect' is a single-site effect being measured on 'site'. 
    """
    N = len(lpdo)
    ind_con = []
    # effect
    ind_con = []
    # indices effect
    ind_con.append([4*site+3, 4*site+2])
    # indices lpdo
    for i in range(site):
        ind_con.append([4*i+2, 4*i+1, 4*i-1, 4*i+3])
        ind_con.append([4*i+2, 4*i+1, 4*i, 4*i+4])
    ind_con.append([4*site+2, 4*site+1, 4*site-1, 4*site+4])
    ind_con.append([4*site+3, 4*site+1, 4*site, 4*site+5])
    for i in range(site+1, N):
        ind_con.append([4*i+3, 4*i+2, 4*i, 4*i+4])
        ind_con.append([4*i+3, 4*i+2, 4*i+1, 4*i+5])
    ind_con[1][2] = 4*N
    ind_con[2][2] = 4*N+1
    lpdo_con = []
    lpdo_con.append(effect)
    for tensor in range(len(lpdo)):
        lpdo_con.append(lpdo[tensor])
        lpdo_con.append(lpdo[tensor].conjugate())
    pr = ncon(tuple(lpdo_con), tuple(ind_con))
    if abs(np.imag(pr)) < imag_tol:
        pr = np.real(pr)
    return pr


def overlap(rho, sigma, imag_tol=1e-12):
    """
    Calculates trace(rho*sigma) where rho and sigma are LPDOs.
    """
    N = len(rho)
    lpdo_con = []
    for i in range(N):
        lpdo_con.append(rho[i])
        lpdo_con.append(rho[i].conjugate())
        lpdo_con.append(sigma[i])
        lpdo_con.append(sigma[i].conjugate())
    ind_con = []
    # indices of rho ket
    for i in range(N):
        ind_con.append([8*i+3, 8*i+1, 8*i-3, 8*i+5])
        ind_con.append([8*i+4, 8*i+1, 8*i-2, 8*i+6])
        ind_con.append([8*i+4, 8*i+2, 8*i-1, 8*i+7])
        ind_con.append([8*i+3, 8*i+2, 8*i, 8*i+8])
    ind_con[0][2] = 8*N-3
    ind_con[1][2] = 8*N-2
    ind_con[2][2] = 8*N-1
    ind_con[3][2] = 8*N
    ov = ncon(tuple(lpdo_con), tuple(ind_con))
    if abs(np.imag(ov)) < imag_tol:
        ov = np.real(ov)
    return ov

def pauli6_frame(rcond=1e-15):
    """
    Returns the 1-qubit frame of Pauli measurements and the pseudoinverse of
    its overlap matrix. 
    """
    paulis = []
    paulis.append((I+X)/6)
    paulis.append((I-X)/6)
    paulis.append((I+Y)/6)
    paulis.append((I-Y)/6)
    paulis.append((I+Z)/6)
    paulis.append((I-Z)/6)
    paulis = np.array(paulis)
    # overlap matrix
    T = np.real(ncon((paulis, paulis), ([-1, 2, 3], [-4, 3, 2])))
    # pseudoinverse of T
    pinvT = np.linalg.pinv(T, rcond, hermitian=True)
    return paulis, pinvT

def povm_meas(lpdo, local_povm, imag_tol=1e-12):
    '''
    Calculates the prob distribution (p_a1, ..., p_an) of outcomes (a1, ...,
    an) when measuring n tensor copies of local_povm on the n-qubit state lpdo.
    '''
    lpdo_con = []
    ind_con = []
    for i in range(len(lpdo)):
        lpdo_con.append(lpdo[i])
        ind_con.append([5*i+3, 5*i+5, 5*i+1, 5*i+6])
        lpdo_con.append(lpdo[i].conjugate())
        ind_con.append([5*i+4, 5*i+5, 5*i+2, 5*i+7])
        lpdo_con.append(local_povm)
        ind_con.append([-i-1, 5*i+4, 5*i+3])
    ind_con[-3][3] = 1
    ind_con[-2][3] = 2
    pr = ncon(tuple(lpdo_con), tuple(ind_con))
    if (abs(np.imag(pr))).max() < imag_tol:
        pr = np.real(pr)
    return pr

def qr_left(mpo, site0, start=0):
    for site in range(start, site0):
        temp = mpo[site]
        # reshaping
        temp2 = np.zeros((2*temp.shape[1]*temp.shape[2], temp.shape[3]),
                         complex)
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
            temp2[i0*temp.shape[1]*temp.shape[2]+i1*temp.shape[2]+i2][i3] \
              = temp[i0][i1][i2][i3]
        Q, R = np.linalg.qr(temp2)
        # unreshaping
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
              mpo[site][i0][i1][i2][i3] = \
                Q[i0*temp.shape[1]*temp.shape[2]+i1*temp.shape[2]+i2][i3]
        # integrating R into the next site
        temp = mpo[site+1]
        temp2 = np.zeros((temp.shape[2], 2*temp.shape[1]*temp.shape[3]),
                         complex)
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
            temp2[i2][i0*temp.shape[1]*temp.shape[3]+i1*temp.shape[3]+i3] \
              = temp[i0][i1][i2][i3]
        temp2 = R.dot(temp2)
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
              mpo[site+1][i0][i1][i2][i3] = \
                temp2[i2][i0*temp.shape[1]*temp.shape[3]+i1*temp.shape[3]+i3]
    return mpo

def qr_right(mpo, site0, end=None):
    if end==None:
        end = len(mpo)-1
    for site in range(end, site0, -1):
        temp = mpo[site]
        # reshaping
        temp2 = np.zeros((temp.shape[2], 2*temp.shape[1]*temp.shape[3]),
                         complex)
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
            temp2[i2][i0*temp.shape[1]*temp.shape[3]+i1*temp.shape[3]+i3] \
              = temp[i0][i1][i2][i3]
        Q, R = np.linalg.qr(temp2.transpose())
        Q = Q.transpose()
        R = R.transpose()
        # unreshaping
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
              mpo[site][i0][i1][i2][i3] = \
                Q[i2][i0*temp.shape[1]*temp.shape[3]+i1*temp.shape[3]+i3]
        # integrating R into the next site
        temp = mpo[site-1]
        temp2 = np.zeros((2*temp.shape[1]*temp.shape[2], temp.shape[3]),
                         complex)
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
            temp2[i0*temp.shape[1]*temp.shape[2]+i1*temp.shape[2]+i2][i3] \
              = temp[i0][i1][i2][i3]
        temp2 = temp2.dot(R)
        for i0, i1, i2, i3 in it.product(range(2), range(temp.shape[1]),
                                         range(temp.shape[2]),
                                         range(temp.shape[3])):
              mpo[site-1][i0][i1][i2][i3] = \
                temp2[i0*temp.shape[1]*temp.shape[2]+i1*temp.shape[2]+i2][i3]
    return mpo


def random_lpdo(N, pd, bd_max=None, C=1):
    """
    Returns the a random pure state in form of LPDO, with N sites, pd physical
    dimensions, and 2*bd bond dimension, of a general shape of
    (N, 2, pd, k, bd, bd), where k is the Kraus dimension between the ket and
    the bra of each site. 
    """
    if bd_max==None:
        bd_max = 2**int(N/2)
    A = []
    for i in range(int(N/2)):
        dim1 = min(2**i, bd_max)
        dim2 = min(2**(i+1), bd_max)
        A.append(np.zeros((pd, dim1, dim2), complex))
        for j in range(pd):
            A[i][j] = np.random.randn(dim1, dim2) + \
                C*np.random.randn(dim1, dim2)*1j
    if N/2 - int(N/2) > 0:
        dim = min(2**int(N/2), bd_max)
        A.append(np.zeros((pd, dim, dim), complex))
        for j in range(pd):
            A[int(N/2)][j] = np.random.randn(dim, dim) + \
                C*np.random.randn(dim, dim)*1j
        for i in range(int(N/2)+1, N):
            dim1 = min(2**(N-i), bd_max)
            dim2 = min(2**(N-i-1), bd_max)
            A.append(np.zeros((pd, dim1, dim2), complex))
            for j in range(pd):
                A[i][j] = np.random.randn(dim1, dim2) + \
                    C*np.random.randn(dim1, dim2)*1j
    else:
        for i in range(int(N/2), N):
            dim1 = min(2**(N-i), bd_max)
            dim2 = min(2**(N-i-1), bd_max)
            A.append(np.zeros((pd, dim1, dim2), complex))
            for j in range(pd):
                A[i][j] = np.random.randn(dim1, dim2) + \
                    C*np.random.randn(dim1, dim2)*1j
    kraus = kraus_dep(0)
    lpdo = []
    for i in range(N):
        # each site is representes by a tensor (2, physical, kraus, bond, bond)
        lpdo.append(np.zeros((2, pd, 4, A[i].shape[1], A[i].shape[2]), complex))
        lpdo[i] = ncon((kraus, A[i]), ([-2, -1, 1], [1, -3, -4]))
    tr = trace(lpdo)
    for j in range(pd):
        for k in range(4):
            lpdo[0][j][k] = lpdo[0][j][k]/(tr)**0.5
    return lpdo


def random_prod_lpdo(n, pd):
    """
    Returns the LPDO of the (kl)-th element of the canonical basis of
    the matrix space in dimension (pd)**N
    """
    m1 = np.zeros((1, 2), complex)
    m1[0][0] = 1
    m = np.diag(m1[0])
    lpdo = []
    lpdo.append([])
    lpdo[0].append(m)
    lpdo[0].append(np.zeros((2, 2), complex))
    for site in range(1, n):
        lpdo.append([])
        lpdo[site].append(m)
        lpdo[site].append(np.zeros((2, 2), complex))
    kraus = kraus_dep(0)
    lpdo = np.array(lpdo)
    lpdo2 = []
    for i in range(n):
        # each site is representes by a tensor (physical, kraus, bond, bond)
        lpdo2.append(np.zeros((pd, 4, lpdo[i].shape[1], lpdo[i].shape[2]),
                            complex))
        lpdo2[i] = ncon((kraus, lpdo[i]), ([-2, -1, 1], [1, -3, -4]))
    lpdo3 = []
    for site in range(n):
        lpdo3.append(ncon((np.array(qt.rand_unitary_haar(2)), lpdo2[site]),
                         ([-1, 1], [1, -2, -3, -4])))
    return lpdo3


def random_rotation_site(lpdo, site, rot=[]):
    '''
    Applies a random unitary on the site-th qubit of an LPDO.
    '''
    if rot==[]:
        alpha = 2*np.pi*np.random.rand()
        beta = 2*np.pi*np.random.rand()
        theta = np.pi*np.random.rand()
        mat = np.cos(alpha)*np.cos(beta)*X + np.cos(alpha)*np.sin(beta)*Y + \
            np.sin(alpha)*Z
        rot = sp.linalg.expm(-1j*theta*mat)
    temp = []
    for i in range(site):
        temp.append(lpdo[i])
    temp.append(ncon((rot, lpdo[site]), ([-1, 1], [1, -2, -3, -4])))
    for i in range(site+1, len(lpdo)):
        temp.append(lpdo[i])
    return temp


def renorm(psi):
    '''
    Renormalises the LPDO.
    '''
    tr = 0
    h = -1
    while abs(tr-1) > 1e-6:
        h = np.mod(h+1, len(psi))
        tr = trace(psi)
        for j in range(2):
            for k in range(psi[h][j].shape[0]):
                psi[h][j][k] = psi[h][j][k]/(tr)**0.5
    return psi

def rotate_lpdo(lpdo, theta_max, cnot_on=False):
    '''
    Applies a local random rotation to each site of an LPDO. 
    '''
    temp = []
    for site in range(len(lpdo)):
        theta = np.random.rand()*theta_max
        rot = np.array([[np.cos(theta), -np.sin(theta)],
    	                [np.sin(theta), np.cos(theta)]])
        temp.append(ncon((rot, lpdo[site]), ([-1, 1], [1, -2, -3, -4])))
    if cnot_on==True:
        for site in range(len(lpdo)-1):
            temp2 = cnot(temp, [site, site+1])[0]
    else:
        temp2 = temp
    return temp2

def trace(lpdo):
    '''
    Calculates the trace of an LPDO.
    '''
    N = len(lpdo)
    lpdo_con = []
    ind_con = []
    for i in range(N-1):
        lpdo_con.append(lpdo[i])
        lpdo_con.append(lpdo[i].conjugate())
        ind_con.append([2*i+1, 2*i+2, 2*(N+i)+1, 2*(N+i)+3])
        ind_con.append([2*i+1, 2*i+2, 2*(N+i)+2, 2*(N+i)+4])
    lpdo_con.append(lpdo[N-1])
    lpdo_con.append(lpdo[N-1].conjugate())
    ind_con.append([2*N-1, 2*N, 4*N-1, 2*N+1])
    ind_con.append([2*N-1, 2*N, 4*N, 2*N+2])
    tr = ncon(tuple(lpdo_con), ind_con)
    if abs(np.imag(tr))<1e-12:
        tr = float(np.real(tr))
    return tr
