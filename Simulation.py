from operator import index
import random as rand
import math
import cvxpy as cvx
import mosek
import numpy as np
from random import *
from matplotlib import pyplot as plt
import copy
import time
import pandas as pd
from munkres import Munkres

MAXBAND = 20
MAXPOWER = 2e+2
MAXA = 1.0
UAVNUM = 4
USERNUM = 60

def Euclidean(p, q):
    return math.sqrt(math.pow(p[0]-q[0],2) + math.pow(p[1]-q[1],2) + math.pow(p[2]-q[2],2))

class cTask():
    def __init__(self):
        self.phi = 0.14
        self.alpha = 0.1
        self.A = MAXA

class cDevice():
    def __init__(self, _es):
        self.task = cTask()
        self.f = 1e+1
        self.kapa = 1
        self.pos = _es
        self.beta = MAXA/2 # optimization param.
        self.subbandIndex = 0 # optimization param.
        self.power = MAXPOWER/2 # optimization param.

    def printPOS(self):
        print(self.pos[0], ' ', self.pos[1])
        return

class cUAV():
    def __init__(self, _es):
        self.kapa = 1e-10
        self.pos = _es
        self.fs = 2*MAXPOWER
        self.deviceIdx = []

    def printPOS(self):
        print(self.pos)
        return

def GetDataRate(_iUav, _iUser, _uavs, _users):
    dis = Euclidean(_uavs[_iUav].pos, _users[_uavs[_iUav].deviceIdx[_iUser]].pos)
    g0 = 40
    ik = 0.005
    g = g0 * math.pow(dis, -2) * math.exp(ik * dis)
    Noise = 174
    I = 0
    omega = 5e+2
    for i in range(UAVNUM):
        if i == _iUav: continue
        for j in range(len(_uavs[i].deviceIdx)):
            if _users[_uavs[i].deviceIdx[j]].subbandIndex == _users[_uavs[_iUav].deviceIdx[_iUser]].subbandIndex:
                tmpDis = Euclidean(_uavs[_iUav].pos, _users[_uavs[i].deviceIdx[j]].pos)
                tmpG = g0 * math.pow(tmpDis, -2) * math.exp(ik * tmpDis)
                I += _users[_uavs[i].deviceIdx[j]].power * tmpG
    I *= 10
    SINR = (_users[_uavs[_iUav].deviceIdx[_iUser]].power * g) /  (Noise + I)
    result = omega * math.log2(1 + SINR)
    #print(result, "=", I)
    return result

def GetPreDataRate(_iUav, _iUser, _uavs, _users):
    dis = Euclidean(_uavs[_iUav].pos, _users[_uavs[_iUav].deviceIdx[_iUser]].pos)
    g0 = 40
    ik = 0.005
    g = g0 * math.pow(dis, -2) * math.exp(ik * dis)
    Noise = 174
    I = 0
    omega = 5e+2
    for i in range(UAVNUM):
        if i == _iUav: continue
        for j in range(len(_uavs[i].deviceIdx)):
            if _users[_uavs[i].deviceIdx[j]].subbandIndex == _users[_uavs[_iUav].deviceIdx[_iUser]].subbandIndex:
                tmpDis = Euclidean(_uavs[_iUav].pos, _users[_uavs[i].deviceIdx[j]].pos)
                tmpG = g0 * math.pow(tmpDis, -2) * math.exp(ik * tmpDis)
                I += _users[_uavs[i].deviceIdx[j]].power * tmpG
    I *= 10
    SINR = (_users[_uavs[_iUav].deviceIdx[_iUser]].power * g) /  (Noise + I)
    result = omega * math.log2(1 + SINR)

    CumInters = 0
    for i in range(UAVNUM):
        if i == _iUav: continue
        tmpDis = Euclidean(_uavs[i].pos, _users[_uavs[_iUav].deviceIdx[_iUser]].pos)
        tmpG = g0 * math.pow(tmpDis, -2) * math.exp(ik * tmpDis)
        CumInters += _users[_uavs[_iUav].deviceIdx[_iUser]].power * tmpG
    result -= CumInters
    return result

def TotalUtility(_uavs, _users):
    alpha = 3
    result = 0
    iUav = 0
    for _uav in _uavs:
        for i in range(len(_uav.deviceIdx)):
            result += _users[_uav.deviceIdx[i]].kapa * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].task.alpha * (_users[_uav.deviceIdx[i]].task.A - _users[_uav.deviceIdx[i]].beta) + 1e-2*(_users[_uav.deviceIdx[i]].power * _users[_uav.deviceIdx[i]].beta)/ GetDataRate(iUav, i, _uavs, _users) + _uav.kapa * _uav.fs * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].task.alpha * _users[_uav.deviceIdx[i]].beta
            result += 4 * _users[_uav.deviceIdx[i]].power * _users[_uav.deviceIdx[i]].beta / GetDataRate(iUav, i, _uavs, _users)
        iUav += 1
    return result * 0.5

def EachUtility(_uavs, _iUAV, _users):
    alpha = 3
    _uav = _uavs[_iUAV]
    result = 0
    for i in range(len(_uav.deviceIdx)):
        result += _users[_uav.deviceIdx[i]].kapa * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].task.alpha * (_users[_uav.deviceIdx[i]].task.A - _users[_uav.deviceIdx[i]].beta) + 1e-2*(_users[_uav.deviceIdx[i]].power * _users[_uav.deviceIdx[i]].beta)/ GetDataRate(_iUAV, i, _uavs, _users) + _uav.kapa * _uav.fs * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].task.alpha * _users[_uav.deviceIdx[i]].beta
        result += 4 * _users[_uav.deviceIdx[i]].power * _users[_uav.deviceIdx[i]].beta / GetDataRate(_iUAV, i, _uavs, _users)
    return result * 0.5

def TotalRate(_uavs, _users):
    result = 0
    for iUav in range(UAVNUM):
        for iUser in range(len(_uavs[iUav].deviceIdx)):
            result += GetDataRate(iUav, iUser, _uavs, _users)
    print("TotalRate: ", result)
    return result

def EachRate(_uavs, _iUAV, _users):
    result = 0
    for iUser in range(len(_uavs[_iUAV].deviceIdx)):
        result += GetDataRate(_iUAV, iUser, _uavs, _users)
    return result

def TaskOffloading(_uavs, _users):
    iUav = 0
    for _uav in _uavs:
        for i in range(len(_uav.deviceIdx)):
            qbeta = cvx.Variable(shape=(1,), pos=True)

            qO = _users[_uav.deviceIdx[i]].kapa * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].task.alpha * (_users[_uav.deviceIdx[i]].task.A - qbeta) + 1e-2*(_users[_uav.deviceIdx[i]].power * qbeta)/ GetDataRate(iUav, i, _uavs, _users) + _uav.kapa * _uav.fs * _users[_uav.deviceIdx[i]].f * _users[_uav.deviceIdx[i]].task.alpha * qbeta
            qLoc = (_users[_uav.deviceIdx[i]].task.alpha * (_users[_uav.deviceIdx[i]].task.A - qbeta)) / _users[_uav.deviceIdx[i]].f
            qRem = qbeta/GetDataRate(iUav, i, _uavs, _users) + (_users[_uav.deviceIdx[i]].task.alpha * qbeta) / _uav.fs

            constraints = [
                qLoc <= _users[_uav.deviceIdx[i]].task.phi,
                qRem <= _users[_uav.deviceIdx[i]].task.phi,
                qbeta <= _users[_uav.deviceIdx[i]].task.A,
            ]

            problem = cvx.Problem(cvx.Minimize(qO), constraints)
            problem.solve(solver=cvx.MOSEK)
            
            _users[_uav.deviceIdx[i]].beta = qbeta.value[0]
        iUav += 1
    return

def SubbandAssignment(_uavs, _users):
    _iUav = 0 
    for _uav in _uavs:
        isDones = [False for _ in range(len(_uav.deviceIdx))]
        isBandDones = [False for _ in range(MAXBAND)]
        while True:
            #_uav.Print()
            isDone = True
            for id in range(len(_uav.deviceIdx)):
                if isDones[id] == False:
                    isDone = False
                    if isBandDones[_users[_uav.deviceIdx[id]].subbandIndex]:
                        for bd in range(MAXBAND):
                            if isBandDones[bd] == False:
                                _users[_uav.deviceIdx[id]].subbandIndex = bd
                                break
            if isDone: break

            # Caculating user-all band
            dQ = [[] for _ in range(MAXBAND)]
            for _iUser in range(len(_uav.deviceIdx)):
                if isDones[_iUser]: continue
                initBand = _users[_uav.deviceIdx[_iUser]].subbandIndex
                indexQ = initBand
                valueQ = GetDataRate(_iUav, _iUser, _uavs, _users)
                for b in range(MAXBAND):
                    if isBandDones[b] or b == initBand: continue
                    _users[_uav.deviceIdx[_iUser]].subbandIndex = b
                    if(valueQ < GetDataRate(_iUav, _iUser, _uavs, _users)):
                        indexQ =  b
                        valueQ = GetDataRate(_iUav, _iUser, _uavs, _users)
                dQ[indexQ].append(_iUser)
                _users[_uav.deviceIdx[_iUser]].subbandIndex = initBand
            
            #print(dQ)
            # Caculating band-all user
            for dq in range(MAXBAND):
                if len(dQ[dq]) == 0 or isBandDones[dq]: continue
                indexQ = dQ[dq][0]
                _users[_uav.deviceIdx[indexQ]].subbandIndex = dq
                valueQ = GetPreDataRate(_iUav, indexQ, _uavs, _users)
                for dqq in dQ[dq]:
                    if indexQ == dqq: continue
                    _users[_uav.deviceIdx[dqq]].subbandIndex = dq
                    if valueQ < GetPreDataRate(_iUav, dqq, _uavs, _users):
                        indexQ = dqq
                        valueQ = GetPreDataRate(_iUav, dqq, _uavs, _users)
                _users[_uav.deviceIdx[indexQ]].subbandIndex = dq
                isDones[indexQ] = True
                isBandDones[dq] = True
        _iUav += 1
    return

def PowerControl(_uavs, _users):
    varphi = 1e-2
    omega = 5e+2
    sigma = 174

    totalNum = len(_users)

    idxUAV = []
    idxUSER = []
    idxBand = []
    idxBeta = []
    idxPower = []
    idxGain = []
    for i in range(UAVNUM):
        for j in range(len(_uavs[i].deviceIdx)):
            idxUAV.append(i)
            idxUSER.append(j)
            idxBand.append(_users[_uavs[i].deviceIdx[j]].subbandIndex)
            idxBeta.append(_users[_uavs[i].deviceIdx[j]].beta)
            idxPower.append(_users[_uavs[i].deviceIdx[j]].power)
            
            dis = Euclidean(_uavs[i].pos, _users[_uavs[i].deviceIdx[j]].pos)
            g0 = 40
            ik = 0.005
            g = g0 * dis**(-2) * math.exp(ik * dis)
            idxGain.append(g)

    qGG = []
    qTMP = []
    deltaTMP = []
    for i in range(totalNum):
        qgg = 0
        qtmp = 0
        for j in range(totalNum):
            if idxUAV[i] == idxUAV[j]: continue
            if idxBand[i] == idxBand[j]:
                dis = Euclidean(_uavs[idxUAV[i]].pos, _users[_uavs[idxUAV[j]].deviceIdx[idxUSER[j]]].pos)
                g0 = 40
                ik = 0.005
                g = g0 * dis**(-2) * math.exp(ik * dis)
                qgg += g
                qtmp += idxPower[j]*g
        qGG.append(qgg)
        qTMP.append(qtmp)
        deltaTMP.append(qGG[i] / (math.log(2) * (qTMP[i] + sigma)))

    #########################################################

    qp = cvx.Variable(shape=(totalNum,), pos=True)
    P_max = MAXPOWER * np.ones(totalNum)
    P_min = 5 * np.ones(totalNum)
    ome_var = (varphi*omega) * np.ones(totalNum)
    qSigma = sigma * np.ones(totalNum)

    qPG = []
    for i in range(totalNum):
        qPG.append(cvx.sum(cvx.hstack(idxGain[j]*qp[j] for j in range(totalNum) if idxBand[i] == idxBand[j])))
        #qPG.append(cvx.sum(cvx.hstack(idxGain[j]*idxPower[j] for j in range(totalNum) if idxBand[i] == idxBand[j])))
    qR = cvx.log(cvx.hstack(qPG) + qSigma) - (cvx.log(qTMP + qSigma) + cvx.multiply(deltaTMP, (qp - cvx.hstack(idxPower))))
    #qR = cvx.log(cvx.hstack(qPG) + qSigma) - cvx.log(qTMP + qSigma)
    qR = omega * qR

    qE =  cvx.multiply(cvx.multiply(cvx.hstack(idxPower), cvx.hstack(idxBeta)), cvx.hstack(qR**(-1)))
    #qE =  cvx.multiply(cvx.multiply(qp, cvx.hstack(idxBeta)), cvx.hstack(qR**(-1)))

    constraints = [
        cvx.hstack(qR) >= cvx.multiply(idxBeta, ome_var**(-1)) ,
        qp <= P_max,
        qp >= P_min,
        cvx.sum(qp) <= 60 * MAXPOWER / 2
    ]
    
    problem = cvx.Problem(cvx.Minimize(cvx.sum(qE)), constraints)
    problem.solve(solver= cvx.MOSEK, gp=False)
    
    for i in range(len(qp.value)):
        _users[_uavs[idxUAV[i]].deviceIdx[idxUSER[i]]].power = qp.value[i]
    return

def Deployment(_uavs, _users):
    g0 = 40
    ik = 0.005

    idxUAV = []
    idxUSER = []
    idxPower = []
    idxBeta = []
    idxBand = []
    for i in range(UAVNUM):
        for j in range(len(_uavs[i].deviceIdx)):
            idxUAV.append(i)
            idxUSER.append(j)
            idxBeta.append(_users[_uavs[i].deviceIdx[j]].beta)
            idxPower.append(_users[_uavs[i].deviceIdx[j]].power)
            idxBand.append(_users[_uavs[i].deviceIdx[j]].subbandIndex)

    Pos_max = 250 * np.ones(shape=(2, ))
    Pos_min = -250 * np.ones(shape=(2, ))

    iUav = 0
    for _uav in _uavs:
        qPos = cvx.Variable(shape=(2,))

        qdis = []
        cdis = []
        for j in range(len(_uav.deviceIdx)):
            tmpUAV = [_uav.pos[0], _uav.pos[1]]
            tmpUSER = [_users[_uav.deviceIdx[idxUSER[j]]].pos[0], _users[_uav.deviceIdx[idxUSER[j]]].pos[1]]
            qdis.append(cvx.sum((cvx.hstack(qPos) - cvx.hstack(tmpUSER))**2) + 2500)
            cdis.append((tmpUAV[0]-tmpUSER[0])**2 + (tmpUAV[1]-tmpUSER[1])**2 + 2500)

        qPG = []
        for i in range(len(_uav.deviceIdx)):
            qPG.append((qdis[i] * cvx.exp(ik * cvx.sqrt(cdis[i])))/ (idxPower[i] * g0))

        constraints = [
            qPos <= Pos_max,
            qPos >= Pos_min,
        ]
        problem = cvx.Problem(cvx.Minimize(cvx.sum(qPG)), constraints)
        problem.solve(solver= cvx.MOSEK, gp=False) 

        _uav.pos[0] = qPos.value[0]
        _uav.pos[1] = qPos.value[1] 
        iUav += 1
    return

def DeploymentCentered(_uavs, _users):
    for i in range(UAVNUM):
        centered = [0, 0]
        for j in range(len( _uavs[i].deviceIdx)):
            centered[0] += _users[_uavs[i].deviceIdx[j]].pos[0]
            centered[1] += _users[_uavs[i].deviceIdx[j]].pos[1]
        centered[0] /= len( _uavs[i].deviceIdx)
        centered[1] /= len( _uavs[i].deviceIdx)
        _uavs[i].pos[0] = centered[0]
        _uavs[i].pos[1] = centered[1]

# def Clustering(_uavs, _users):
#     # Clustering based Balanced-K-means
#     clusterNum = len(_uavs)
#     dataNum = len(_users)

#     clusterSize = int(dataNum/clusterNum)

#     datas = []
#     # create data
#     for i in range(dataNum):
#         datas.append([_users[i].pos[0], _users[i].pos[1]])
#     # initialize centroid
#     centroids = []
#     for _ in range(clusterSize):
#         for uav in _uavs:
#             centroids.append(uav.pos)
#     # Main Loop for Balanced K-Means
#     max_iter = 100
#     for _ in range(max_iter):
#         # Calculate edge weights
#         G = []
#         for i in range(dataNum):
#             node = []
#             for j in range(dataNum):
#                 node.append(Euclidean([datas[i][0], datas[i][1], 0], [centroids[j][0], centroids[j][1], 0]))
#             G.append(node)
#         # Solve Assignment Problem
#         m = Munkres()
#         indexes = m.compute(G)
#         # Calculate new centroid locations
#         newCentroids = [[0.0,0.0] for _ in range(clusterNum)]
#         for row, column in indexes:
#             newCentroids[column%clusterNum][0] += datas[row][0]
#             newCentroids[column%clusterNum][1] += datas[row][1]
#         for i in range(clusterNum):
#             newCentroids[i][0] = newCentroids[i][0]/clusterSize
#             newCentroids[i][1] = newCentroids[i][1]/clusterSize
#         # if not change centroid
#         if centroids[0]==newCentroids[0] and centroids[1]==newCentroids[1] and centroids[2]==newCentroids[2]:
#             break
#         # else continue
#         for i in range(clusterSize):
#             for j in range(clusterNum):
#                 centroids[clusterNum*i + j] = newCentroids[j]
    
#     for row, column in indexes:
#         value = G[row][column]
#         _uavs[column%clusterNum].deviceIdx.append(row)
#         #print(f'({row}, {column%clusterNum}) -> {value}')    
#     return

def InitialSetting(_uavs, _users):
    for uav in _uavs:
        id = 0
        for ue in uav.deviceIdx:
            _users[ue].beta = MAXA/2
            _users[ue].subbandIndex = id
            _users[ue].power = MAXPOWER/2
            id += 1
    return

if __name__ == '__main__':
    for _ in range(1):
        uavs = [cUAV([-125, 125, 50]), cUAV([125, 125, 50]), cUAV([-125, -125, 50]), cUAV([125, -125, 50])]
        users = [cDevice([rand.randrange(-250, 250), rand.randrange(-250, 250), 0]) for _ in range(USERNUM)]
#        Clustering(uavs, users)
        InitialSetting(uavs, users)
        DeploymentCentered(uavs, users)
        
        for _ in range(2):
            TaskOffloading(uavs, users)
            SubbandAssignment(uavs, users)
            PowerControl(uavs, users)
            Deployment(uavs, users)