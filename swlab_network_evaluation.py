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
from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# You can generate an API token from the "API Tokens Tab" in the UI
token = "REzqvTvY6F9Og5pvb-xPCFXpybOnZPPCrLZwTrfS01-fdsKb4FeUeIUcXnEZxWC6ZPE2GSyRDs3xGqSyfo7OZA=="
org = "networkinglab"
bucket = "NetEv"

# 3.5e+9 Hz(3.5GHz), Mbit/s
MAXBAND = 2e+6
MAXPOWER = 4e+3
UAVNUM = 1
USERNUM = 25

meandownDataRates = 0
meanupEfficiency = 0
meandownEfficiency = 0
meanenergyEfficiency = 0
meanlatency = 0

basket = [[],[],[],[],[]]

def Euclidean(p, q):
    return math.sqrt(math.pow(p[0]-q[0],2) + math.pow(p[1]-q[1],2) + math.pow(p[2]-q[2],2))

class cDevice():
    def __init__(self, _es):
        self.pos = _es
        self.band = 0.01 # optimization param.
        self.power = 0.01 # optimization param.
        self.downDataRate = 0
        self.upEfficiency = 0
        self.downEfficiency = 0
        self.energyEfficiency = 0
        self.latency = 0

    def printPOS(self):
        print(self.pos[0], ' ', self.pos[1])
        return

class cUAV():
    def __init__(self, _es):
        self.pos = _es
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
    SINR = (_users[_uavs[_iUav].deviceIdx[_iUser]].power * MAXPOWER * g) /  (Noise)
    result = _users[_uavs[_iUav].deviceIdx[_iUser]].band * MAXBAND * math.log2(1 + SINR)
    return result

def GetUplinkRate(_iUav, _iUser, _uavs, _users):
    dis = Euclidean(_uavs[_iUav].pos, _users[_uavs[_iUav].deviceIdx[_iUser]].pos)
    g0 = 40
    ik = 0.005
    g = g0 * math.pow(dis, -2) * math.exp(ik * dis)
    Noise = 174
    SINR = (0.01 * MAXPOWER * g) /  (Noise)
    result = _users[_uavs[_iUav].deviceIdx[_iUser]].band * MAXBAND * math.log2(1 + SINR)
    return result

def TotalRate(_uavs, _users):
    result = 0
    for iUav in range(UAVNUM):
        tmpres = 0
        for iUser in range(len(_uavs[iUav].deviceIdx)):
            tmpres += GetDataRate(iUav, iUser, _uavs, _users)
        print(tmpres)
        result += tmpres
    print("TotalRate: ", result)
    return result

def EachRate(_uavs, _iUAV, _users):
    result = 0
    for iUser in range(len(_uavs[_iUAV].deviceIdx)):
        result += GetDataRate(_iUAV, iUser, _uavs, _users)
    return result

def CalResult(_uavs, _users):
    for iUav in range(UAVNUM):
        for iUser in range(len(_uavs[iUav].deviceIdx)):
            #print("Downlink Data Rate:\t{: .4f}(Mbit/s),\tDownlink Peak Spectral Efficiency:\t{: .4f}(bit/s/Hz)".format(GetDataRate(iUav, iUser, _uavs, _users), GetDataRate(iUav, iUser, _uavs, _users)/35))
            #print("Uplink Data Rate:\t{: .4f}(Mbit/s),\tUplink Peak Spectral Efficiency:\t{: .4f}(bit/s/Hz)".format(GetUplinkRate(iUav, iUser, _uavs, _users), GetUplinkRate(iUav, iUser, _uavs, _users)/27))
            #print("Network energy efficiency:\t{: .4f}(Small/Micro),\tMinimum User Plane Latency:\t{: .4f}(ms)".format(GetDataRate(iUav, iUser, _uavs, _users)/120, 5.5e+3/GetDataRate(iUav, iUser, _uavs, _users)))
            _users[iUser].downDataRate = GetDataRate(iUav, iUser, _uavs, _users)
            _users[iUser].upEfficiency = GetUplinkRate(iUav, iUser, _uavs, _users)/27
            _users[iUser].downEfficiency = GetDataRate(iUav, iUser, _uavs, _users)/35
            _users[iUser].energyEfficiency = GetDataRate(iUav, iUser, _uavs, _users)/120
            _users[iUser].latency = 5e+3/GetDataRate(iUav, iUser, _uavs, _users)
    return

def MeanResult(_uavs, _users):
    global meandownDataRates
    global meanupEfficiency
    global meandownEfficiency
    global meanenergyEfficiency
    global meanlatency
    for iUav in range(UAVNUM):
        for iUser in range(len(_uavs[iUav].deviceIdx)):
            meandownDataRates += _users[iUser].downDataRate
            meanupEfficiency += _users[iUser].upEfficiency
            meandownEfficiency += _users[iUser].downEfficiency
            meanenergyEfficiency += _users[iUser].energyEfficiency
            meanlatency += _users[iUser].latency
    meandownDataRates /= USERNUM
    meanupEfficiency /= USERNUM
    meandownEfficiency /= USERNUM
    meanenergyEfficiency /= USERNUM
    meanlatency /= USERNUM

    basket[0].append(meandownDataRates)
    basket[1].append(meanupEfficiency)
    basket[2].append(meandownEfficiency)
    basket[3].append(meanenergyEfficiency)
    basket[4].append(meanlatency)

    print("Downlink Data Rate:\t{: .4f}(Mbit/s),\tDownlink Peak Spectral Efficiency:\t{: .4f}(bit/s/Hz)".format(meandownDataRates, meandownEfficiency))
    print("Uplink Peak Spectral Efficiency:\t{: .4f}(bit/s/Hz)".format(meanupEfficiency))
    print("Network energy efficiency:\t{: .4f}(Small/Micro),\tMinimum User Plane Latency:\t{: .4f}(ms)".format(meanenergyEfficiency, meanlatency))
    return

def BandOptimizaion(_uavs, _users):
    iUav = 0
    for _uav in _uavs:
        totalNum = len(_uav.deviceIdx)
        qband = cvx.Variable(shape=(totalNum,), pos=True)

        qPG = []
        for i in range(totalNum):
            dis = Euclidean(_uavs[iUav].pos, _users[_uavs[iUav].deviceIdx[i]].pos)
            g0 = 40
            ik = 0.005
            g = g0 * dis**(-2) * math.exp(ik * dis)
            Noise = 174
            SINR = (_users[_uavs[iUav].deviceIdx[i]].power * MAXPOWER * g) /  (Noise)
            qPG.append(qband[i] * MAXBAND * math.log2(1 + SINR))
        qPG = cvx.hstack(qPG)
        #meanPG = cvx.abs(cvx.hstack(qPG) - cvx.sum(qPG)/len(_uav.deviceIdx)**(0.8))

        constraints = [
            qPG <= 670 * np.ones(len(_uav.deviceIdx)),
            qPG - cvx.sum(qPG)/len(_uav.deviceIdx) >= -10 * np.ones(len(_uav.deviceIdx)),
            qPG - cvx.sum(qPG)/len(_uav.deviceIdx) <= 10 * np.ones(len(_uav.deviceIdx)),
            #qPG >= 600 * np.ones(len(_uav.deviceIdx)),
            qband <= np.ones(len(_uav.deviceIdx)),
            cvx.sum(qband) <= 1,
        ]

        problem = cvx.Problem(cvx.Maximize(cvx.sum(qPG)), constraints)
        problem.solve(solver=cvx.MOSEK)
        
        for i in range(totalNum):
            _users[_uavs[iUav].deviceIdx[i]].band = qband.value[i]
        iUav += 1
    return

def PowerOptimizaion(_uavs, _users):
    iUav = 0
    for _uav in _uavs:
        totalNum = len(_uav.deviceIdx)
        qpow = cvx.Variable(shape=(totalNum,), pos=True)

        qPG = []
        for i in range(totalNum):
            dis = Euclidean(_uavs[iUav].pos, _users[_uavs[iUav].deviceIdx[i]].pos)
            g0 = 40
            ik = 0.005
            g = g0 * dis**(-2) * math.exp(ik * dis)
            Noise = 174
            SINR = (_users[_uavs[iUav].deviceIdx[i]].band * MAXPOWER * g) /  (Noise)
            qPG.append(qpow[i] * MAXBAND * math.log2(1 + SINR))
        qPG = cvx.hstack(qPG)
        #meanPG = cvx.abs(cvx.hstack(qPG) - cvx.sum(qPG)/len(_uav.deviceIdx)**(0.8))

        constraints = [
            qPG <= 670 * np.ones(len(_uav.deviceIdx)),
            qPG - cvx.sum(qPG)/len(_uav.deviceIdx) >= -10 * np.ones(len(_uav.deviceIdx)),
            qPG - cvx.sum(qPG)/len(_uav.deviceIdx) <= 10 * np.ones(len(_uav.deviceIdx)),
            #qPG >= 600 * np.ones(len(_uav.deviceIdx)),
            qpow <= np.ones(len(_uav.deviceIdx)),
            cvx.sum(qpow) <= 1,
        ]

        problem = cvx.Problem(cvx.Maximize(cvx.sum(qPG)), constraints)
        problem.solve(solver=cvx.MOSEK)
        
        for i in range(totalNum):
            _users[_uavs[iUav].deviceIdx[i]].power = qpow.value[i]
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

def Clustering(_uavs, _users):
    # Clustering based Balanced-K-means
    clusterNum = len(_uavs)
    dataNum = len(_users)

    clusterSize = int(dataNum/clusterNum)

    datas = []
    # create data
    for i in range(dataNum):
        datas.append([_users[i].pos[0], _users[i].pos[1]])
    # initialize centroid
    centroids = []
    for _ in range(clusterSize):
        for uav in _uavs:
            centroids.append(uav.pos)
    # Main Loop for Balanced K-Means
    max_iter = 100
    for _ in range(max_iter):
        # Calculate edge weights
        G = []
        for i in range(dataNum):
            node = []
            for j in range(dataNum):
                node.append(Euclidean([datas[i][0], datas[i][1], 0], [centroids[j][0], centroids[j][1], 0]))
            G.append(node)
        # Solve Assignment Problem
        m = Munkres()
        indexes = m.compute(G)
        # Calculate new centroid locations
        newCentroids = [[0.0,0.0] for _ in range(clusterNum)]
        for row, column in indexes:
            newCentroids[column%clusterNum][0] += datas[row][0]
            newCentroids[column%clusterNum][1] += datas[row][1]
        for i in range(clusterNum):
            newCentroids[i][0] = newCentroids[i][0]/clusterSize
            newCentroids[i][1] = newCentroids[i][1]/clusterSize
        # if not change centroid
        if centroids[0]==newCentroids[0] and centroids[1]==newCentroids[1] and centroids[2]==newCentroids[2]:
            break
        # else continue
        for i in range(clusterSize):
            for j in range(clusterNum):
                centroids[clusterNum*i + j] = newCentroids[j]
    
    for row, column in indexes:
        value = G[row][column]
        _uavs[column%clusterNum].deviceIdx.append(row)
        #print(f'({row}, {column%clusterNum}) -> {value}')    
    return

def InitialSetting(_uavs, _users):
    for uav in _uavs:
        id = 0
        for ue in uav.deviceIdx:
            _users[ue].band = 1/len(uav.deviceIdx)
            _users[ue].power = 1/len(uav.deviceIdx)
            id += 1
    return

def SendingDate():
    global meandownDataRates
    global meanupEfficiency
    global meandownEfficiency
    global meanenergyEfficiency
    global meanlatency
    with InfluxDBClient(url="http://163.180.116.47:8086", token=token, org=org) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        data = "mem,tagKet=mean downDataRate=" + str(meandownDataRates)
        write_api.write(bucket, org, data)
        data = "mem,tagKet=mean upEfficiency=" + str(meanupEfficiency)
        write_api.write(bucket, org, data)
        data = "mem,tagKet=mean downEfficiency=" + str(meandownEfficiency)
        write_api.write(bucket, org, data)
        data = "mem,tagKet=mean energyEfficiency=" + str(meanenergyEfficiency)
        write_api.write(bucket, org, data)
        data = "mem,tagKet=mean latency=" + str(meanlatency)
        write_api.write(bucket, org, data)


if __name__ == '__main__':
    USERNUM = int(input("테스트 사용자 수 결정: "))
    startTime = time.time()
    print("=================TEST START=================")
    while(time.gmtime(time.time() - startTime).tm_min != 1): #tm_sec
        uavs = [cUAV([0, 0, 0])]
        users = [cDevice([rand.randrange(-125, 125), rand.randrange(-125, 125), 0]) for _ in range(USERNUM)]
        for i in range(USERNUM):
            uavs[0].deviceIdx.append(i)
        #Clustering(uavs, users)
        #DeploymentCentered(uavs, users)
        InitialSetting(uavs, users)
        for _ in range(5):
            BandOptimizaion(uavs, users)
            PowerOptimizaion(uavs, users)
        CalResult(uavs, users)
        MeanResult(uavs, users)
        print("=====================================")
        SendingDate()
        time.sleep(1)
    print("=================TEST END=================")
    print("==========================================")
    print("=================RESULTS==================")

    print("Downlink Peak Spectral Efficiency:\t{: .4f}(bit/s/Hz)".format(sum(basket[2])/len(basket[2])))
    print("Uplink Peak Spectral Efficiency:\t{: .4f}(bit/s/Hz)".format(sum(basket[1])/len(basket[1])))
    print("Network energy efficiency:\t\t{: .4f}(Small/Micro)".format(sum(basket[3])/len(basket[3])))
    print("Downlink Data Rate:\t\t\t{: .4f}(Mbit/s)".format(sum(basket[0])/len(basket[0])))
    print("Minimum User Plane Latency:\t\t{: .4f}(ms)".format(sum(basket[4])/len(basket[4])))