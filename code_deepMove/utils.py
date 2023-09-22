# coding: utf-8

import os
import pdb
import time
import json
import shutil
import logging
import hashlib
import numpy as np
import geohash
from math import radians, cos, sin, asin, sqrt



def hash_args(*args):
    # json.dumps will keep the dict keys always sorted.
    string = json.dumps(args, sort_keys=True, default=str)  # frozenset
    return hashlib.md5(string.encode()).hexdigest()


def use_gpu(idx):
    # 0->2,3->1,1->3,2->0
    map = {0:2, 3:1, 1:3, 2:0}
    return map[idx]


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc


def get_gps(gps_file):
    with open(gps_file) as f:
        gpss = f.readlines()
    X = []
    Y = []
    for gps in gpss:
        x, y = float(gps.split()[0]), float(gps.split()[1])
        X.append(x)
        Y.append(y)
    return X, Y


def read_data_from_file(fp)->np.array:
    """
    read a bunch of trajectory data from txt file
    :param fp:
    :return:
    """
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[int(t) for t in tmp]]
    return np.asarray(dat, dtype='int64')


def write_data_to_file(fp, dat):
    """Write a bunch of trajectory data to txt file.
    Parameters
    ----------
    fp : str
        file path of data
    dat : list
        list of trajs
    """
    with open(fp, 'w') as f:
        for i in range(len(dat)):
            line = [str(p) for p in dat[i]]
            line_s = ' '.join(line)
            f.write(line_s + '\n')


def read_logs_from_file(fp):
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[float(t) for t in tmp]]
    return np.asarray(dat, dtype='float')


def prep_workspace(workspace, datasets, oridata):
    """
    prepare a workspace directory
    :param workspace:
    :param oridata:
    :return:
    """
    data_path = '/data/stu/yangzeyu/trajgen'
    if not os.path.exists(data_path+'/%s/%s' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s' % (datasets,workspace))
    if not os.path.exists(data_path+'/%s/%s/data' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s/data' % (datasets,workspace))
    if not os.path.exists(data_path+'/%s/%s/logs' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s/logs' % (datasets,workspace))
    if not os.path.exists(data_path+'/%s/%s/figs' % (datasets,workspace)):
        os.mkdir(data_path+'/%s/%s/figs' % (datasets,workspace))
    '''
    shutil.copy("../data/%s/real.data" %
                oridata, "../%s/%s/data/real.data" % (datasets,workspace))
    shutil.copy("../data/%s/val.data" %
                oridata, "../%s/%s/data/val.data" % (datasets,workspace))
    shutil.copy("../data/%s/test.data" %
                oridata, "../%s/%s/data/test.data" % (datasets,workspace))
    shutil.copy("../data/%s/dispre_10.data" %
                oridata, "../%s/%s/data/dispre.data" % (datasets,workspace))
    '''
    with open(data_path+'/%s/%s/logs/loss.log' % (datasets,workspace), 'w') as f:
        pass

    with open(data_path+'/%s/%s/logs/jsd.log' % (datasets,workspace), 'w') as f:
        pass
    

def get_workspace_logger(datasets):
   
    data_path = '../data'  
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s")
    fh = logging.FileHandler(data_path+'/%s/logs/all.log' % (datasets), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def geodistance(lng1,lat1,lng2,lat2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

def get_geohash_distance(hash1, hash2):
    lat1, lon1 = geohash.decode(hash1)
    lat2, lon2 = geohash.decode(hash2)
    return geodistance(lon1, lat1, lon2, lat2)

if __name__ == '__main__':
    print(get_geohash_distance('wtw350', 'wtw34c'))