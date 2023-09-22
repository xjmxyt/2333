# %%
import pandas as pd
import numpy as np
import json
import tqdm
import random
from utils import read_data_from_file

base_dir = '/home/xjm/MoveSim/data/traffic'

def gen_eid_dict():
    """生成eid和id之间的映射关系
    """ 
    edge_df = pd.read_csv(base_dir+'/edges.csv')
    id2eid = dict(zip(edge_df.iloc[:,0], edge_df.iloc[:,2]))
    eid2id = dict(zip(edge_df.iloc[:,2], edge_df.iloc[:,0]))
    # generate id to longitude and latitude location
    id2long = dict(zip(edge_df.iloc[:,0], (edge_df.iloc[:,3]+edge_df.iloc[:,5])/2))
    id2lat = dict(zip(edge_df.iloc[:,0], (edge_df.iloc[:,4]+edge_df.iloc[:,6])/2))
    id2loc = {key: {'long': id2long[key], 'lat': id2lat[key]} for key in id2long}
    with open(base_dir+'/id2loc.json', 'w') as f:
        json.dump(id2loc, f)   
    with open('../data/traffic/id2eid.json', 'w') as f:
        json.dump(id2eid, f)
    f.close()
    with open('../data/traffic/eid2id.json', 'w') as f:
        json.dump(eid2id, f)
    f.close()    

def gen_traj(x:pd.DataFrame, seq_len=48, interval_len=4, alpha=0.4):
    """从速度不为0的点开始生成

    Args:
        x (pd.DataFrame): _description_
        seq_len (int, optional): 序列长度. Defaults to 48.
        interval_len (int, optional): 生成序列的间隔长度. Defaults to 4.
        alpha (float, optional): 满足不为0的数量. Defaults to 0.4.
    """ 
    temp_df = x[x['speed']!=0]
    #print(len(temp_df))
    if len(temp_df) == 0:
        return None
    start_pos = list(temp_df['timestamp'])[0]  
    ans = []
    for i in range(start_pos, len(x)-seq_len, interval_len):
        seq = x.iloc[i: i+seq_len]
        if (len(seq[seq['speed']!=0]))/seq_len > alpha:
            ans.append(list(seq['eid']).copy())
    if len(ans) == 0:
        return None
    return np.array(ans)

def get_eid_dic(file_name='../data/traffic/eid2id.json')->dict:
    with open(file_name, 'r') as f:
        eid2id = json.load(f) 
    newdict = {}
    for key in eid2id.keys():
        newdict[int(key)] = eid2id[key]
    eid2id = newdict    
    return eid2id

def reindex_traj(data:np.array, outfile):
    """将eid重新编码

    Args:
        data (np.array): 按照原来eid的数据
        outfile (_type_): 输出的文件名

    Returns:
        _type_: np.array
    """    
    eid2id = get_eid_dic()
    # train_data = read_data_from_file('../data/traffic/real.data')
    def f(x):
        return eid2id[x]
    applyall = np.vectorize(f)
    data = applyall(data)
    np.savetxt(outfile, data, fmt="%d")
          
def preprocess(file):
    df = pd.read_csv(file, index_col=0)
    
    # generate edge info
    edge_cols = ['eid', 'lon1', 'lat1', 'lon2', 'lat2']
    edge_df = df[edge_cols]
    edge_df.drop_duplicates(inplace=True)
    edge_df.reset_index(inplace=True)
    edge_df.to_csv('../data/traffic/edges.csv')
    gen_eid_dict()
    
    select_cols = ['vehicle_id', 'speed', 'timestamp', 'eid']
    df = df[select_cols]
    #df = df[df['speed']!=0]
    print("eid count: ", len(df['eid'].unique()))
    print("total count: ", len(df['eid']))
    # exclude long-stop point
    # mean roll for traj
    def roll_mean(x:pd.DataFrame):
        x['speed'] = x['speed'].rolling(window=3, center=True).mean()
        x['speed'] = x['speed'].fillna(0)
        return x
    roll_df = df.groupby('vehicle_id').apply(roll_mean)
    # generate trajectory of each vehicle
    grouped = roll_df.groupby('vehicle_id')
    vehicle2traj = dict()
    for key, group in tqdm.tqdm(grouped):
        traj = gen_traj(group)
        if traj is not None:
            vehicle2traj[key] = traj
        else:
            pass
    # save data
    veh2ind = {}
    trajs = []
    start_ind = 0
    for key in vehicle2traj.keys():
        veh2ind[str(key)] = {'start_ind':start_ind, 'len':len(vehicle2traj[key])}
        start_ind = start_ind + len(vehicle2traj[key])
        trajs.append(vehicle2traj[key])
    trajs = np.vstack(trajs)
    np.save('../data/traffic/trajs.npy', trajs)
    with open('../data/traffic/veh2ind', 'w') as f:
        json.dump(veh2ind, f)
    print("successfully preprocess data")

def gen_real_data():
    data = np.load('../data/traffic/trajs.npy')
    np.savetxt('../data/traffic/real_ori.data', data, fmt="%d")
    print("real data generated")
    return data
    
def gen_fake_data(seq_len=48, nums=1000000):
    edge_df = pd.read_csv('../data/traffic/edges.csv', index_col=0)
    eids = list(edge_df['eid'])
    result = []
    for i in range(nums):
        fake_data = random.sample(eids, seq_len)
        result.append(fake_data)
    result = np.array(result)
    np.savetxt('../data/traffic/dispre_ori.data', result, fmt="%d")
    print("fake data generated")
    return result

def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        lis.append(l)
    return lis 

def gen_node_gps():
    edge_df = pd.read_csv('../data/traffic/edges.csv', index_col=0)
    edge_df['mid_lon'] = (edge_df.iloc[:, 2] + edge_df.iloc[:, 4])/2
    edge_df['mid_lat'] = (edge_df.iloc[:, 3] + edge_df.iloc[:, 5])/2
    edge_df['mid_lon'].fillna(edge_df['mid_lon'].mean(), inplace=True)
    edge_df['mid_lat'].fillna(edge_df['mid_lat'].mean(), inplace=True)
    new_df = edge_df[['mid_lon', 'mid_lat']]
    np.savetxt("../data/traffic/gps", new_df.values, fmt="%.3f")

def gen_val_data():
    with open("../data/traffic/veh2ind", 'r') as f:
        vec2ind = json.load(f) 
        f.close()
    vehicles = list(vec2ind.keys())
    vehicles.sort()
    pos = int(len(vehicles) * 0.8)
    ind = vec2ind[vehicles[pos]]['start_ind']
    real_data = read_file('../data/traffic/real.data')
    val_data = real_data[ind:]
    np.savetxt("../data/traffic/val.data", val_data, fmt="%d")
    
    
    
if __name__ == '__main__':
    #preprocess("../data/geolife/final.csv")
    #gen_real_data()
    #gen_fake_data()
    # real_data = read_file('../data/traffic/real_ori.data')
    # fake_data = read_file('../data/traffic/dispre_ori.data')
    # print("real data shape is ", len(real_data))
    # print("fake data shape is ", len(fake_data))
    # reindex_traj(real_data, '../data/traffic/real.data')
    # reindex_traj(fake_data, '../data/traffic/dispre.data')
    # gen_node_gps()
    gen_eid_dict()
    
    
    




