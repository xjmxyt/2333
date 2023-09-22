# %%
import pandas as pd
import numpy as np
import json
import tqdm
import random
import geohash
from utils import read_data_from_file

GEN_PATH = '/home/xjm/MoveSim/data1/traffic_hash1'

def gen_eid_dict(useHash=False):
    """生成eid和id之间的映射关系
    """ 
    edge_df = pd.read_csv(GEN_PATH+'/edges.csv')
    id2eid = dict(zip(edge_df.iloc[:,0], edge_df.iloc[:,2]))
    eid2id = dict(zip(edge_df.iloc[:,2], edge_df.iloc[:,0]))
    id2hash = dict(zip(edge_df.iloc[:,0], edge_df.iloc[:,-1]))
    
    # generate id to longitude and latitude location
    id2long = dict(zip(edge_df.iloc[:,0], (edge_df.iloc[:,3]+edge_df.iloc[:,5])/2))
    id2lat = dict(zip(edge_df.iloc[:,0], (edge_df.iloc[:,4]+edge_df.iloc[:,6])/2))
    id2loc = {key: {'long': id2long[key], 'lat': id2lat[key]} for key in id2long}
    
    if useHash:
        total_hash = set(edge_df.iloc[:,-1])
        hash2num = dict(zip(list(total_hash), range(len(total_hash))))
        temp1 = {}
        for key in eid2id.keys():
            temp1[key] = hash2num[id2hash[eid2id[key]]]
        eid2id = temp1
        # hash和定义的hash编号关系
        id2eid = dict(zip(eid2id.values(), eid2id.keys()))    
        num2hash = dict(zip(hash2num.values(), hash2num.keys()))    
        with open(GEN_PATH+'/hash2num.json', 'w') as f:
            json.dump(hash2num, f)
        with open(GEN_PATH+'/num2hash.json', 'w') as f:
            json.dump(num2hash, f)
        with open(GEN_PATH+'/id2loc.json', 'w') as f:
            json.dump(id2loc, f)        
    with open(GEN_PATH+'/id2eid.json', 'w') as f:
        json.dump(id2eid, f)
    f.close()
    with open(GEN_PATH+'/eid2id.json', 'w') as f:
        json.dump(eid2id, f)
    f.close()    

def gen_traj(x:pd.DataFrame, seq_len=48, interval_len=10, alpha=0.1):
    """从速度不为0的点开始生成

    Args:
        x (pd.DataFrame): _description_
        seq_len (int, optional): 序列长度. Defaults to 48.
        interval_len (int, optional): 生成序列的间隔长度. Defaults to 4.
        alpha (float, optional): 满足不为0的数量. Defaults to 0.4.
    """ 
    temp_df = x[x['speed']!=0]
    #print(temp_df)
    if len(temp_df) == 0:
        return None
    start_pos = list(temp_df['timestamp'])[0]  
    ans = []
    time = []
    for i in range(0, len(x)-seq_len, interval_len):
        seq = x.iloc[i: i+seq_len]
        if (len(seq[seq['speed']!=0]))/seq_len > alpha:
            ans.append(list(seq['eid']).copy())
            time.append(x.iloc[i]['timestamp'])
    if len(ans) == 0:
        return None
    #print(time)
    return np.array(ans), np.array(time)

def get_eid_dic()->dict:
    with open(GEN_PATH+'/eid2id.json', 'r') as f:
        eid2id = json.load(f) 
    newdict = {}
    for key in eid2id.keys():
        newdict[int(key)] = int(eid2id[key])
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
    # train_data = read_data_from_file(GEN_PATH+'/real.data')
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
    def f(x):
        try:
            return geohash.encode(x['lat2'], x['lon2'], precision=6)
        except:
            print(x)
            return "000000"
            
    edge_df['hash'] = edge_df.apply(lambda x: f(x), axis=1)
    edge_df.to_csv(GEN_PATH+'/edges.csv')
    gen_eid_dict(useHash=True)
    
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
    # 每隔k行取数据
    k = 1
    a = []
    for i in range(len(df)):
        if i%k == 0:
            a.append(i)
    df = df.iloc[a]
    print(len(df))
    roll_df = df.groupby('vehicle_id').apply(roll_mean)
    # generate trajectory of each vehicle
    grouped = roll_df.groupby('vehicle_id')
    vehicle2traj = dict()
    vehicle2time = dict()
    for key, group in tqdm.tqdm(grouped):
        out = gen_traj(group)
        if out is not None:
            traj, time = out
            vehicle2traj[key] = traj
            vehicle2time[key] = time
        else:
            pass
    # save data
    veh2ind = {}
    trajs = []
    times = []
    start_ind = 0
    for key in vehicle2traj.keys():
        veh2ind[str(key)] = {'start_ind':start_ind, 'len':len(vehicle2traj[key])}
        start_ind = start_ind + len(vehicle2traj[key])
        trajs.append(vehicle2traj[key])
        times.extend(vehicle2time[key])
    trajs = np.vstack(trajs)
    times = np.array(times)
    np.save(GEN_PATH + '/trajs.npy', trajs)
    np.save(GEN_PATH + '/times.npy', times)

    with open(GEN_PATH + '/veh2ind', 'w') as f:
        json.dump(veh2ind, f)
    print("successfully preprocess data")

def gen_real_data():
    data = np.load(GEN_PATH+'/trajs.npy')
    np.savetxt(GEN_PATH+'/real_ori.data', data, fmt="%d")
    print("real data generated")
    return data
    
def gen_fake_data(seq_len=48, nums=1000000):
    edge_df = pd.read_csv(GEN_PATH+'/edges.csv', index_col=0)
    eids = list(edge_df['eid'])
    result = []
    for i in range(nums):
        fake_data = random.sample(eids, seq_len)
        result.append(fake_data)
    result = np.array(result)
    np.savetxt(GEN_PATH+'/dispre_ori.data', result, fmt="%d")
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
    '''
    生成每个node id对应的gps
    '''
    '''
    edge_df = pd.read_csv(GEN_PATH+'/edges.csv', index_col=0)
    edge_df['mid_lon'] = (edge_df.iloc[:, 2] + edge_df.iloc[:, 4])/2
    edge_df['mid_lat'] = (edge_df.iloc[:, 3] + edge_df.iloc[:, 5])/2
    edge_df['mid_lon'].fillna(edge_df['mid_lon'].mean(), inplace=True)
    edge_df['mid_lat'].fillna(edge_df['mid_lat'].mean(), inplace=True)
    new_df = edge_df[['mid_lon', 'mid_lat']]
    np.savetxt(GEN_PATH+"/gps", new_df.values, fmt="%.3f")
    '''
    locs = []
    with open(GEN_PATH+'/num2hash.json', 'r') as f:
        num2hash = json.load(f)
        for i in range(len(num2hash)):
            hash_code = num2hash[str(i)]
            lat, lon = geohash.decode(hash_code)
            locs.append([lon, lat])
    np.savetxt(GEN_PATH+"/gps", np.array(locs), fmt="%.3f")

def gen_val_data():
    with open(GEN_PATH+"/veh2ind", 'r') as f:
        vec2ind = json.load(f) 
        f.close()
    vehicles = list(vec2ind.keys())
    vehicles.sort()
    pos = int(len(vehicles) * 0.8)
    ind = vec2ind[vehicles[pos]]['start_ind']
    real_data = read_file(GEN_PATH+'/real.data')
    val_data = real_data[ind:]
    train_data = real_data[:ind]
    np.savetxt(GEN_PATH+"/real.data", train_data, fmt="%d")
    np.savetxt(GEN_PATH+"/val.data", val_data, fmt="%d")
    
    
    
if __name__ == '__main__':
    
    preprocess("../data/geolife/final.csv")
    gen_real_data()
    gen_fake_data()
    real_data = read_file(GEN_PATH+'/real_ori.data')
    fake_data = read_file(GEN_PATH+'/dispre_ori.data')
    print("real data shape is ", len(real_data))
    print("fake data shape is ", len(fake_data))
    reindex_traj(real_data, GEN_PATH+'/real.data')
    reindex_traj(fake_data, GEN_PATH+'/dispre.data')
    
    # gen_node_gps()
    # gen_val_data()
    # gen_eid_dict(useHash=True)
    
    
    
    
    




