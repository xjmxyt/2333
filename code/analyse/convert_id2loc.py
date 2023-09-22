# -- coding: utf-8 --**
import sys
sys.path.append('../')
from utils import read_data_from_file
import pandas as pd
import json
import argparse

base_dir = '/home/xjm/MoveSim/data/traffic_hash1'
data_name = 'gene_LSTM.data'
out_name = 'gene_LSTM.json'
dic_path = base_dir+'/id2loc.json'

def id2loc(base_dir=base_dir, data_name=data_name, out_name=out_name, dic_path=dic_path, limit=2000):
    '''
    transfer node id to longitude and latitude
    '''
    gen_data = read_data_from_file(base_dir+'/'+data_name)
    print(gen_data.shape)
    # calculate step
    step = max(1, gen_data.shape[0] // limit)
    gen_data = gen_data[::step]
    print("gen shape:", gen_data.shape)
    
    with open(dic_path, 'r') as f:
        id2loc = json.load(f)
    output = []
    cnt = 0
    for line in gen_data:
        tmp = []
        for num in line:
            loc = id2loc[str(num)]
            long = loc['long']
            lat = loc['lat']
            # print(long)
            tmp.extend([long, lat])
        cnt = cnt + 1
        output.append(tmp)
    with open(base_dir+'/'+out_name, 'w') as f:
        s = json.dumps(output)
        f.write(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=base_dir, type=str)
    parser.add_argument('--data_name', default=data_name, type=str)
    parser.add_argument('--out_name', default=out_name, type=str)
    parser.add_argument('--dic_path', default=dic_path, type=str)
    parser.add_argument('--limit', default=2000, type=int)
    opt = parser.parse_args()
    
    id2loc(base_dir=opt.base_dir, data_name=opt.data_name, out_name=opt.out_name, dic_path=dic_path, limit=opt.limit)
    
        
    
    
    