import sys
sys.path.append('../')
from traffic import read_file
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# base_dir = '/home/xjm/MoveSim/data/geolife'
base_dir = '/home/xjm/MoveSim/data/traffic_hash1'
eps = 1e-6
pic_num = 3
# 计算熵
def calc_ent(x):
    """
        calculate shanno ent of x
    """
    total = np.sum(x)
    if total == 0:
        return -1
    p = x/total
    return np.sum(-p * np.log(p + eps))


# 统计每个点的出现次数
def cal_freq(file_name):
    real_data = read_file(file_name)
    real_data = np.array(real_data)
    counter = Counter(real_data.reshape(-1))
    freq = list(dict(counter).values())
    freq.sort(reverse=True)
    plt.plot(range(len(freq)), freq)
    plt.savefig("freq_%s.png"%pic_num, dpi=200)


# 统计从每点转移出去的分布情况    
def cal_distribute(file_name=base_dir+'/M1.npy'):
    m = np.load(file_name)
    result = []
    for i in range(len(m)):
        ent = calc_ent(m[i])
        result.append(ent)
    result.sort(reverse=True)
    plt.plot(range(len(result)), result)
    plt.savefig("entropy_%s.png"%pic_num, dpi=200)    

if __name__ == '__main__':
    file_name = base_dir+'/real.data'
    cal_freq(file_name)
    # cal_distribute()
    # print(calc_ent(np.array([1,2,3])))