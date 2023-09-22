import numpy as np
import argparse
from utils import read_data_from_file
from evaluations import IndividualEval
from traffic import read_file

DATA_PATH = '../data'


def markov(data="traffic", seq_len=48, task='markov'):
    GENE_DATA = DATA_PATH+'/%s/gene_%s.data' % (data, task)
    # calculate transition probabilty
    m1 = np.load('../data/%s/M1.npy' % data)
    print(m1.shape) # node * node, m1[i][j] represent node[i]->node[j] count
    m1 = m1.T
    eps = 0.000001
    m1 = m1 + eps # avoid div by 0
    prob = m1/np.sum(m1, axis=0)
    prob = prob.T 
    
    #load val data
    val_data = read_file('../data/%s/val.data' %data)
    val_data = np.array(val_data)
    gen_num = len(val_data)
    print("gen num: ", gen_num)
    # generate sequence using transition probability
    start_prob = np.load('../data/%s/start.npy' %data)
    print(start_prob.shape)
    locs = [i for i in range(len(m1))]
    
    start_point = np.random.choice(locs, size=gen_num, p=start_prob)
    
    result = []
    for s_node in start_point:
        x = [s_node]
        node = s_node
        for step in range(seq_len-1):
            p = list(prob[node].reshape(-1))
            node = np.random.choice(locs, size=1, p=p)
            x.append(node[0])
        result.append(x)
    with open(GENE_DATA, 'w') as fout:
        for sample in result:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)
    individualEval = IndividualEval(data=data)
    
    gene_data = np.array(result, dtype=np.int32)
    val_data = val_data.astype(np.int32)
    JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=val_data)

    with open( DATA_PATH+'/%s/logs/markov_jsd.log' % (data), 'a') as f:
        f.write(' '.join([str(j) for j in JSDs]))
        f.write('\n')
    
    print("Current JSD: %.8f, %.8f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))
    
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default="traffic_hash1", type=str)
    opt = parser.parse_args()
    markov(opt.data)
    
    
