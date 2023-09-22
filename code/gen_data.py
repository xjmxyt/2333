import pandas as pd
from evaluations import *
from utils import *
from traffic import get_eid_dic

def distance(gps1,gps2):
    x1,y1 = gps1
    x2,y2 = gps2
    return np.sqrt((x1-x2)**2+(y1-y2)**2 )

    
def gen_start(data='traffic'):
    file_name = '../data/%s/M1.npy'%data 
    m = np.load(file_name)
    data_cnt = np.sum(m, axis=1)
    start = data_cnt / np.sum(data_cnt)
    print("start shape is ", start.shape)
    np.save('../data/%s/start.npy'%data, start)

def gen_matrix(data='geolife', isHash=False):
    if data[:7] == 'traffic':
        id2eid = get_eid_dic('../data/%s/id2eid.json'%data)
        train_data = read_data_from_file('../data/%s/real.data'%data)
        train_data = np.array(train_data)
        edge_df = pd.read_csv('../data/%s/edges.csv'%data, index_col=0)
        max_locs = len(id2eid)
        print(max_locs)
        reg1 = np.zeros([max_locs,max_locs])
        for i in range(0, len(train_data)):
            line = train_data[i]
            for j in range(len(line)-1):
                reg1[line[j], line[j+1]] +=1
        print("M1 generated")
        np.save('../data/%s/M1.npy'%data,reg1)
        
        reg2 = np.zeros([max_locs,max_locs])
        if isHash:
            print("Using Hash Mode: use longitude col: ", edge_df.columns[-1])
            num2hash = get_eid_dic('../data/%s/num2hash.json'%data)
            for i in range(max_locs):
                for j in range(i+1, max_locs):
                    reg2[i][j] = reg2[j][i] = get_geohash_distance(num2hash[i], num2hash[j])
            print("M2 generated") 
            print(reg2.shape)
            np.save('../data/%s/M2.npy'%data,reg2)  
        else:            
            print("No Hash Mode: use longitude col: ", edge_df.columns[2], edge_df.columns[4])
            edge_df['mid_lon'] = (edge_df.iloc[:, 2] + edge_df.iloc[:, 4])/2
            edge_df['mid_lat'] = (edge_df.iloc[:, 3] + edge_df.iloc[:, 5])/2
            edge_df['mid_lon'].fillna(edge_df['mid_lon'].mean(), inplace=True)
            edge_df['mid_lat'].fillna(edge_df['mid_lat'].mean(), inplace=True)
            m1 = edge_df[['mid_lon', 'mid_lat']].values
            m2 = edge_df[['mid_lon', 'mid_lat']].values
            m1 = m1[:, np.newaxis, ...]
            m2 = m2[np.newaxis, ...]
            print(m1.shape, m2.shape)
            m3 = m1-m2
            print(m3[0][0])
            reg2 = np.linalg.norm(m3, axis=-1)
            print(reg2.shape)
            print("M2 generated") 
            
            np.save('../data/%s/M2.npy'%data,reg2)     
    else:
        train_data = read_data_from_file('../data/%s/real.data'%data)
        gps = get_gps('../data/%s/gps'%data)
        print(np.array(gps).shape)
        if data=='mobile':
            max_locs = 8606
        else:
            max_locs = 23768

        reg1 = np.zeros([max_locs,max_locs])
        for i in range(len(train_data)):
            line = train_data[i]
            for j in range(len(line)-1):
                reg1[line[j],line[j+1]] +=1
        reg2 = np.zeros([max_locs,max_locs])
        for i in range(max_locs):
            for j in range(max_locs):
                if i!=j:
                    reg2[i,j] = distance((gps[0][i],gps[1][i]),(gps[0][j],gps[1][j]))
    

        np.save('../data/%s/M11.npy'%data,reg1)
        np.save('../data/%s/M22.npy'%data,reg2)

    print('Matrix Generation Finished')

    

    
if __name__ == '__main__':
    gen_matrix(data='traffic_hash1', isHash=True)
    #gen_start()



    
