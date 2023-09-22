import pandas as pd
import random
import json
from collections import deque
import time

def calculate_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 的执行时间为: {execution_time} 秒")
        return result
    return wrapper

class NumHandler:
    num_to_hash = {}
    
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.num_to_hash = json.load(file)
        
    def get_hash(self, loc: int) -> str:
        if str(loc) in self.num_to_hash.keys():
            return self.num_to_hash[str(loc)]
        return ""

class GeoHash2RoadHandler:
    hash_to_index = {}
    
    def __init__(self, file_path):
        self.gen_hash_dict(file_path=file_path)

    def gen_hash_dict(self, file_path):
        # 生成hash编码到具体路径的映射函数
        # 读取表格数据
        data = pd.read_csv(file_path)

        # 建立hash到index的映射
        for index, row in data.iterrows():
            hash_value = row['hash']
            if hash_value not in self.hash_to_index:
                self.hash_to_index[hash_value] = []
            self.hash_to_index[hash_value].append(index)
        

    def get_random_index(self, input_hash):
        # 输入一个hash，随机选择一个index
        index_list = self.hash_to_index.get(input_hash, [])
        random_index = random.choice(index_list) if index_list else None
        return random_index

class PathFinder:
    
    adjacency_dict = {}
    
    def __init__(self, file_path):
        self.read_adjacency_dict(file_path)
        # print(len(self.adjacency_dict))
        # for item in self.adjacency_dict:
        #     print(item, self.adjacency_dict[item])
        #     break
        
    @calculate_execution_time
    def fill_seq(self, seq):
        path = [seq[0]]
        for i in range(1, len(seq)):
            path_temp = self.find_path(seq[i-1], seq[i])
            path.extend(path_temp[1:])
        return path
        
    @calculate_execution_time
    def find_path(self, start_node, end_node):
        visited = set()
        queue = deque([(start_node, [])])

        while queue:
            node, path = queue.popleft()
            # print(node)
            visited.add(node)
            path.append(node)

            if node == end_node:
                return path

            if str(node) in self.adjacency_dict:
                neighbors = self.get_neighbor(node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append((neighbor, path[:]))

        return []

    # 读取节点邻接关系记录的JSON文件
    def read_adjacency_dict(self, file_path):
        with open(file_path, 'r') as file:
            self.adjacency_dict = json.load(file)
    
    def get_neighbor(self, node):
        neighbors = self.adjacency_dict[str(node)]
        # print(node, '->', neighbors)
        return neighbors
    

# 调用函数
# input_hash = 'wtw3se'
# output_hash = 'wtw4tu'
num2hash_file_path = '/home/xjm/MoveSim/data/traffic_hash1/num2hash.json'
num_handler = NumHandler(num2hash_file_path)
input_hash = num_handler.get_hash(362)
output_hash = num_handler.get_hash(1842)
print(input_hash, output_hash)

edge_file_path = '/home/xjm/MoveSim/data/traffic_hash1/edges.csv'
handler = GeoHash2RoadHandler(edge_file_path)
input_index = handler.get_random_index(input_hash)
output_index = handler.get_random_index(output_hash)
print("随机选择的index:", input_index, output_index)


input_index, output_index = 29164, 28912
near_edge_file_path = '/home/xjm/MoveSim/data/traffic_hash1/near_edge.json'
path_finder = PathFinder(near_edge_file_path)
path = path_finder.find_path(input_index, output_index)
print(path)

print(path_finder.fill_seq([28577, 28576, 28576, 28917, 28576, 28576, 28576, 28576, 46592, 28915, 29164, 28912, 35891, 35897, 35885, 35885, 38719, 57742, 31064, 28674, 57646, 38155, 30542, 38160, 35707, 38161, 46095, 35695, 46794, 35759, 35759, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076, 31076]))
