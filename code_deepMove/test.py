import torch 
import numpy as np
from models.generator import ATGenerator

DATA_PATH = '../data'
print("start to load model ...")
device = 'cuda:1'
TOTAL_LOCS = 46165

generator = ATGenerator(device=device,total_locations=TOTAL_LOCS,starting_sample='real',
                        starting_dist=np.load(f'{DATA_PATH}/traffic/start.npy'),data="traffic")
#generator.to(device)
#torch.save(generator.state_dict(), "test.pth")
#print("save succeed")

# checkpoint = torch.load(DATA_PATH+'/%s/pretrain/%s_generator.pth' %  ("traffic", "attention"))
checkpoint = torch.load(DATA_PATH+'/%s/pretrain/%s_generator.pth' %  ("traffic", "attention"))
print("loaded")
generator.load_state_dict(torch.load(DATA_PATH+'/%s/pretrain/%s_generator.pth' %  ("traffic", "attention"), map_location=device))
print("generator params loaded ...")
