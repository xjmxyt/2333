# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np
import json
from collections import deque, Counter
from models.deepMove import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong
from evaluations import IndividualEval
from pathlib import Path
import pandas as pd
import os
import argparse
import logging
import time

class RnnParameterData(object):
    def __init__(self, device, min_seq_len, num_locs, max_dist, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 datapath='', savepath='', data='foursquare'):
        self.device = device
        self.datapath = datapath
        self.savepath = savepath
        self.data = data

        path = f'{self.savepath}/data.json' # 预先根据ipynb文件处理得到的json文件
        data = json.load(open(path))

        # self.vid_list = recursive_transform(data['vid_list']) 
        self.uid_list = recursive_transform(data['uid_list'])
        self.data_neural = recursive_transform(data['data_neural'])

        self.min_seq_len = min_seq_len
        self.num_locs = num_locs
        self.max_dist = max_dist
        # self.num_locs = len(self.vid_list) # gps包含未被随机选择到的loc，因此实际loc id大于len(self.vid_list)
        self.uid_size = len(self.uid_list)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.history_mode = history_mode
        self.model_mode = model_mode

def run_simple(device, data, run_idx, lr, clip, model, optimizer, criterion, mode2=None): # train test diff
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""
    total_loss = []
    model.train(True)
    for u in data.keys(): # just one user
        for i in run_idx[u]:  # each path of each user
            trace = data[u][i]
            loc = trace['loc'].to(device)
            tim = trace['tim'].to(device)
            target = trace['target'].to(device)
        optimizer.zero_grad()
        if mode2 == 'attn_local_long':
            target_len = target.data.size()[0]
            scores = model(loc, tim, target_len)
        else:
            print('Model Type is wrong!')
        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        loss = criterion(scores, target)
        # preds.append(torch.argmax(scores, dim = 1).cpu().numpy())
        loss.backward()
        try: # gradient clipping
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            for p in model.parameters():
                if p.requires_grad:
                    p.data.add_(-lr, p.grad.data)
        except:
            pass
        optimizer.step()
        total_loss.append(loss.data.cpu().numpy())
        avg_loss = np.mean(total_loss, dtype=np.float64)
    return model, avg_loss

def run(args):
    device = torch.device("cuda:" + args.cuda)
    logging.basicConfig(filename=f'{args.savepath}/log.txt',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO)

    parameters = RnnParameterData(device = device, min_seq_len = args.min_seq_len, 
                                  num_locs = args.num_locs, max_dist = args.max_dist, 
                                  loc_emb_size=args.loc_emb_size, uid_emb_size=args.uid_emb_size,
                                  voc_emb_size=args.voc_emb_size, tim_emb_size=args.tim_emb_size,
                                  hidden_size=args.hidden_size, dropout_p=args.dropout_p,
                                  data=args.data, lr=args.learning_rate,
                                  lr_step=args.lr_step, lr_decay=args.lr_decay, L2=args.L2, 
                                  rnn_type=args.rnn_type, optim=args.optim, attn_type=args.attn_type,
                                  clip=args.clip, epoch_max=args.epoch_max, history_mode=args.history_mode,
                                  model_mode=args.model_mode, datapath=args.datapath, savepath=args.savepath)
    argv = {'loc_emb_size': args.loc_emb_size, 'uid_emb_size': args.uid_emb_size, 'voc_emb_size': args.voc_emb_size,
            'tim_emb_size': args.tim_emb_size, 'hidden_size': args.hidden_size,
            'dropout_p': args.dropout_p, 'data': args.data, 'learning_rate': args.learning_rate,
            'lr_step': args.lr_step, 'lr_decay': args.lr_decay, 'L2': args.L2, 'act_type': 'selu',
            'optim': args.optim, 'attn_type': args.attn_type, 'clip': args.clip, 'rnn_type': args.rnn_type,
            'epoch_max': args.epoch_max, 'history_mode': args.history_mode, 'model_mode': args.model_mode}
    print('*' * 15 + 'start training' + '*' * 15)
    logging.info('model_mode:{} history_mode:{} users:{}'.format(parameters.model_mode, parameters.history_mode, parameters.uid_size))

    if parameters.model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple(parameters=parameters).to(device)
    elif parameters.model_mode == 'attn_avg_long_user': 
        model = TrajPreAttnAvgLongUser(parameters=parameters).to(device)
    elif parameters.model_mode == 'attn_local_long':
        model = TrajPreLocalAttnLong(parameters=parameters).to(device) # default
    if args.pretrain == 1:
        model.load_state_dict(torch.load("../pretrain/" + args.model_mode + "/res.m"))

    if 'max' in parameters.model_mode:
        parameters.history_mode = 'max'
    elif 'avg' in parameters.model_mode:
        parameters.history_mode = 'avg'
    else:
        parameters.history_mode = 'whole'
 
    criterion = nn.NLLLoss().to(device) # classify
    # criterion = torch.nn.MSELoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=parameters.lr_step,
                                                     factor=parameters.lr_decay, threshold=1e-3)
    lr = parameters.lr
    metrics = {'train_loss': [], 'valid_loss': [], 'accuracy': [], 'valid_acc': {}}
    candidate = parameters.data_neural.keys()  # get uid from raw data

    df = pd.read_csv(f'{args.datapath}/{args.data}-hour-pid.csv')
    MAX_POI = df['poi_id'].max()
    EOS = MAX_POI + 1

    if 'long' in parameters.model_mode:
        long_history = True
    else:
        long_history = False
    # load data_train, train_idx

    individualEval = IndividualEval(args)
    for epoch in range(parameters.epoch):
        st = time.time()
        # if args.pretrain == 0:
        model, avg_loss = run_simple(device, data_train, train_idx, lr, parameters.clip, 
                                    model, optimizer, criterion, parameters.model_mode)
        if epoch % 5 == 0:
            logging.info('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
            print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
        metrics['train_loss'].append(avg_loss)
 
        checkpath = f'{args.savepath}/checkpoint'
        Path(checkpath).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f'{checkpath}/ep_{epoch}.m')

        scheduler.step(avg_loss) 
        lr_last = lr
        lr = optimizer.param_groups[0]['lr']
        if lr_last > lr:
            load_epoch = np.argmin(metrics['train_loss'])
            model.load_state_dict(torch.load(f'{checkpath}/ep_{load_epoch}.m'))
            logging.info('load epoch={} model state'.format(load_epoch))
        if epoch == 0:
            logging.info('single epoch time cost:{}'.format(time.time() - st))
        if lr <= 0.9 * 1e-5:
            break
        if args.pretrain == 1:
            break

    # select test first poi from train data
    first_pois  = []
    model.train(False)
    for u in data_train.keys(): # just one user
        for i in train_idx[u]:  # each path of each user
            trace = data_train[u][i]
            first_pois.append(trace['loc'][0].item())
    uni, cnt = np.unique(first_pois, return_counts=True)

    test_data, pred_path = [], []
    for u in data_test.keys(): # just one user
        for i in test_idx[u]:  # each path of each user
            trace = data_test[u][i]
            test_data.append(trace['target'].tolist())
            path = []
            poi = torch.tensor(np.random.choice(uni, 1, replace=True, p=cnt/cnt.sum())).to(device)
            path.append(poi.item()) 
            pred_p = model.inference(poi, EOS)  # poi: [id]
            path.extend(pred_p)
            pred_path.append(path)
    
    # compute jsd
    individualEval = IndividualEval(args)
    JSDs = individualEval.get_individual_jsds(t1=test_data, t2=pred_path)
    logging.info(f"{args.data} Test JSD: %f, %f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))
    print(f"{args.data} Test JSD: %f, %f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))
    with open(args.savepath + '/jsd.log', 'a') as f: # 重写W，追加a
        f.write(' '.join([str(j) for j in JSDs]))
        f.write('\n')

    with open(f'{args.savepath}/gene.txt', 'w') as f:
        for path in pred_path:
            f.write(' '.join([str(poi) for poi in path]))
            f.write('\n')        
        
if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # for debug

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',  default="2", type=str)
    parser.add_argument('--min_seq_len', default='8', type=int) 
    parser.add_argument('--num_locs', default='2862', type=int) 
    parser.add_argument('--max_dist', default='0.4353543766624795', type=float)
    parser.add_argument('--method', default='DeepMove', type=str)
    parser.add_argument('--loc_emb_size', type=int, default=500, help="location embeddings size")
    parser.add_argument('--uid_emb_size', type=int, default=40, help="user id embeddings size")
    parser.add_argument('--voc_emb_size', type=int, default=50, help="words embeddings size")
    parser.add_argument('--tim_emb_size', type=int, default=10, help="time embeddings size")
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--data', type=str, default='TKY', choices=['geolife', 'gowalla', 'brightkite', 'NYC', 'TKY'])
    parser.add_argument('--learning_rate', type=float, default=5 * 1e-4)
    parser.add_argument('--lr_step', type=int, default=2)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--L2', type=float, default=1 * 1e-5, help=" weight decay (L2 penalty)")
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--epoch_max', type=int, default=20)
    parser.add_argument('--history_mode', type=str, default='avg', choices=['max', 'avg', 'whole'])
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--attn_type', type=str, default='dot', choices=['general', 'concat', 'dot'])
    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--savepath', type=str, default='')
    parser.add_argument('--model_mode', type=str, default='attn_local_long',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--pretrain', type=int, default=0)
    args = parser.parse_args()
    args.datapath = f'/home/wangyu/nfs/00-covid/GeneTraj/gen-data/min_len_{args.min_seq_len}/{args.data}'
    args.savepath = f'/home/wangyu/nfs/00-covid/GeneTraj/{args.method}/min_len_{args.min_seq_len}/{args.data}'
    Path(args.savepath).mkdir(parents=True, exist_ok=True)

    print(args)
    run(args)

