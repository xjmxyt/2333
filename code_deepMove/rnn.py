# coding=utf-8
import pdb
import torch
import random
import argparse
import setproctitle
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter


from train import *
from utils import *
from rollout import Rollout
from evaluations import IndividualEval
from gen_data import *
from models.rnn import RNN
from models.gan_loss import GANLoss, distance_loss, period_loss
from data_iter import GenDataIter, NewGenIter, DisDataIter




def main(opt):
    # all parameters
    # assigned in argparse
    print(opt)       
    
    # fixed parameters
    SEED = 88
    EPOCHS = 200
    BATCH_SIZE = 32
    SEQ_LEN = 48
    GENERATED_NUM = 10000
    
    DATA_PATH = '../data'
    REAL_DATA = DATA_PATH+'/%s/real.data' % opt.data
    VAL_DATA = DATA_PATH+'/%s/val.data' % opt.data
    TEST_DATA = DATA_PATH+'/%s/test.data' % opt.data
    GENE_DATA = DATA_PATH+'/%s/gene_%s.data' % (opt.data, opt.task)
    MODEL_PATH = DATA_PATH+'/%s/model_%s.pth'%(opt.data, opt.task)
    TB_LOG_PATH = DATA_PATH+'/%s/logs/log_%s'%(opt.data, opt.task) # tensorboard log path

    random.seed(SEED)
    np.random.seed(SEED)
    
    writer = SummaryWriter(TB_LOG_PATH)

    
    if opt.data == 'mobile':
        TOTAL_LOCS = 8606
        individualEval = IndividualEval(data='mobile')
    elif opt.data == 'traffic':
        TOTAL_LOCS = 46165
        individualEval = IndividualEval(data="traffic")
    elif opt.data == 'traffic_hash1':
        TOTAL_LOCS = 5288
        individualEval = IndividualEval(data="traffic_hash1")            
    else:
        TOTAL_LOCS = 23768
        individualEval = IndividualEval(data='geolife')
    
    device = torch.device("cuda:"+opt.cuda)
    # device = 'cpu'
    logger = get_workspace_logger(opt.data)

    if opt.preprocess:
        print('Pre-processing Data...')
        gen_matrix(opt.data)
    else:
        print("preprocess ignored")

    if opt.task == "LSTM" or opt.task == "GRU":
        model = RNN(rnn_type=opt.task, 
                    device=device, 
                    total_locations=TOTAL_LOCS, starting_sample="real",
                    starting_dist=np.load(f'{DATA_PATH}/{opt.data}/start.npy')
        )
    gen_train_fixstart = True
    
    print("device is: ", device)
    
    if opt.load_prev:
        print("Load previous model")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) 
    
    model.to(device)  
    
    # model settings
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.NLLLoss(reduction='sum')
    if gen_train_fixstart:
        data_iter = NewGenIter(REAL_DATA, BATCH_SIZE)
    else:
        data_iter = GenDataIter(REAL_DATA, BATCH_SIZE)
          
    
    generate_samples(model, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, GENE_DATA)

    for epoch in range(EPOCHS):
        if epoch % 10 == 0:
            generate_samples(model, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, GENE_DATA)
            gene_data = read_data_from_file(GENE_DATA)
            val_data = read_data_from_file(VAL_DATA)

            JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=val_data)

            with open( DATA_PATH+'/%s/logs/jsd_%s.log' % (opt.data, opt.task), 'a') as f:
                f.write(' '.join([str(j) for j in JSDs]))
                f.write('\n')
            
            print("Current JSD: %f, %f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))
            writer.add_scalars('eval_distance', {
                "Distance": JSDs[0],
                "Radius": JSDs[1],
                "Duration": JSDs[2],
                "DailyLoc": JSDs[3],
                "G-rank": JSDs[4],
                "I-rank": JSDs[5]
                }, epoch)
        
        loss = train_epoch("default", model=model, data_iter=data_iter, criterion=criterion, optimizer=optimizer,
                    batch_size=BATCH_SIZE, device=device)
        
            

        logger.info('Epoch [%d] Loss: %f' %(epoch, loss))
        writer.add_scalar('loss', float(loss), epoch)
        with open( DATA_PATH+'/%s/logs/loss_%s.log' % (opt.data, opt.task), 'a') as f:
            f.write(' '.join([str(j)
                              for j in [epoch, float(loss)]]))
            f.write('\n')
        torch.save(model.state_dict(), MODEL_PATH)
    '''
    test_data = read_data_from_file(TEST_DATA)
    gene_data = read_data_from_file(GENE_DATA)
    JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=val_data)
    print("Test JSD: %f, %f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))
    '''
    torch.save(model.state_dict(), MODEL_PATH)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_prev',action='store_true')
    parser.add_argument('--cuda',  default="0", type=str)
    parser.add_argument('--task', default='LSTM', type=str)    
    parser.add_argument('--data', default='traffic', type=str)
    parser.add_argument('--length', default=48, type=int)
    parser.add_argument('--preprocess', action="store_true")

    opt = parser.parse_args()
    main(opt)
