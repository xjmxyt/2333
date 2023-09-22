# coding=utf-8
import pdb
import torch
import random
import argparse
import setproctitle
from torch import nn, optim

from train import *
from utils import *
from rollout import Rollout
from evaluations import IndividualEval
from gen_data import *
from models.generator import *
from models.discriminator import Discriminator
from models.gan_loss import GANLoss, distance_loss, period_loss
from data_iter import GenDataIter, NewGenIter, DisDataIter
from torch.utils.tensorboard import SummaryWriter





def main(opt):
    # all parameters
    # assigned in argparse
    print(opt)       
    
    # fixed parameters
    SEED = 88
    EPOCHS = 50
    BATCH_SIZE = 32
    SEQ_LEN = 48
    GENERATED_NUM = 10000
    
    DATA_PATH = '../data'
    REAL_DATA = DATA_PATH+'/%s/real.data' % opt.data
    VAL_DATA = DATA_PATH+'/%s/val.data' % opt.data
    TEST_DATA = DATA_PATH+'/%s/test.data' % opt.data
    GENE_DATA = DATA_PATH+'/%s/gene_%s.data' % (opt.data, opt.task)
    TB_LOG_PATH = DATA_PATH+'/%s/logs/log_%s' % (opt.data, opt.task) # tensorboard log path
    
    D_MODEL_FILE = DATA_PATH+'/%s/model/%s_discriminator.pth'%  (opt.data, opt.task)
    G_MODEL_FILE = DATA_PATH+'/%s/model/%s_generator.pth'%  (opt.data, opt.task)

    random.seed(SEED)
    np.random.seed(SEED)
    
    data2locnum = {
        'mobile': 8606, 
        'traffic': 46165,
        'traffic_hash1': 5288
    }
    
    # setting tensorboard
    writer = SummaryWriter(TB_LOG_PATH)
    
    if opt.data == 'mobile':
        TOTAL_LOCS = data2locnum[opt.data]
        individualEval = IndividualEval(data='mobile')
    elif opt.data == 'traffic':
        TOTAL_LOCS = data2locnum[opt.data]
        individualEval = IndividualEval(data="traffic")
    elif opt.data == 'traffic_hash1':
        TOTAL_LOCS = data2locnum[opt.data]
        individualEval = IndividualEval(data="traffic_hash1")        
    else:
        TOTAL_LOCS = 23768
        individualEval = IndividualEval(data='geolife')
    
    device = torch.device("cuda:"+opt.cuda)
    # device = 'cpu'

    if opt.preprocess:
        print('Pre-processing Data {}...'.format(opt.data))
        isHash = False
        if ("hash" in opt.data):
            isHash = True
            print("Enable Hash Mode")
        gen_matrix(opt.data, isHash=isHash)
        if opt.data[:7] == "traffic":
            gen_start(opt.data)
    else:
        print("preprocess ignored")

    # assigned according to task
    if opt.task == 'attention':
        d_pre_epoch = 10
        g_pre_epoch = 10
        ploss_alpha = float(opt.ploss)
        dloss_alpha = float(opt.dloss)
        generator = ATGenerator(device=device,total_locations=TOTAL_LOCS,starting_sample='real',
                                starting_dist=np.load(f'{DATA_PATH}/{opt.data}/start.npy'),data=opt.data)
        discriminator = Discriminator(total_locations=TOTAL_LOCS)
        gen_train_fixstart = True
    elif opt.task == 'base':
        d_pre_epoch = 10
        g_pre_epoch = 10
        ploss_alpha = float(opt.ploss)
        dloss_alpha = float(opt.dloss)
        generator = Generator(device=device, total_locations=TOTAL_LOCS, starting_sample='real', 
                              starting_dist=np.load(f'{DATA_PATH}/{opt.data}/start.npy'))
        discriminator = Discriminator(total_locations=TOTAL_LOCS)
        gen_train_fixstart = True
        
    generator = generator.to(device)
    discriminator = discriminator.to(device)   
    print(next(discriminator.parameters()).device)   

    # prepare files and datas
    logger = get_workspace_logger(opt.data)
    
    if opt.pretrain:
        # generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM,
                         # DATA_PATH+'/%s/gene_epoch_init.data' % opt.data)
        # pretrain discriminator
        logger.info('pretrain discriminator ...')
        pretrain_real =  DATA_PATH+'/%s/real.data' %  opt.data
        pretrain_fake =  DATA_PATH+'/%s/dispre.data' %  opt.data
        dis_data_iter = DisDataIter(pretrain_real, pretrain_fake, BATCH_SIZE, SEQ_LEN)
        
        dis_criterion = nn.NLLLoss(reduction='sum')
        dis_criterion = dis_criterion.to(device)
        
        dis_optimizer = optim.Adam(discriminator.parameters(),lr=0.000001)
        pretrain_model("D", d_pre_epoch, discriminator, dis_data_iter,
                       dis_criterion, dis_optimizer, BATCH_SIZE, device=device)
        torch.save(discriminator.state_dict(), DATA_PATH+'/%s/pretrain/%s_discriminator.pth'%  (opt.data, opt.task))
        # pretrain generator
        logger.info('pretrain generator ...')
        if gen_train_fixstart:
            logger.info('Using NewGenIter  ...')
            gen_data_iter = NewGenIter(REAL_DATA, BATCH_SIZE)
        else:
            logger.info('Using GenDataIter  ...')
            gen_data_iter = GenDataIter(REAL_DATA, BATCH_SIZE)
        gen_criterion = nn.NLLLoss(reduction='sum')
        gen_optimizer = optim.Adam(generator.parameters(),lr=0.0001)
        gen_criterion = gen_criterion.to(device)
        pretrain_model("G", g_pre_epoch, generator, gen_data_iter,
                       gen_criterion, gen_optimizer, BATCH_SIZE, device=device)
        torch.save(generator.state_dict(), DATA_PATH+'/%s/pretrain/%s_generator.pth'%  (opt.data, opt.task))
        
    else:
        print("start to load model ...")
        generator.load_state_dict(torch.load(DATA_PATH+'/%s/pretrain/%s_generator.pth' %  (opt.data, opt.task), map_location=device))    
        print("generator params loaded ...")
        discriminator.load_state_dict(torch.load( DATA_PATH+'/%s/pretrain/%s_discriminator.pth' % (opt.data, opt.task), map_location=device))
        print('discriminator params loaded ...')
        
    if opt.load_prev:
        print("Load previous model")
        generator.load_state_dict(torch.load(G_MODEL_FILE, map_location=device)) 
        discriminator.load_state_dict(torch.load(D_MODEL_FILE, map_location=device)) 
    print('advtrain generator and discriminator ...')
    rollout = Rollout(generator, 0.8)

    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(),lr=0.0001)

    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters(),lr=0.00001)
    
    gen_gan_loss = gen_gan_loss.to(device)
    dis_criterion = dis_criterion.to(device)
    
    print("start generate data ...")
    generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, GENE_DATA)
    generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM,
                          DATA_PATH+'/%s/gene_epoch_%d_%s.data' % (opt.data,0, opt.task))
    print("generate data finished ...")

    for epoch in range(EPOCHS):
        gene_data = read_data_from_file(GENE_DATA)
        val_data = read_data_from_file(VAL_DATA)

        JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=val_data)

        with open( DATA_PATH+'/%s/logs/jsd_%s.log' % (opt.data, opt.task), 'a') as f:
            f.write(' '.join([str(j) for j in JSDs]))
            f.write('\n')
        writer.add_scalars('eval_distance', {
            "Distance": JSDs[0],
            "Radius": JSDs[1],
            "Duration": JSDs[2],
            "DailyLoc": JSDs[3],
            "G-rank": JSDs[4],
            "I-rank": JSDs[5]
            }, epoch)        
        print("Current JSD: %.8f, %.8f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))

        # Train the generator for one step
        for it in range(1):
            samples = generator.sample(BATCH_SIZE, SEQ_LEN)
            # construct the input to the genrator, add zeros before samples and
            # delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            zeros = zeros.to(device)
            inputs = torch.cat([zeros, samples.data], dim=1)[
                              :, :-1].contiguous()
            tim = torch.LongTensor([i%24 for i in range(48)]).to(device)
            tim = tim.repeat(BATCH_SIZE).reshape(BATCH_SIZE, -1)
            
            targets = samples.contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, discriminator)
            rewards = torch.Tensor(rewards)
            rewards = torch.exp(rewards.to(device)).contiguous().view((-1,))
            prob = generator.forward(inputs, tim)
            
            try:
                gloss = gen_gan_loss(prob, targets, rewards, device)
            except:
                gloss = gen_gan_loss(prob, targets, rewards, device)

            if ploss_alpha != 0.:
                p_crit = period_loss(24)
                p_crit = p_crit.to(device)
                pl = p_crit(samples.float())
                gloss += ploss_alpha * pl
            if dloss_alpha != 0.:
                d_crit = distance_loss(device=device,datasets=opt.data)
                d_crit = d_crit.to(device)
                dl = d_crit(samples.float())
                gloss += dloss_alpha * dl
            gen_gan_optm.zero_grad()
            gloss.backward()
            gen_gan_optm.step()
        
        rollout.update_params()
        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, SEQ_LEN,
                             GENERATED_NUM, GENE_DATA)
            dis_data_iter = DisDataIter(REAL_DATA, GENE_DATA, BATCH_SIZE, SEQ_LEN)
            for _ in range(2):
                dloss = train_epoch(
                    "D", discriminator, dis_data_iter, dis_criterion, dis_optimizer, BATCH_SIZE, device=device)

        logger.info('Epoch [%d] Generator Loss: %f, Discriminator Loss: %f' %
                    (epoch, gloss.item(), dloss))
        writer.add_scalar('GLoss', gloss.item(), epoch)
        writer.add_scalar('DLoss', dloss, epoch)
        with open( DATA_PATH+'/%s/logs/loss_%s.log' % (opt.data, opt.task), 'a') as f:
            f.write(' '.join([str(j)
                              for j in [epoch, float(gloss.item()), dloss]]))
            f.write('\n')
        if (epoch + 1) % 20 == 0:
            generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM,  DATA_PATH+'/%s/gene_epoch_%d.data' %
                             (opt.data, epoch + 1))
        torch.save(generator.state_dict(), G_MODEL_FILE)
        torch.save(discriminator.state_dict(), D_MODEL_FILE)
    writer.close()
    '''
    test_data = read_data_from_file(TEST_DATA)
    gene_data = read_data_from_file(GENE_DATA)
    JSDs = individualEval.get_individual_jsds(t1=gene_data, t2=test_data)
    print("Test JSD: %.8f, %.8f, %f, %f, %f, %f" % (JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5]))

    torch.save(generator.state_dict(), G_MODEL_FILE)
    torch.save(discriminator.state_dict(), D_MODEL_FILE)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain',action='store_true')
    parser.add_argument('--cuda',  default="0", type=str)
    parser.add_argument('--task', default='attention', type=str)    
    parser.add_argument('--ploss', default='3.0', type=float)
    parser.add_argument('--dloss', default='1.5', type=float)
    parser.add_argument('--data', default='geolife', type=str)
    parser.add_argument('--length', default=48, type=int)
    parser.add_argument('--preprocess', action="store_true")
    parser.add_argument('--load_prev',action='store_true')

    opt = parser.parse_args()
    main(opt)
