# encoding: utf-8
import torch
import numpy as np


def generate_samples(model, batch_size, seq_len, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(
            batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def generate_samples_to_mem(model, batch_size, seq_len, generated_num):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(
            batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)


def pretrain_model(
        name,
        pre_epochs,
        model,
        data_iter,
        criterion,
        optimizer,
        batch_size,
        device=None):
    lloss = 0.
    criterion = criterion.to(device)
    for epoch in range(pre_epochs):
        loss = train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device)
        print('Epoch [%d], loss: %f' % (epoch + 1, loss))
        if loss < 0.01 or 0 < lloss - loss < 0.01:
            print("early stop at epoch %d" % (epoch + 1))
            break
        

def generate_time(batch_size, step, random_t):
    tim = step * 5
    t = torch.LongTensor([tim])
    t = t.repeat(batch_size).reshape(batch_size, -1)
    # 将随机张量与 t 相加
    t_with_random = t + random_t
    # 使用 torch.divmod 进行计算
    hour = t_with_random.div(60).floor()
    minute = t_with_random%60
    # 将 hour 和 minute 拼接为一个形状为 (n, 2) 的张量
    t = torch.stack((hour, minute), dim=2)  
    print(t.shape)
    return t 

def train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device=None):
    total_loss = 0.
    if name == "G":
        random_t = torch.randint(0, 1000, (batch_size, 1))
        tim = torch.cat([generate_time(batch_size, i, random_t) for i in range(47)], axis=1).to(device)
        print(tim.shape)
    # print("data iter size: ",len(data_iter))
    for i, (data, target) in enumerate(data_iter):
        data = torch.LongTensor(data).to(device)
        target = torch.LongTensor(target).to(device)
        target = target.contiguous().view(-1)
        if name == "G":
            print(data.shape, tim.shape)
            pred = model(data, tim)
        else:
            pred = model(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return total_loss / (i + 1)

def train_epoch_deepMove(model, data_iter, criterion, optimizer, device=None):
    total_loss = []
    model.train(True)
    for i, (tim, data, target) in enumerate(data_iter):
        tim = torch.LongTensor(tim).to(device)
        data = torch.LongTensor(data).to(device)
        target = torch.LongTensor(target).to(device)
        
        optimizer.zero_grad()
        pred = model(data, tim)
        loss = criterion(pred, target)  
        
        loss.backward()
        # TODO: gradiemt clipping
        # try: # gradient clipping
        #     torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        #     for p in model.parameters():
        #         if p.requires_grad:
        #             p.data.add_(-lr, p.grad.data)
        # except:
        #     pass
        optimizer.step()
        total_loss.append(loss.data.cpu().numpy())
    avg_loss = np.mean(total_loss, dtype=np.float64)
    data_iter.reset()
    return avg_loss 
            
        