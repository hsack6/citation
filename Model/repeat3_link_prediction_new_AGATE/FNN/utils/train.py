from torch.autograd import Variable
import torch
import numpy as np
import random

def _random_subset(seq,m):
    targets=set()
    while len(targets)<m:
        x=random.choice(seq)
        targets.add(x)
    return targets

def balancer(target, mask):
    target = target[0 < mask]
    n_positive = int(target.sum())
    n_negative = int(len(target) - n_positive)
    if n_positive <= n_negative:
        sample_idx_list = np.where(target==1)[0].tolist()
        negative_idx = np.where(target==0)[0]
        sample_idx_list.extend(list(_random_subset(negative_idx, n_positive)))
    else:
        sample_idx_list = np.where(target==0)[0].tolist()
        positive_idx = np.where(target==1)[0]
        sample_idx_list.extend(list(_random_subset(positive_idx, n_negative)))
    return sample_idx_list

def train(epoch, dataloader, net, criterion, optimizer, opt):
    train_loss = 0
    net.train()
    for i, (sample_idx, annotation, label, mask) in enumerate(dataloader, 0):
        net.zero_grad()
        padding = torch.zeros(opt.batchSize, opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)

        if opt.cuda:
            init_input      = init_input.cuda()
            label = label.cuda()
            mask = mask.cuda()

        init_input      = Variable(init_input)
        target = Variable(label)
        mask = Variable(mask)

        output = net(init_input)
        sample_idx_list = balancer(target.numpy(), mask.numpy())
        if len(sample_idx_list) == 0: # posあるいはnegしかない状況
            loss = criterion(output[0<mask], target[0<mask])
        else:
            loss = criterion(output[0 < mask][sample_idx_list], target[0 < mask][sample_idx_list])

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))

    train_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    return train_loss
