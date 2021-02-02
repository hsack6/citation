import argparse
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import GCN
from utils.train import train
from utils.valid import valid
from utils.test import test
from utils.inference import inference
from utils.data.dataset import BADataset
from utils.data.dataloader import BADataloader
from utils.pytorchtools import EarlyStopping

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import Model_repeat3_attribute_prediction_new_utilize_new_attribute_link_InputDir as InputDir
from setting_param import Model_repeat3_attribute_prediction_new_utilize_new_attribute_link_GCN_OutputDir as OutputDir
from setting_param import all_node_num
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_worker
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_batchSize
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_lr
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_init_L
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_annotation_dim
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_state_dim
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_output_dim
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_n_steps
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_niter
from setting_param import repeat3_attribute_prediction_new_utilize_new_attribute_link_patience

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=repeat3_attribute_prediction_new_utilize_new_attribute_link_worker)
parser.add_argument('--batchSize', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_state_dim, help='GGNN hidden state size')
parser.add_argument('--annotation_dim', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_annotation_dim, help='GGNN input annotation size')
parser.add_argument('--output_dim', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_init_L, help='number of observation time step')
parser.add_argument('--niter', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_niter, help='number of epochs to train for')
parser.add_argument('--n_steps', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_n_steps, help='propogation steps number of GGNN')
parser.add_argument('--patience', type=int, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_patience, help='Early stopping patience')
parser.add_argument('--lr', type=float, default=repeat3_attribute_prediction_new_utilize_new_attribute_link_lr, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = InputDir

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

opt.L = opt.init_L

def main(opt):
    train_dataset = BADataset(opt.dataroot, opt.L, True, False, False)
    train_dataloader = BADataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=opt.workers, drop_last=True)

    valid_dataset = BADataset(opt.dataroot, opt.L, False, True, False)
    valid_dataloader = BADataloader(valid_dataset, batch_size=opt.batchSize, \
                                     shuffle=True, num_workers=opt.workers, drop_last=True)

    test_dataset = BADataset(opt.dataroot, opt.L, False, False, True)
    test_dataloader = BADataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=True, num_workers=opt.workers, drop_last=True)

    all_dataset = BADataset(opt.dataroot, opt.L, False, False, False)
    all_dataloader = BADataloader(all_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=opt.workers, drop_last=False)

    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node
    opt.n_existing_node = all_node_num

    net = GCN(opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=opt.state_dim, annotation_dim_bottleneck=opt.annotation_dim)
    net.double()
    print(net)

    criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True)

    os.makedirs(OutputDir, exist_ok=True)
    train_loss_ls = []
    valid_loss_ls = []
    test_loss_ls = []

    for epoch in range(0, opt.niter):
        train_loss = train(epoch, train_dataloader, net, criterion, optimizer, opt)
        valid_loss = valid(valid_dataloader, net, criterion, opt)
        test_loss = test(test_dataloader, net, criterion, opt)

        train_loss_ls.append(train_loss)
        valid_loss_ls.append(valid_loss)
        test_loss_ls.append(test_loss)

        early_stopping(valid_loss, net, OutputDir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    df = pd.DataFrame({'epoch':[i for i in range(1, len(train_loss_ls)+1)], 'train_loss': train_loss_ls, 'valid_loss': valid_loss_ls, 'test_loss': test_loss_ls})
    df.to_csv(OutputDir + '/loss.csv', index=False)

    net.load_state_dict(torch.load(OutputDir + '/checkpoint.pt'))
    inference(all_dataloader, net, criterion, opt, OutputDir)


if __name__ == "__main__":
    main(opt)
