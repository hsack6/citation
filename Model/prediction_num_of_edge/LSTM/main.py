import argparse
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTM
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
from setting_param import Model_prediction_num_of_edge_InputDir as InputDir
from setting_param import Model_prediction_num_of_edge_new_LSTM_OutputDir
from setting_param import Model_prediction_num_of_edge_appeared_LSTM_OutputDir
from setting_param import Model_prediction_num_of_edge_disappeared_LSTM_OutputDir
from setting_param import prediction_num_of_edge_worker
from setting_param import prediction_num_of_edge_batchSize
from setting_param import prediction_num_of_edge_lr
from setting_param import prediction_num_of_edge_init_L
from setting_param import prediction_num_of_edge_state_dim
from setting_param import prediction_num_of_edge_output_dim
from setting_param import prediction_num_of_edge_niter
from setting_param import prediction_num_of_edge_patience

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=prediction_num_of_edge_worker)
parser.add_argument('--batchSize', type=int, default=prediction_num_of_edge_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=prediction_num_of_edge_state_dim, help='GGNN hidden state size')
parser.add_argument('--output_dim', type=int, default=prediction_num_of_edge_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=prediction_num_of_edge_init_L, help='number of observation time step')
parser.add_argument('--niter', type=int, default=prediction_num_of_edge_niter, help='number of epochs to train for')
parser.add_argument('--patience', type=int, default=prediction_num_of_edge_patience, help='Early stopping patience')
parser.add_argument('--lr', type=float, default=prediction_num_of_edge_lr, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('edge_type', type=str)

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = InputDir

if opt.edge_type == "new":
    OutputDir = Model_prediction_num_of_edge_new_LSTM_OutputDir
elif opt.edge_type == "appeared":
    OutputDir = Model_prediction_num_of_edge_appeared_LSTM_OutputDir
elif opt.edge_type == "disappeared":
    OutputDir = Model_prediction_num_of_edge_disappeared_LSTM_OutputDir

print(opt.edge_type, InputDir, OutputDir)

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

    net = LSTM(opt, hidden_state=opt.state_dim*5)
    net.double()
    print(net)

    criterion = nn.MSELoss()

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
        train_loss = train(epoch, train_dataloader, net, criterion, optimizer, opt, opt.edge_type)
        valid_loss = valid(valid_dataloader, net, criterion, opt, opt.edge_type)
        test_loss = test(test_dataloader, net, criterion, opt, opt.edge_type)

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
    inference(all_dataloader, net, criterion, opt, OutputDir, opt.edge_type)


if __name__ == "__main__":
    main(opt)
