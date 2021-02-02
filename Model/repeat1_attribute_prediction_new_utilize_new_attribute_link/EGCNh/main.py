import argparse
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import EGCN
from utils.train import train
from utils.valid import valid
from utils.test import test
from utils.inference import inference
from utils.data.dataset import BADataset
from utils.data.dataloader import BADataloader
from utils.pytorchtools import EarlyStopping

import egcn_utils as u

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import Model_repeat1_attribute_prediction_new_utilize_new_attribute_link_InputDir as InputDir
from setting_param import Model_repeat1_attribute_prediction_new_utilize_new_attribute_link_EGCNh_OutputDir as OutputDir
from setting_param import all_node_num
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_worker
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_batchSize
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_lr
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_init_L
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_annotation_dim
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_state_dim
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_output_dim
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_n_steps
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_niter
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_patience

from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_feats_per_node
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_feats_per_node_min
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_feats_per_node_max
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_1_feats
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_1_feats_min
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_1_feats_max
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_2_feats
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_2_feats_same_as_l1
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_k_top_grcu
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_num_layers
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_layers
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_feats
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_feats_min
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_feats_max
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l2_layers
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l2_feats
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l2_feats_same_as_l1
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_cls_feats
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_cls_feats_min
from setting_param import repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_cls_feats_max

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=repeat1_attribute_prediction_new_utilize_new_attribute_link_worker)
parser.add_argument('--batchSize', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_state_dim, help='GGNN hidden state size')
parser.add_argument('--annotation_dim', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_annotation_dim, help='GGNN input annotation size')
parser.add_argument('--output_dim', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_init_L, help='number of observation time step')
parser.add_argument('--niter', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_niter, help='number of epochs to train for')
parser.add_argument('--n_steps', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_n_steps, help='propogation steps number of GGNN')
parser.add_argument('--patience', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_patience, help='Early stopping patience')
parser.add_argument('--lr', type=float, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_lr, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--feats_per_node', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_feats_per_node)
parser.add_argument('--feats_per_node_min', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_feats_per_node_min)
parser.add_argument('--feats_per_node_max', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_feats_per_node_max)
parser.add_argument('--layer_1_feats', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_1_feats)
parser.add_argument('--layer_1_feats_min', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_1_feats_min)
parser.add_argument('--layer_1_feats_max', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_1_feats_max)
parser.add_argument('--layer_2_feats', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_2_feats)
parser.add_argument('--layer_2_feats_same_as_l1', type=bool, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_layer_2_feats_same_as_l1)
parser.add_argument('--k_top_grcu', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_k_top_grcu)
parser.add_argument('--num_layers', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_num_layers)
parser.add_argument('--lstm_l1_layers', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_layers)
parser.add_argument('--lstm_l1_feats', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_feats)
parser.add_argument('--lstm_l1_feats_min', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_feats_min)
parser.add_argument('--lstm_l1_feats_max', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l1_feats_max)
parser.add_argument('--lstm_l2_layers', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l2_layers)
parser.add_argument('--lstm_l2_feats', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l2_feats)
parser.add_argument('--lstm_l2_feats_same_as_l1', type=bool, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_lstm_l2_feats_same_as_l1)
parser.add_argument('--cls_feats', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_cls_feats)
parser.add_argument('--cls_feats_min', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_cls_feats_min)
parser.add_argument('--cls_feats_max', type=int, default=repeat1_attribute_prediction_new_utilize_new_attribute_link_egcnh_parameters_cls_feats_max)

opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = InputDir

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

opt.L = opt.init_L

gcn_parameters = ['feats_per_node', 'feats_per_node_min', 'feats_per_node_max', 'layer_1_feats', 'layer_1_feats_min', 'layer_1_feats_max', 'layer_2_feats', 'layer_2_feats_same_as_l1', 'k_top_grcu', 'num_layers', 'lstm_l1_layers', 'lstm_l1_feats', 'lstm_l1_feats_min', 'lstm_l1_feats_max', 'lstm_l2_layers', 'lstm_l2_feats', 'lstm_l2_feats_same_as_l1', 'cls_feats', 'cls_feats_min', 'cls_feats_max', 'output_dim']
gcn_args = {k:opt.__dict__[k] for k in gcn_parameters}
gcn_args = u.Namespace(gcn_args)
gcn_args.feats_per_node = repeat1_attribute_prediction_new_utilize_new_attribute_link_state_dim
opt.device = 'cuda' if opt.cuda else 'cpu'

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

    net = EGCN(gcn_args, activation = torch.nn.RReLU(), device = opt.device)
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

    #net.load_state_dict(torch.load(OutputDir + '/checkpoint.pt'))
    net = torch.load(OutputDir + '/checkpoint.pt')
    inference(all_dataloader, net, criterion, opt, OutputDir)


if __name__ == "__main__":
    main(opt)
