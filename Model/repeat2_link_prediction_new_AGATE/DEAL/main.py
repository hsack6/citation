import argparse
import random
import pandas as pd

import torch
import torch.nn as nn

from model import *

from utils.train import train
from utils.inference import inference
from utils.data.dataset import BADataset
from utils.data.dataloader import BADataloader

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )

from setting_param import Model_repeat2_link_prediction_new_AGATE_InputDir as InputDir
from setting_param import Model_repeat2_link_prediction_new_DEAL_AGATE_OutputDir as OutputDir

from setting_param import link_prediction_new_worker
from setting_param import link_prediction_new_batchSize
from setting_param import link_prediction_new_lr
from setting_param import link_prediction_new_init_L
from setting_param import link_prediction_new_annotation_dim
from setting_param import link_prediction_new_state_dim
from setting_param import link_prediction_new_output_dim
from setting_param import link_prediction_new_n_steps
from setting_param import link_prediction_new_niter
from setting_param import link_prediction_new_patience

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=link_prediction_new_worker)
parser.add_argument('--batchSize', type=int, default=link_prediction_new_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=link_prediction_new_state_dim, help='GGNN hidden state size')
parser.add_argument('--annotation_dim', type=int, default=link_prediction_new_annotation_dim, help='GGNN input annotation size')
#parser.add_argument('--output_dim', type=int, default=link_prediction_new_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=link_prediction_new_init_L, help='number of observation time step')
parser.add_argument('--niter', type=int, default=link_prediction_new_niter, help='number of epochs to train for')
parser.add_argument('--n_steps', type=int, default=link_prediction_new_n_steps, help='propogation steps number of GGNN')
parser.add_argument('--patience', type=int, default=link_prediction_new_patience, help='Early stopping patience')
#parser.add_argument('--lr', type=float, default=link_prediction_new_lr, help='learning rate')
#parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# DEAL args
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# general
parser.add_argument('--comment', dest='comment', default='0', type=str, help='comment')
parser.add_argument('--task', dest='task', default='link', type=str, help='link: link prediction; node: node classification')
parser.add_argument('--dataset', dest='dataset', default='All', type=str, help='All; Cora; grid; communities; ppi')
parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether use gpu')
parser.add_argument('--cpu', dest='gpu', action='store_false', help='whether use cpu')
parser.add_argument('--sa', dest='strong_A', action='store_true', help='use Strong Alignment')
parser.add_argument('--wa', dest='strong_A', action='store_false', help='use Weak Alignment')
parser.add_argument('--cuda', dest='cuda', default='0', type=int)
parser.add_argument('--res_dir', dest='res_dir', default=None, type=str)
parser.add_argument('--ps', dest='ps', default='', type=str)
# dataset
parser.add_argument('--train_ratio', dest='train_ratio', default=0.2, type=float, help='The ratio between the training dataset size and node numbers')
parser.add_argument('--dropout', dest='dropout', action='store_true',  help='whether dropout, default 0.3')
parser.add_argument('--dropout_no', dest='dropout', action='store_false', help='whether dropout, default 0.3')
parser.add_argument('--batch_size', dest='batch_size', default=8, type=int)  # implemented via accumulating gradient
parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
parser.add_argument('--feature_dim', dest='feature_dim', default=64, type=int)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=64, type=int)
parser.add_argument('--output_dim', dest='output_dim', default=64, type=int)
parser.add_argument('--lr', dest='lr', default=5e-2, type=float)
parser.add_argument('--epoch_num', dest='epoch_num', default=3000, type=int)
parser.add_argument('--repeat_num', dest='repeat_num', default=10, type=int)  # 10
parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)
parser.add_argument('--gamma', dest='gamma', default=2, type=float)
parser.add_argument('--approximate', dest='approximate', default=-1, type=int, help='k-hop shortest path distance. -1 means exact shortest path')  # -1, 2
parser.add_argument('--use_order', dest='use_order', default=False, type=str2bool, help='whether use Order Strategy, default False')
parser.add_argument('--ind', dest='inductive', default=True, type=str2bool, help='Inductive mode, default True')
parser.add_argument('--cache', dest='cache', default=True, type=str2bool, help='If use cache, default True')
parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.2, type=float)
parser.add_argument('--rm_feature', dest='rm_feature', default=False, type=str2bool, help='If use cache, default False')
parser.add_argument('--mode', dest='train_mode', default='cos', type=str, help='cos, dot, all, pdist, default cos')
parser.add_argument('--loss', dest='loss', default=None, type=str, help='loss function options: default, etc.')
parser.add_argument('--attr_model', dest='attr_model', default='Emb', type=str, help='Attribute embedding model, Emb, SAGE, GAT ... , default Emb')
parser.add_argument('--bce', dest='BCE_mode', default=True, type=str2bool, help='If use BCE_mode, default True')
parser.set_defaults(
    dataset='',
    gpu=False,
    layer_num=2,
    lr=1e-2,
    repeat_num=1,
    loss='default',
    epoch_num=5000,
    epoch_log=2,
    task='link',
    train_mode='cos',
    train_ratio=0.1,
)

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

from setting_param import all_node_num
from setting_param import n_expanded
opt.n_node = all_node_num + n_expanded


train_dataset = BADataset(opt.dataroot, opt.L, True, False, False)
train_dataloader = BADataloader(train_dataset, batch_size=opt.batchSize, \
                                  shuffle=True, num_workers=opt.workers, drop_last=True)

# valid_dataset = BADataset(opt.dataroot, opt.L, False, True, False)
# valid_dataloader = BADataloader(valid_dataset, batch_size=opt.batchSize, \
#                                  shuffle=True, num_workers=opt.workers, drop_last=True)

# test_dataset = BADataset(opt.dataroot, opt.L, False, False, True)
# test_dataloader = BADataloader(test_dataset, batch_size=opt.batchSize, \
#                                  shuffle=False, num_workers=opt.workers, drop_last=True)

all_dataset = BADataset(opt.dataroot, opt.L, False, False, False)
all_dataloader = BADataloader(all_dataset, batch_size=opt.batchSize, \
                                 shuffle=False, num_workers=opt.workers, drop_last=False)

device = torch.device('cuda:' + str(opt.cuda) if opt.gpu else 'cpu')
net = DEAL(opt.output_dim, opt.annotation_dim, all_node_num, device, opt, locals()[opt.attr_model])
net.double()
print(net)

if opt.cuda:
    net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

os.makedirs(OutputDir, exist_ok=True)
train(train_dataloader, net, optimizer, opt, OutputDir)
net.load_state_dict(torch.load(OutputDir + '/checkpoint.pt'))
inference(all_dataloader, net, opt, OutputDir)