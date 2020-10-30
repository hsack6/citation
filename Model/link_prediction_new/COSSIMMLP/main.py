import argparse
import random
import pandas as pd

import torch
import torch.nn as nn

from model import COSSIMMLP
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

from setting_param import Model_link_prediction_new_Baseline_mix_InputDir as Baseline_mix_InputDir
from setting_param import Model_link_prediction_new_Baseline_learning_InputDir as Baseline_learning_InputDir
from setting_param import Model_link_prediction_new_Baseline_inference_InputDir as Baseline_inference_InputDir

from setting_param import Model_link_prediction_new_DeepMatchMax_mix_InputDir as DeepMatchMax_mix_InputDir
from setting_param import Model_link_prediction_new_DeepMatchMax_learning_InputDir as DeepMatchMax_learning_InputDir
from setting_param import Model_link_prediction_new_DeepMatchMax_inference_InputDir as DeepMatchMax_inference_InputDir

from setting_param import Model_link_prediction_new_FNN_mix_InputDir as FNN_mix_InputDir
from setting_param import Model_link_prediction_new_FNN_learning_InputDir as FNN_learning_InputDir
from setting_param import Model_link_prediction_new_FNN_inference_InputDir as FNN_inference_InputDir

from setting_param import Model_link_prediction_new_COSSIMMLP_Baseline_mix_OutputDir as COSSIMMLP_Baseline_mix_OutputDir
from setting_param import Model_link_prediction_new_COSSIMMLP_Baseline_learning_OutputDir as COSSIMMLP_Baseline_learning_OutputDir
from setting_param import Model_link_prediction_new_COSSIMMLP_Baseline_inference_OutputDir as COSSIMMLP_Baseline_inference_OutputDir

from setting_param import Model_link_prediction_new_COSSIMMLP_DeepMatchMax_mix_OutputDir as COSSIMMLP_DeepMatchMax_mix_OutputDir
from setting_param import Model_link_prediction_new_COSSIMMLP_DeepMatchMax_learning_OutputDir as COSSIMMLP_DeepMatchMax_learning_OutputDir
from setting_param import Model_link_prediction_new_COSSIMMLP_DeepMatchMax_inference_OutputDir as COSSIMMLP_DeepMatchMax_inference_OutputDir

from setting_param import Model_link_prediction_new_COSSIMMLP_FNN_mix_OutputDir as COSSIMMLP_FNN_mix_OutputDir
from setting_param import Model_link_prediction_new_COSSIMMLP_FNN_learning_OutputDir as COSSIMMLP_FNN_learning_OutputDir
from setting_param import Model_link_prediction_new_COSSIMMLP_FNN_inference_OutputDir as COSSIMMLP_FNN_inference_OutputDir

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
parser.add_argument('--output_dim', type=int, default=link_prediction_new_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=link_prediction_new_init_L, help='number of observation time step')
parser.add_argument('--niter', type=int, default=link_prediction_new_niter, help='number of epochs to train for')
parser.add_argument('--n_steps', type=int, default=link_prediction_new_n_steps, help='propogation steps number of GGNN')
parser.add_argument('--patience', type=int, default=link_prediction_new_patience, help='Early stopping patience')
parser.add_argument('--lr', type=float, default=link_prediction_new_lr, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('out_input_method', type=str)
parser.add_argument('learning_type', type=str)

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.out_input_method == "Baseline":
    if opt.learning_type == "mix":
        InputDir = Baseline_mix_InputDir
        OutputDir = COSSIMMLP_Baseline_mix_OutputDir
    elif opt.learning_type == "learning":
        InputDir = Baseline_learning_InputDir
        OutputDir = COSSIMMLP_Baseline_learning_OutputDir
    elif opt.learning_type == "inference":
        InputDir = Baseline_inference_InputDir
        OutputDir = COSSIMMLP_Baseline_inference_OutputDir
elif opt.out_input_method == "DeepMatchMax":
    if opt.learning_type == "mix":
        InputDir = DeepMatchMax_mix_InputDir
        OutputDir = COSSIMMLP_DeepMatchMax_mix_OutputDir
    elif opt.learning_type == "learning":
        InputDir = DeepMatchMax_learning_InputDir
        OutputDir = COSSIMMLP_DeepMatchMax_learning_OutputDir
    elif opt.learning_type == "inference":
        InputDir = DeepMatchMax_inference_InputDir
        OutputDir = COSSIMMLP_DeepMatchMax_inference_OutputDir
elif opt.out_input_method == "FNN":
    if opt.learning_type == "mix":
        InputDir = FNN_mix_InputDir
        OutputDir = COSSIMMLP_FNN_mix_OutputDir
    elif opt.learning_type == "learning":
        InputDir = FNN_learning_InputDir
        OutputDir = COSSIMMLP_FNN_learning_OutputDir
    elif opt.learning_type == "inference":
        InputDir = FNN_inference_InputDir
        OutputDir = COSSIMMLP_FNN_inference_OutputDir

opt.dataroot = InputDir

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

opt.L = opt.init_L

from setting_param import all_node_num
from setting_param import n_expanded
opt.n_node = all_node_num + n_expanded

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

    net = COSSIMMLP(opt)
    net.double()
    print(net)

    criterion = nn.BCELoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    #optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    optimizer = ""
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
