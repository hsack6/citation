import argparse

from utils.inference import inference
from utils.data.dataset import BADataset
from utils.data.dataloader import BADataloader

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import Model_prediction_num_of_node_InputDir as InputDir
from setting_param import Model_prediction_num_of_node_new_Baseline_OutputDir
from setting_param import Model_prediction_num_of_node_lost_Baseline_OutputDir
from setting_param import prediction_num_of_node_worker
from setting_param import prediction_num_of_node_batchSize
from setting_param import prediction_num_of_node_init_L
from setting_param import prediction_num_of_node_state_dim
from setting_param import prediction_num_of_node_output_dim

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=prediction_num_of_node_worker)
parser.add_argument('--batchSize', type=int, default=prediction_num_of_node_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=prediction_num_of_node_state_dim, help='GGNN hidden state size')
parser.add_argument('--output_dim', type=int, default=prediction_num_of_node_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=prediction_num_of_node_init_L, help='number of observation time step')
parser.add_argument('node_type', type=str)

opt = parser.parse_args()
print(opt)


opt.dataroot = InputDir

if opt.node_type == "new":
    OutputDir = Model_prediction_num_of_node_new_Baseline_OutputDir
elif opt.node_type == "lost":
    OutputDir = Model_prediction_num_of_node_lost_Baseline_OutputDir

print(opt.node_type, InputDir, OutputDir)

opt.L = opt.init_L

def main(opt):
    all_dataset = BADataset(opt.dataroot, opt.L, False, False, False)
    all_dataloader = BADataloader(all_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=opt.workers, drop_last=False)

    inference(all_dataloader, opt, OutputDir, opt.node_type)


if __name__ == "__main__":
    main(opt)
