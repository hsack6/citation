import argparse
from utils.inference import inference
from utils.data.dataset import BADataset
from utils.data.dataloader import BADataloader

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import Model_prediction_num_of_edge_InputDir as InputDir
from setting_param import Model_prediction_num_of_edge_new_Baseline_OutputDir
from setting_param import Model_prediction_num_of_edge_appeared_Baseline_OutputDir
from setting_param import Model_prediction_num_of_edge_disappeared_Baseline_OutputDir
from setting_param import prediction_num_of_edge_worker
from setting_param import prediction_num_of_edge_batchSize
from setting_param import prediction_num_of_edge_init_L
from setting_param import prediction_num_of_edge_state_dim
from setting_param import prediction_num_of_edge_output_dim

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=prediction_num_of_edge_worker)
parser.add_argument('--batchSize', type=int, default=prediction_num_of_edge_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=prediction_num_of_edge_state_dim, help='GGNN hidden state size')
parser.add_argument('--output_dim', type=int, default=prediction_num_of_edge_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=prediction_num_of_edge_init_L, help='number of observation time step')
parser.add_argument('edge_type', type=str)

opt = parser.parse_args()
print(opt)

opt.dataroot = InputDir

if opt.edge_type == "new":
    OutputDir = Model_prediction_num_of_edge_new_Baseline_OutputDir
elif opt.edge_type == "appeared":
    OutputDir = Model_prediction_num_of_edge_appeared_Baseline_OutputDir
elif opt.edge_type == "disappeared":
    OutputDir = Model_prediction_num_of_edge_disappeared_Baseline_OutputDir

print(opt.edge_type, InputDir, OutputDir)

opt.L = opt.init_L

def main(opt):
    all_dataset = BADataset(opt.dataroot, opt.L, False, False, False)
    all_dataloader = BADataloader(all_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=opt.workers, drop_last=False)

    inference(all_dataloader, opt, OutputDir, opt.edge_type)


if __name__ == "__main__":
    main(opt)
