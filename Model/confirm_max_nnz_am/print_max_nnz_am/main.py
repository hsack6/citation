import argparse
from utils.data.dataset import BADataset

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )

# independentの場合
from setting_param import Model_link_prediction_appeared_InputDir as InputDir

# repeat1 の場合は下記のうちの大きい方
#from setting_param import MakeSample_repeat1_link_prediction_appeared_utilize_existing_attribute_OutputDir as InputDir
#from setting_param import MakeSample_repeat1_link_prediction_appeared_utilize_new_attribute_link_OutputDir as InputDir
#from setting_param import MakeSample_repeat1_link_prediction_appeared_utilize_appeared_OutputDir as InputDir

from setting_param import link_prediction_appeared_worker
from setting_param import link_prediction_appeared_batchSize
from setting_param import link_prediction_appeared_init_L
from setting_param import link_prediction_appeared_state_dim
from setting_param import link_prediction_appeared_output_dim

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=link_prediction_appeared_worker)
parser.add_argument('--batchSize', type=int, default=link_prediction_appeared_batchSize, help='input batch size')
parser.add_argument('--state_dim', type=int, default=link_prediction_appeared_state_dim, help='GGNN hidden state size')
parser.add_argument('--output_dim', type=int, default=link_prediction_appeared_output_dim, help='Model output state size')
parser.add_argument('--init_L', type=int, default=link_prediction_appeared_init_L, help='number of observation time step')

opt = parser.parse_args()
print(opt)

opt.dataroot = InputDir

opt.L = opt.init_L

def main(opt):
    BADataset(opt.dataroot, opt.L, False, False, False)

if __name__ == "__main__":
    main(opt)
