from torch.autograd import Variable
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )

from setting_param import prediction_num_of_edge_max_appeared as max_appeared
from setting_param import prediction_num_of_edge_min_appeared as min_appeared
from setting_param import prediction_num_of_edge_max_disappeared as max_disappeared
from setting_param import prediction_num_of_edge_min_disappeared as min_disappeared
from setting_param import prediction_num_of_edge_max_new as max_new
from setting_param import prediction_num_of_edge_min_new as min_new

def inference(dataloader, net, criterion, opt, OutputDir, edge_type):
    net.eval()
    for i, (sample_idx, input_new, input_appeared, input_disappeared, label_new, label_appeared, label_disappeared) in enumerate(dataloader, 0):

        if opt.cuda:
            input_new = input_new.cuda()
            input_appeared = input_appeared.cuda()
            input_disappeared = input_disappeared.cuda()
            label_new = label_new.cuda()
            label_appeared = label_appeared.cuda()
            label_disappeared = label_disappeared.cuda()

        if edge_type == "new":
            input = Variable(input_new).double()
            target = Variable(label_new).double()
            max_ = max_new
            min_ = min_new
        elif edge_type == "appeared":
            input = Variable(input_appeared).double()
            target = Variable(label_appeared).double()
            max_ = max_appeared
            min_ = min_appeared
        elif edge_type == "disappeared":
            input = Variable(input_disappeared).double()
            target = Variable(label_disappeared).double()
            max_ = max_disappeared
            min_ = min_disappeared

        output = net(input)

        # 予測結果とラベルを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            np.save(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), output.detach().numpy()[batch] * (max_ - min_) + min_)
            np.save(OutputDir + "/output/true" + str(sample_idx.numpy()[batch]), target[batch].numpy() * (max_ - min_) + min_)
