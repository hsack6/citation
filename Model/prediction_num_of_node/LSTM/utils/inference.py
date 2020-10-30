from torch.autograd import Variable
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )

from setting_param import prediction_num_of_node_max_new as max_new
from setting_param import prediction_num_of_node_min_new as min_new
from setting_param import prediction_num_of_node_max_lost as max_lost
from setting_param import prediction_num_of_node_min_lost as min_lost

def inference(dataloader, net, criterion, opt, OutputDir, node_type):
    net.eval()
    for i, (sample_idx, input_new, input_lost, label_new, label_lost) in enumerate(dataloader, 0):

        if opt.cuda:
            input_new = input_new.cuda()
            input_lost = input_lost.cuda()
            label_new = label_new.cuda()
            label_lost = label_lost.cuda()

        if node_type == "new":
            input = Variable(input_new).double()
            target = Variable(label_new).double()
            max_ = max_new
            min_ = min_new
        elif node_type == "lost":
            input = Variable(input_lost).double()
            target = Variable(label_lost).double()
            max_ = max_lost
            min_ = min_lost

        output = net(input)

        # 予測結果とラベルを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            np.save(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), output.detach().numpy()[batch] * (max_ - min_) + min_)
            np.save(OutputDir + "/output/true" + str(sample_idx.numpy()[batch]), target[batch].numpy() * (max_ - min_) + min_)
