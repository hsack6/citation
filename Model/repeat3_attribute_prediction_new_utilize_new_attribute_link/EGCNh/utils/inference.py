from torch.autograd import Variable
import numpy as np
import os
import torch
from scipy.sparse import lil_matrix
from scipy.io import mmwrite
os.environ['KMP_DUPLICATE_LIB_OK']='True' # mmwriteでエラーが出るのを回避

def inference(dataloader, net, criterion, opt, OutputDir):
    net.eval()
    for i, (sample_idx, annotation, adj_matrix, label, mask) in enumerate(dataloader, 0):
        padding = torch.zeros(opt.batchSize, opt.n_node, opt.L, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 3)

        if opt.cuda:
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            init_input = init_input.cuda()
            label = label.cuda()
            mask = mask.cuda()

        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        init_input = Variable(init_input)
        target = Variable(label).float()
        mask = Variable(mask)

        output = net(init_input, annotation, adj_matrix)

        # 予測結果とラベルを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            p = output.detach().numpy()[batch]
            t = target[batch].numpy()
            m = mask[batch].numpy()
            mmwrite(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), lil_matrix(p * m)) # 対象要素以外をゼロにして疎行列にする
            mmwrite(OutputDir + "/output/true" + str(sample_idx.numpy()[batch]), lil_matrix(t))
            mmwrite(OutputDir + "/output/mask" + str(sample_idx.numpy()[batch]), lil_matrix(m))

