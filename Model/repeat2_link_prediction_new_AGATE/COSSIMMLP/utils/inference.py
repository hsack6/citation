from torch.autograd import Variable
import numpy as np
import os
import torch
from scipy.sparse import lil_matrix, csr_matrix
from scipy.io import mmwrite
os.environ['KMP_DUPLICATE_LIB_OK']='True' # mmwriteでエラーが出るのを回避

import networkx as nx

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import all_node_num
from setting_param import n_expanded


def inference(dataloader, net, criterion, opt, OutputDir):
    net.eval()
    for i, (sample_idx, annotation, label_, mask_, indic) in enumerate(dataloader, 0):
        padding = torch.zeros(opt.batchSize, opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)

        label = []
        mask = []
        for batch in range(label_.shape[0]):
            label_graph = nx.Graph()
            label_graph.add_edges_from(label_[batch].numpy())
            label.append(nx.to_numpy_matrix(label_graph, nodelist=list(range(all_node_num + n_expanded))))

            mask_graph = nx.Graph()
            mask_graph.add_edges_from(mask_[batch].numpy())
            mask.append(nx.to_numpy_matrix(mask_graph, nodelist=list(range(all_node_num + n_expanded))))
        label = torch.tensor(np.array(label), dtype=torch.double)
        mask = torch.tensor(np.array(mask), dtype=torch.double)

        if opt.cuda:
            init_input = init_input.cuda()
            label = label.cuda()
            mask = mask.cuda()

        init_input = Variable(init_input)
        target = Variable(label)
        mask = Variable(mask)

        output = net(init_input, indic, mask_)

        # 予測結果とラベルを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            # p = output.detach().to_dense().numpy()[batch]
            t = target[batch].numpy()
            m = mask[batch].numpy()

            row = list(output[batch]._indices()[0])
            col = list(output[batch]._indices()[1])
            data = list(output[batch]._values())
            p = csr_matrix((data, (row, col)), shape=(all_node_num+n_expanded, all_node_num+n_expanded))

            # mmwrite(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), lil_matrix(p * m)) # 対象要素以外をゼロにして疎行列にする
            mmwrite(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), p)
            mmwrite(OutputDir + "/output/true" + str(sample_idx.numpy()[batch]), lil_matrix(t))
            mmwrite(OutputDir + "/output/mask" + str(sample_idx.numpy()[batch]), lil_matrix(m))
