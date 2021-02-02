from torch.autograd import Variable
import torch
import numpy as np
import networkx as nx

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import all_node_num
from setting_param import n_expanded


def test(dataloader, net, criterion, opt):
    test_loss = 0
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
        test_loss += criterion(output[0 < mask], target[0 < mask]).item()

    test_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss
