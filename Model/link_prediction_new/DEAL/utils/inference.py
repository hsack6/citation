from torch.autograd import Variable
import numpy as np
import os
import torch
from scipy.sparse import lil_matrix
from scipy.io import mmwrite
from utils.utils_DEAL import *

import sys
import os
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )
from setting_param import all_node_num
from setting_param import n_expanded

os.environ['KMP_DUPLICATE_LIB_OK']='True' # mmwriteでエラーが出るのを回避

def inference(dataloader, net, opt, OutputDir):
    net.eval()
    lambda_list = (0.1, 0.85, 0.05)
    device = torch.device('cuda:' + str(opt.cuda) if opt.gpu else 'cpu')

    for i, (sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask) in enumerate(dataloader, 0):
        sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask = sample_idx[0], data_arrays_link, dists[0], ind_train_A[0], ind_train_X[0], nodes_keep[0], A[0], X[0], label[0], mask[0]

        if opt.cuda:
            label = label.cuda()
            mask = mask.cuda()
        target = Variable(label)
        mask = Variable(mask)

        # for DEAL
        A, X, A_train, X_train, data, train_ones, val_edges, test_edges, val_labels, gt_labels, nodes_keep = load_datafile(data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, opt)
        sp_X = convert_sSp_tSp(X).to(device).to_dense().double()


        true_, pred_ = get_pred(net, test_edges, gt_labels, sp_X, nodes_keep, lambdas=lambda_list)

        true = np.zeros((all_node_num+n_expanded, all_node_num+n_expanded))
        true[all_node_num:] = true_.reshape(n_expanded, all_node_num+n_expanded)
        true_between_new_nodes = true[all_node_num:, all_node_num:]
        true = true + true.T
        true[all_node_num:, all_node_num:] = true_between_new_nodes

        pred = np.zeros((all_node_num+n_expanded, all_node_num+n_expanded))
        pred[all_node_num:] = pred_.reshape(n_expanded, all_node_num+n_expanded)
        pred_between_new_nodes = pred[all_node_num:, all_node_num:]
        pred = pred + pred.T
        pred[all_node_num:, all_node_num:] = pred_between_new_nodes

        #assert np.all(target.numpy().astype(np.int32) == true.astype(np.int32)) # (マッチングで割り当てられずに余った真の new nodeのリンクはall_node_num以下の座標に入ったままになっているためそれを検知するがmask処理で無視されるので大丈夫)

        # 予測結果とラベルを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        p = pred
        t = target.numpy()
        m = mask.numpy()
        mmwrite(OutputDir + "/output/pred" + str(sample_idx.numpy()), lil_matrix(p * m)) # 対象要素以外をゼロにして疎行列にする
        mmwrite(OutputDir + "/output/true" + str(sample_idx.numpy()), lil_matrix(t))
        mmwrite(OutputDir + "/output/mask" + str(sample_idx.numpy()), lil_matrix(m))
