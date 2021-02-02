import torch
from torch.autograd import Variable
import numpy as np
import random
from tqdm import tqdm

from utils.utils_DEAL import *

def train(dataloader, net, optimizer, opt, OutputDir):
    train_loss = 0
    net.train()

    # for DEAL
    device = torch.device('cuda:' + str(opt.cuda) if opt.gpu else 'cpu')
    print("Device: using ", device)
    neg_num = 1
    # node / attr /  inter
    theta_list = (0.1, 0.85, 0.05)
    lambda_list = (0.1, 0.85, 0.05)
    print(f'theta_list:{theta_list}')

    for i, (sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask) in enumerate(dataloader, 0):
        sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask = sample_idx[0], data_arrays_link, dists[0], ind_train_A[0], ind_train_X[0], nodes_keep[0], A[0], X[0], label[0], mask[0]

        # for DEAL
        A, X, A_train, X_train, data, train_ones, val_edges, test_edges, val_labels, gt_labels, nodes_keep = load_datafile(data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, opt)
        sp_X = convert_sSp_tSp(X).to(device).to_dense().double()
        sp_attrM = convert_sSp_tSp(X_train).to(device)
        val_labels = A_train[val_edges[:, 0], val_edges[:, 1]].A1

        init_delta = get_delta(np.stack(A_train.nonzero()), A_train)

        data_loader = iter(get_train_data(A_train, int(X_train.shape[0] * opt.train_ratio), np.vstack((test_edges, val_edges)), opt.inductive))  # ,neg_num
        inputs, labels = next(data_loader)

        result_list = []
        margin_dict = {}
        margin_pairs = {}
        best_state_dict = None

        for repeat in tqdm(range(opt.repeat_num)):
            for d in margin_dict:
                margin_dict[d].append([])

            max_val_score = np.zeros(1)
            val_result = np.zeros(2)

            running_loss = 0.0

            # ここからDEAL用のepoch回す
            for epoch in range(opt.epoch_num):
                inputs, labels = next(data_loader)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                #
                # forward + backward + optimize
                #
                loss = net.default_loss(inputs, labels, data, thetas=theta_list, train_num=int(X_train.shape[0] *opt.train_ratio)*2)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                b_num = 5

                if epoch % b_num == b_num - 1:
                    avg_loss = running_loss / b_num
                    val_scores = tran_eval(net, val_edges, val_labels, data, lambdas=lambda_list)

                    running_loss = 0.0
                    val_result = np.vstack((val_result, np.array(val_scores)))
                    tmp_max = np.maximum(np.mean(val_scores), max_val_score)
                    rprint('[%8d]  val %.4f %.4f' % (epoch + 1, *val_scores))
                    if tmp_max > max_val_score:
                        torch.save(net.state_dict(), OutputDir + '/checkpoint.pt')
                        max_val_score = tmp_max
                        final_scores = avg_loss, *ind_eval(net, test_edges, gt_labels, sp_X, nodes_keep, lambdas=lambda_list)
                    for tmp_d in margin_dict:
                        pairs = margin_pairs[tmp_d]
                        margin_dict[tmp_d][repeat].append(
                            [net.node_forward(pairs).mean().item(), net.attr_forward(pairs, data).mean().item()])
            print()
            print(f'ROC-AUC:{final_scores[1]:.4f} AP:{final_scores[2]:.4f}')