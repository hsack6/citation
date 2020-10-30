from torch.autograd import Variable
import numpy as np
import copy
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def Matching(n_list_tup, t_list_tup, n, t):
    pair_list = []
    decided_n = set()
    decided_t = set()
    while True:
        for idx in range(len(n)):
            matched_n, matched_t = n[idx], t[idx]
            if matched_n in decided_n or matched_t in decided_t:
                continue
            pair_list.append((n_list_tup[matched_n][0], t_list_tup[matched_t][0]))
            decided_n.add(matched_n)
            decided_t.add(matched_t)
            if len(decided_n) == len(n_list_tup):
                break
            if len(decided_t) == len(t_list_tup):
                decided_t = set()
                break
        if len(decided_n) == len(n_list_tup):
            break
    return pair_list

def BipartiteMatching(new_vec_dic, teacher_vec_dic):
    eps = 0.000001 # zero-division errorの回避用
    # sort済みのタプルのリスト[(node_id, vector)]
    n_list_tup = sorted(new_vec_dic.items(), key=lambda x: x[0])
    t_list_tup = sorted(teacher_vec_dic.items(), key=lambda x: x[0])
    # 二部グラフ全てのsimilarityを計算
    N = np.array([n_v for n, n_v in n_list_tup])
    T = np.array([t_v for t, t_v in t_list_tup])
    normN = np.sqrt(np.sum(N * N, axis=1)) + eps
    normT = np.sqrt(np.sum(T * T, axis=1)) + eps
    similarity_matrix = np.dot(N / normN.reshape(-1, 1), (T / normT.reshape(-1, 1)).T)
    # similarityのsort
    n, t = np.unravel_index(np.argsort(-similarity_matrix.reshape(-1)), similarity_matrix.shape)
    # Greedy Matching
    node_pair_list = Matching(copy.copy(n_list_tup), copy.copy(t_list_tup), n.tolist(), t.tolist())
    return node_pair_list, similarity_matrix

def inference(dataloader, net, opt, OutputDir):
    gain = 0
    net.eval()
    for i, (sample_idx, new, teacher, new_num, teacher_num, teacher_idx) in enumerate(dataloader, 0):
        new = Variable(new)
        teacher = Variable(teacher)

        new, _, _ = net(new)

        similarity = 0
        node_pair_list_batch = []
        for batch in range(opt.batchSize):
            new_vec_dic = {i: new[batch][i].tolist() for i in range(new_num[batch])}
            teacher_vec_dic = {i: teacher[batch][i].tolist() for i in range(teacher_num[batch])}
            node_pair_list, similarity_matrix = BipartiteMatching(new_vec_dic, teacher_vec_dic)

            score = 0
            for i in range(new_num[batch]):
                score += similarity_matrix[node_pair_list[i]]
            score /= new_num[batch].float()
            similarity += score

            node_pair_list = [(i, teacher_idx[batch][j]) for (i, j) in node_pair_list]
            node_pair_list_batch.append(node_pair_list)

        similarity /= opt.batchSize
        node_pair_list_batch = np.array(node_pair_list_batch)
        gain += similarity
        print("DeepMatchMax_gain:" + str(similarity))

        # 予測結果とノードペアを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            np.save(OutputDir + "/output/new" + str(sample_idx.numpy()[batch]), new.detach().numpy()[batch])
            np.save(OutputDir + "/output/teacher" + str(sample_idx.numpy()[batch]), teacher[batch].numpy())
            np.save(OutputDir + "/output/new_num" + str(sample_idx.numpy()[batch]), new_num[batch].numpy())
            np.save(OutputDir + "/output/teacher_num" + str(sample_idx.numpy()[batch]), teacher_num[batch].numpy())
            np.save(OutputDir + "/output/teacher_idx" + str(sample_idx.numpy()[batch]), teacher_idx[batch].numpy())
            np.save(OutputDir + "/output/node_pair_list" + str(sample_idx.numpy()[batch]), node_pair_list_batch[batch])

    gain /= (len(dataloader.dataset) / opt.batchSize)
    loss = -1 * np.log((gain + 1)/2)
    print(
            'DeepMatchMax loss: {:.4f}, DeepMatchMax gain: {:.4f}'.format(
                loss, gain))