from torch.autograd import Variable
import numpy as np
import os

def inference(dataloader, opt, OutputDir, edge_type):
    for i, (sample_idx, input_new, input_appeared, input_disappeared, label_new, label_appeared, label_disappeared) in enumerate(dataloader, 0):

        if edge_type == "new":
            input = Variable(input_new).double()
            target = Variable(label_new).double()
        elif edge_type == "appeared":
            input = Variable(input_appeared).double()
            target = Variable(label_appeared).double()
        elif edge_type == "disappeared":
            input = Variable(input_disappeared).double()
            target = Variable(label_disappeared).double()

        output = input[:, -1]

        # 予測結果とラベルを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            np.save(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), output.detach().numpy()[batch])
            np.save(OutputDir + "/output/true" + str(sample_idx.numpy()[batch]), target[batch].numpy())