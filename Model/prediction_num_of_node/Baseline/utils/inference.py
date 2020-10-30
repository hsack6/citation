from torch.autograd import Variable
import numpy as np
import os

def inference(dataloader, opt, OutputDir, node_type):
    for i, (sample_idx, input_new, input_lost, label_new, label_lost) in enumerate(dataloader, 0):

        if node_type == "new":
            input = Variable(input_new).double()
            target = Variable(label_new).double()
        elif node_type == "lost":
            input = Variable(input_lost).double()
            target = Variable(label_lost).double()

        output = input[:, -1]

        # 予測結果とラベルを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            np.save(OutputDir + "/output/pred" + str(sample_idx.numpy()[batch]), output.detach().numpy()[batch])
            np.save(OutputDir + "/output/true" + str(sample_idx.numpy()[batch]), target[batch].numpy())
