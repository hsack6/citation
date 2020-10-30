from torch.autograd import Variable
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def inference(dataloader, net, opt, OutputDir):
    net.eval()
    for i, (sample_idx, input, label, input_num) in enumerate(dataloader, 0):
        input = Variable(input)
        label = Variable(label)
        output = net(input)

        # 予測結果とノードペアを保存
        os.makedirs(OutputDir + "/output", exist_ok=True)
        for batch in range(opt.batchSize):
            np.save(OutputDir + "/output/output" + str(sample_idx.numpy()[batch]), output.detach().numpy()[batch])
            np.save(OutputDir + "/output/label" + str(sample_idx.numpy()[batch]), label[batch].numpy())
            np.save(OutputDir + "/output/input_num" + str(sample_idx.numpy()[batch]), input_num[batch].numpy())