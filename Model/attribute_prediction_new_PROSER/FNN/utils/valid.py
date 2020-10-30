from torch.autograd import Variable
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def valid(dataloader, net, criterion, opt):
    valid_loss = 0
    net.eval()
    for i, (sample_idx, input, label, input_num) in enumerate(dataloader, 0):
        input = Variable(input)
        label = Variable(label)
        output = net(input)

        loss = 0
        for batch in range(opt.batchSize):
            loss += criterion(output[batch, :input_num[batch]], label[batch, :input_num[batch], opt.target_idx].unsqueeze(1))
        valid_loss += loss.item()

    valid_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Valid set: Average loss: {:.4f}'.format(valid_loss))

    return valid_loss
