from torch.autograd import Variable
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test(dataloader, net, criterion, opt):
    test_loss = 0
    net.eval()
    for i, (sample_idx, input, label, input_num) in enumerate(dataloader, 0):
        input = Variable(input)
        label = Variable(label)
        output = net(input)

        loss = 0
        for batch in range(opt.batchSize):
            loss += criterion(output[batch, :input_num[batch]], label[batch, :input_num[batch], opt.target_idx].unsqueeze(1))
        test_loss += loss.item()

    test_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss
