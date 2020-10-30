from torch.autograd import Variable
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def train(epoch, dataloader, net, criterion, optimizer, opt):
    train_loss = 0
    net.train()
    for i, (sample_idx, input, label, input_num) in enumerate(dataloader, 0):
        input = Variable(input)
        label = Variable(label)
        output = net(input)

        for batch in range(opt.batchSize):
            loss = 0
            for _ in range(input_num[batch]/16):
                n = [random.randint(0, input_num[batch]-1) for _ in range(16)]
                loss += criterion(output[batch, n, 0], label[batch, n, opt.target_idx])
                loss.backward(retain_graph=True)
                optimizer.step()

        loss = 0
        for batch in range(opt.batchSize):
            loss += criterion(output[batch, :input_num[batch], 0], label[batch, :input_num[batch], opt.target_idx])
        train_loss += loss.item()

        # loss = 0
        # for batch in range(opt.batchSize):
        #     loss += criterion(output[batch, :input_num[batch], 0], label[batch, :input_num[batch], opt.target_idx])
        # loss.backward()
        # optimizer.step()
        # train_loss += loss.item()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))

    train_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    return train_loss
