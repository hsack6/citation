from torch.autograd import Variable

def train(epoch, dataloader, net, criterion, optimizer, opt, node_type):
    train_loss = 0
    net.train()
    for i, (sample_idx, input_new, input_lost, label_new, label_lost) in enumerate(dataloader, 0):
        net.zero_grad()

        if opt.cuda:
            input_new = input_new.cuda()
            input_lost = input_lost.cuda()
            label_new = label_new.cuda()
            label_lost = label_lost.cuda()

        if node_type == "new":
            input = Variable(input_new).double()
            target = Variable(label_new).double()
        elif node_type == "lost":
            input = Variable(input_lost).double()
            target = Variable(label_lost).double()

        output = net(input)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.item()))

    train_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    return train_loss
