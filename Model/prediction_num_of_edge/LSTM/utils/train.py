from torch.autograd import Variable

def train(epoch, dataloader, net, criterion, optimizer, opt, edge_type):
    train_loss = 0
    net.train()
    for i, (sample_idx, input_new, input_appeared, input_disappeared, label_new, label_appeared, label_disappeared) in enumerate(dataloader, 0):
        net.zero_grad()

        if opt.cuda:
            input_new = input_new.cuda()
            input_appeared = input_appeared.cuda()
            input_disappeared = input_disappeared.cuda()
            label_new = label_new.cuda()
            label_appeared = label_appeared.cuda()
            label_disappeared = label_disappeared.cuda()

        if edge_type == "new":
            input = Variable(input_new).double()
            target = Variable(label_new).double()
        elif edge_type == "appeared":
            input = Variable(input_appeared).double()
            target = Variable(label_appeared).double()
        elif edge_type == "disappeared":
            input = Variable(input_disappeared).double()
            target = Variable(label_disappeared).double()

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
