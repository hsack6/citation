from torch.autograd import Variable

def test(dataloader, net, criterion, opt, node_type):
    test_loss = 0
    net.eval()
    for i, (sample_idx, input_new, input_lost, label_new, label_lost) in enumerate(dataloader, 0):

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
        test_loss += criterion(output, target).item()

    test_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss
