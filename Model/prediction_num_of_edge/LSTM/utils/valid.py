from torch.autograd import Variable

def valid(dataloader, net, criterion, opt, edge_type):
    valid_loss = 0
    net.eval()
    for i, (sample_idx, input_new, input_appeared, input_disappeared, label_new, label_appeared, label_disappeared) in enumerate(dataloader, 0):

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
        valid_loss += criterion(output, target).item()

    valid_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Valid set: Average loss: {:.4f}'.format(valid_loss))

    return valid_loss
