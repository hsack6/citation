from torch.autograd import Variable
import torch

def test(dataloader, net, criterion, opt):
    test_loss = 0
    net.eval()
    for i, (sample_idx, annotation, adj_matrix, label, mask) in enumerate(dataloader, 0):
        padding = torch.zeros(opt.batchSize, opt.n_node, opt.L, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 3)

        if opt.cuda:
            init_input = init_input.cuda()
            label = label.cuda()
            mask = mask.cuda()

        init_input = Variable(init_input)
        target = Variable(label)
        mask = Variable(mask)

        output = net(init_input)
        test_loss += criterion(output[0 < mask], target[0 < mask]).item()

    test_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss
