from torch.autograd import Variable
import torch

def test(dataloader, net, criterion, opt):
    test_loss = 0
    net.eval()
    for i, (sample_idx, annotation, adj_matrix, label, mask) in enumerate(dataloader, 0):
        padding = torch.zeros(opt.batchSize, opt.n_node, opt.L, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 3)

        if opt.cuda:
            adj_matrix = adj_matrix.cuda()
            annotation = annotation.cuda()
            init_input = init_input.cuda()
            label = label.cuda()
            mask = mask.cuda()

        adj_matrix = Variable(adj_matrix)
        annotation = Variable(annotation)
        init_input = Variable(init_input)
        target = Variable(label)
        mask = Variable(mask)

        output = net(init_input, annotation, adj_matrix)
        loss = 0
        for batch in range(opt.batchSize):
            o = output[batch, opt.n_existing_node:opt.n_existing_node + mask[batch]]
            t = target[batch, :mask[batch]]
            loss += 1 - criterion(o, t).mean()
        loss /= opt.batchSize
        test_loss += loss.item()

    test_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss
