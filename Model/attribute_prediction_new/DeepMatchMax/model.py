from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self, d0=3, d1=64, d2=128, d3=1024, d4=512, d5=256):
        super(STN, self).__init__()
        self.conv1 = torch.nn.Conv1d(d0, d1, 1)
        self.conv2 = torch.nn.Conv1d(d1, d2, 1)
        self.conv3 = torch.nn.Conv1d(d2, d3, 1)
        self.fc1 = nn.Linear(d3, d4)
        self.fc2 = nn.Linear(d4, d5)
        self.fc3 = nn.Linear(d5, d0*d0)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(d1)
        self.bn2 = nn.BatchNorm1d(d2)
        self.bn3 = nn.BatchNorm1d(d3)
        self.bn4 = nn.BatchNorm1d(d4)
        self.bn5 = nn.BatchNorm1d(d5)

        self.d0 = d0
        self.d3 = d3

    def forward(self, x):
        batchSize = x.size()[0]
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.d3)

        # x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.d0).flatten().astype(np.float32))).view(1,self.d0*self.d0).repeat(batchSize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.d0, self.d0)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, d0=3, d1=64, d2=128, d3=1024, d4=512, d5=256, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN(d0=d0, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5)
        self.conv1 = torch.nn.Conv1d(d0, d1, 1)
        self.conv2 = torch.nn.Conv1d(d1, d2, 1)
        self.conv3 = torch.nn.Conv1d(d2, d3, 1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.bn2 = nn.BatchNorm1d(d2)
        self.bn3 = nn.BatchNorm1d(d3)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STN(d0=d0, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5)

        self.d3 = d3

    def forward(self, x):
        """
        x: (batchSize, d0, n_node)
        """
        n_pts = x.size()[2]                  # n_node
        trans = self.stn(x)                  # (batchSize, d0, d0)
        x = x.transpose(2, 1)                # (batchSize, n_node, d0)
        x = torch.bmm(x, trans)              # (batchSize, n_node, d0)
        x = x.transpose(2, 1)                # (batchSize, d0, n_node)
        # x = F.relu(self.bn1(self.conv1(x)))  # (batchSize, d1, n_node)
        x = F.relu(self.conv1(x))  # (batchSize, d1, n_node)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x                         # (batchSize, d1, n_node)
        # x = F.relu(self.bn2(self.conv2(x)))   # (batchSize, d2, n_node)
        # x = self.bn3(self.conv3(x))           # (batchSize, d3, n_node)
        x = F.relu(self.conv2(x))   # (batchSize, d2, n_node)
        x = self.conv3(x)           # (batchSize, d3, n_node)
        x = torch.max(x, 2, keepdim=True)[0]  # (batchSize, d3, 1)
        x = x.view(-1, self.d3)               # (batchSize, d3)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.d3, 1).repeat(1, 1, n_pts) # (batchSize, d3, n_node)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet(nn.Module):
    def __init__(self, d0=3, d1=64, d2=128, d3=1024, d4=512, d5=256, d6=128, feature_transform=False):
        super(PointNet, self).__init__()
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(d0=d0, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(d1+d3, d4, 1)
        self.conv2 = torch.nn.Conv1d(d4, d5, 1)
        self.conv3 = torch.nn.Conv1d(d5, d6, 1)
        self.conv4 = torch.nn.Conv1d(d6, d0, 1)
        self.bn1 = nn.BatchNorm1d(d4)
        self.bn2 = nn.BatchNorm1d(d5)
        self.bn3 = nn.BatchNorm1d(d6)

    def forward(self, x):
        """
        x: (batchSize, n_node, d0)
        trans: (batchSize, d0, d0)
        trans_feat: None
        """
        x = x.transpose(2,1).contiguous()    # (batchSize, d0, n_node)
        x, trans, trans_feat = self.feat(x)  # (batchSize, d1+d3, n_node)
        # x = F.relu(self.bn1(self.conv1(x)))  # (batchSize, d4, n_node)
        # x = F.relu(self.bn2(self.conv2(x)))  # (batchSize, d5, n_node)
        # x = F.relu(self.bn3(self.conv3(x)))  # (batchSize, d6, n_node)
        x = F.relu(self.conv1(x))  # (batchSize, d4, n_node)
        x = F.relu(self.conv2(x))  # (batchSize, d5, n_node)
        x = F.relu(self.conv3(x))  # (batchSize, d6, n_node)
        x = self.conv4(x)                    # (batchSize, d0, n_node)
        x = x.transpose(2,1).contiguous()    # (batchSize, n_node, d0)
        return x, trans, trans_feat
