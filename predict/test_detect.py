import argparse
import os
import time
import shutil
import sys

from importlib import import_module
from os import path as p

import numpy as np
import torch

from torch import optim
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *
from split_combine import SplitComb
from layers import acc


def test_detect(data_loader, net, get_pbb, save_dir, config, n_gpu):
    start_time = time.time()
    net.eval()
    split_comber = data_loader.dataset.split_comber

    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        s = time.time()
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = config.get('output_feature')
        n_per_run = n_gpu

        print(data.size())
        splitlist = range(0, len(data) + 1, n_gpu)

        if splitlist[-1] != len(data):
            splitlist.append(len(data))

        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            input = Variable(
                data[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()

            inputcoord = Variable(
                coord[splitlist[i]:splitlist[i + 1]], volatile=True).cuda()

            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)

            outputlist.append(output.data.cpu().numpy())

        output = np.concatenate(outputlist,0)
        output = split_comber.combine(output, nzhw=nzhw)

        if isfeat:
            transposed = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
            feature = transposed[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = -3
        pbb, mask = get_pbb(output, thresh, ismask=True)

        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            filepath = p.join(save_dir, shortname + '_feature.npy')
            np.save(filepath, feature_selected)

        # tp, fp, fn, _ = acc(pbb,lbb,0,0.1,0.1)
        # print([len(tp), len(fp), len(fn)])
        print([i_name,shortname])
        e = time.time()
        np.save(p.join(save_dir, shortname + '_pbb.npy'), pbb)
        np.save(p.join(save_dir, shortname + '_lbb.npy'), lbb)

    end_time = time.time()
    print('elapsed time is %3.2f seconds' % (end_time - start_time))
