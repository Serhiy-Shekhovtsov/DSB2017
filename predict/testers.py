from os import path as p

import numpy as np
from torch.autograd import Variable


def test_detect(data_loader, net, get_pbb, n_gpu=0, **kwargs):
    save_dir = kwargs['save_dir']
    side_len = kwargs.get('side_len', 144)
    net.eval()
    split_comber = data_loader.dataset.split_comber

    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1]
        shortname = name.split('_clean')[0]
        data = data[0][0]
        len_data = len(data)
        coord = coord[0][0]
        isfeat = kwargs.get('output_feature')
        splitlist = range(0, len_data + 1, n_gpu) if n_gpu else range(len_data)

        if splitlist[-1] != len_data:
            splitlist = list(splitlist) + [len_data]

        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            split_i, split_i_p1 = splitlist[i], splitlist[i + 1]
            input = Variable(data[split_i:split_i_p1], volatile=True)
            inputcoord = Variable(coord[split_i:split_i_p1], volatile=True)

            if n_gpu:
                input, inputcoord = input.cuda(), inputcoord.cuda()

            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)

            outputlist.append(output.data.cpu().numpy())

        output2 = np.concatenate(outputlist, 0)
        output3 = split_comber.combine(output2, nzhw=nzhw)

        if isfeat:
            transposed = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
            feature = transposed[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, side_len)[..., 0]

        thresh = -3
        pbb, mask = get_pbb(output3, thresh, ismask=True)

        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            filepath = p.join(save_dir, shortname + '_feature.npy')
            np.save(filepath, feature_selected)

        np.save(p.join(save_dir, shortname + '_pbb.npy'), pbb)
        np.save(p.join(save_dir, shortname + '_lbb.npy'), lbb)


def test_classify(data_loader, model, n_gpu=0):
    model.eval()

    for x, coord in data_loader:
        coord = Variable(coord).cuda() if n_gpu else Variable(coord)
        x = Variable(x).cuda() if n_gpu else Variable(x)
        prediction = model(x, coord)[1]
        yield prediction.data.cpu().numpy()
