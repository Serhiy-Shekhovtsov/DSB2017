from os import path as p, mkdir

import numpy as np
import pandas as pd
import torch

from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader

from . import pre_processor, net_detector, net_classifier
from .data_classifier import DataBowl3Classifier
from .data_detector import DataBowl3Detector, collate
from .testers import test_classify, test_detect
from .utils import SplitComb
from .misc_utils import gendir, torch_loader

TOP_DIR = p.dirname(p.abspath(p.dirname(__file__)))


def predict(datapath=None, **kwargs):
    batch_size = kwargs['batch_size']
    n_gpu = kwargs['n_gpu']
    bbox_result_path = p.join(TOP_DIR, kwargs['bbox_result_folder'])
    prep_result_path = p.join(TOP_DIR, kwargs['preprocess_result_folder'])
    paths = kwargs.get('paths', list(gendir(datapath)))
    detector_path = p.join(TOP_DIR, 'model', 'detector.ckpt')
    classifier_path = p.join(TOP_DIR, 'model', 'classifier.ckpt')

    if not p.exists(bbox_result_path):
        mkdir(bbox_result_path)

    if p.splitext(paths[0])[1]:
        # we have a directory of DICOM images
        datapath, _dirlist = p.split(datapath)
        dirlist = [_dirlist]
    else:
        # we have a directory of directories of DICOM images
        dirlist = paths

    pre_processor.process(datapath, prep_result_path, dirlist=dirlist, **kwargs)

    if n_gpu:
        torch.cuda.device(0)

    cudnn.benchmark = True

    # detector
    detector_checkpoint = torch_loader(detector_path)
    nod_net = net_detector.Detector(datadir=prep_result_path)
    nod_net.load_state_dict(detector_checkpoint['state_dict'])
    parallel_nod_net = DataParallel(nod_net.cuda()) if n_gpu else nod_net
    split_comber = SplitComb(**nod_net.config)
    detect_dataset = DataBowl3Detector(
        dirlist, split_comber=split_comber, **nod_net.config)

    dl_kwargs = {
        'batch_size': batch_size, 'shuffle': False, 'pin_memory': False,
        'collate_fn': collate}

    detect_loader = DataLoader(detect_dataset, **dl_kwargs)

    _td_kwargs = {'save_dir': bbox_result_path, 'n_gpu': n_gpu}
    td_kwargs = dict(**_td_kwargs, **nod_net.config)
    test_detect(detect_loader, parallel_nod_net, nod_net.get_pbb, **td_kwargs)

    # classifier
    classifier_checkpoint = torch_loader(classifier_path)
    ckwargs = {
        'bboxpath': bbox_result_path, 'datadir': prep_result_path, 'topk': 5}

    case_net = net_classifier.Classifier(**ckwargs)
    case_net.load_state_dict(classifier_checkpoint['state_dict'])
    parallel_case_net = DataParallel(case_net.cuda()) if n_gpu else case_net
    classify_dataset = DataBowl3Classifier(dirlist, **case_net.config)
    dl_kwargs = {'batch_size': batch_size, 'shuffle': False, 'pin_memory': True}

    classify_loader = DataLoader(classify_dataset, **dl_kwargs)
    _predictions = test_classify(
        classify_loader, parallel_case_net, n_gpu=n_gpu)

    predictions = np.concatenate(list(_predictions))
    anstable = np.concatenate([[dirlist], predictions.T], 0).T
    df = pd.DataFrame(anstable)
    df.columns = {'id', 'cancer'}
    return df
