from importlib import import_module

import torch
import pandas

from torch import optim
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *
from layers import acc
from data_detector import DataBowl3Detector, collate
from data_classifier import DataBowl3Classifier
from split_combine import SplitComb
from test_detect import test_detect
from preprocessing import full_prep
from config_submit import config as config_submit

datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']
skip_prep = config_submit['skip_preprocessing']
skip_detect = config_submit['skip_detect']

if datapath.startswith('s3://'):
    print('loading %s from s3...' % datapath)
    import boto3

    client = boto3.client('s3')
    bucket_name = datapath.split('/')[2]
    paginator = client.get_paginator('list_objects')
    result = paginator.paginate(Bucket=bucket_name, Delimiter='/')

    dirlist = [
        res.get('Prefix').rstrip('/')
        for res in result.search('CommonPrefixes')]
else:
    print('loading %s from file-system...' % datapath)
    dirlist = os.listdir(datapath)

if skip_prep:
    print('skipping prep...')
    testsplit = dirlist
else:
    print('prepping...')
    processed = full_prep(
        datapath, prep_result_path,
        n_worker=config_submit['n_worker_preprocessing'],
        use_existing=config_submit['use_exsiting_preprocessing'],
        dirlist=dirlist)

print('processed %i files' % sum(processed))

nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
nodmodel_config, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param'])
nod_net.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
nod_net = nod_net.cuda()
cudnn.benchmark = True
nod_net = DataParallel(nod_net)

bbox_result_path = './bbox_result'

if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)

# dirlist = [
#     f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]

if not skip_detect:
    margin = 32
    sidelen = 144
    nodmodel_config['datadir'] = prep_result_path

    split_comber = SplitComb(
        sidelen, nodmodel_config['max_stride'], nodmodel_config['stride'], margin,
        pad_value=nodmodel_config['pad_value'])

    dataset = DataBowl3Detector(
        dirlist, nodmodel_config, phase='test', split_comber=split_comber)

    test_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=32,
        pin_memory=False, collate_fn=collate)

    test_detect(
        test_loader, nod_net, get_pbb, bbox_result_path, nodmodel_config,
        n_gpu=config_submit['n_gpu'])

casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
casenet = casemodel.CaseNet(topk=5)
casemodel_config = casemodel.config
checkpoint = torch.load(config_submit['classifier_param'])
casenet.load_state_dict(checkpoint['state_dict'])

torch.cuda.set_device(0)
casenet = casenet.cuda()
cudnn.benchmark = True
casenet = DataParallel(casenet)
filename = config_submit['outputfile']


def test_casenet(model, testset):
    data_loader = DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

    # model = model.cuda()
    model.eval()
    predlist = []

    # weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i, (x, coord) in enumerate(data_loader):
        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        nodulePred, casePred, _ = model(x,coord)
        predlist.append(casePred.data.cpu().numpy())
        # print(
        #     [i, data_loader.dataset.split[i, 1], casePred.data.cpu().numpy()])

    return np.concatenate(predlist)

casemodel_config['bboxpath'] = bbox_result_path
casemodel_config['datadir'] = prep_result_path

dataset = DataBowl3Classifier(dirlist, casemodel_config, phase='test')
predlist = test_casenet(casenet, dataset).T
anstable = np.concatenate([[dirlist], predlist], 0).T
df = pandas.DataFrame(anstable)
df.columns = {'id', 'cancer'}
df.to_csv(filename, index=False)
