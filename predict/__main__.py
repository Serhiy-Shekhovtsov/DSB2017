from . import predict

import torch

n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
print('Detected {} GPU(s).'.format(n_gpu) if n_gpu else 'No GPU detected!')

kwargs = {
    # 'datapath': 's3://dd-stage2/',
    # 'datapath': 's3://dd-sample/',
    'datapath': '/Users/reubano/Documents/Projects/alcf/tests/assets/LIDC-IDRI-0002/1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329/1.3.6.1.4.1.14519.5.2.1.6279.6001.619372068417051974713149104919',
    'bbox_result_folder': './bbox_result/',
    'preprocess_result_folder': './prep_result/',
    'batch_size': 1,
    'n_gpu': n_gpu,
    'limit': None if n_gpu else 8}


print(predict(**kwargs))
