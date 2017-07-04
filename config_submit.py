import torch

n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
print('Detected {} GPU(s).'.format(n_gpu) if n_gpu else 'No GPU detected!')

config = {
    'datapath': 's3://dd-stage2/',
    'preprocess_result_path': './prep_result/',
    'outputfile': 'prediction.csv',

    'detector_model': 'net_detector',
    'detector_param': './model/detector.ckpt',
    'classifier_model': 'net_classifier',
    'classifier_param': './model/classifier.ckpt',
    'n_gpu': n_gpu,
    'n_worker_preprocessing': None,
    'use_exsiting_preprocessing': False,
    'skip_preprocessing': False,
    'skip_detect': False}
