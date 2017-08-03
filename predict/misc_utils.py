"""
    Provides misc utility functions
"""
import pickle

from os import path, listdir

import torch

try:
    import boto3
except ImportError:
    ClientError = boto3 = None
else:
    from botocore.exceptions import ClientError

try:
    from s3fs import S3FileSystem
except ImportError:
    s3fs = None
else:
    s3fs = S3FileSystem()


def gen_s3_dir_names(dirpath):
    split_path = list(filter(None, dirpath.split('/')))
    split_path_len = len(split_path)

    if split_path_len < 2:
        # invalid path, e.g., s3://
        raise FileNotFoundError('No bucket found in your path.')

    bucket_name = split_path[1]
    error_msg = "No such bucket: '{}'".format(bucket_name)

    if split_path_len == 2:
        # in the bucket's top level, e.g., s3://dd-stage2
        s3client = boto3.client('s3')
        paginator = s3client.get_paginator('list_objects')
        result = paginator.paginate(Bucket=bucket_name, Delimiter='/')

        try:
            for res in result.search('CommonPrefixes'):
                yield res.get('Prefix').rstrip('/')
        except ClientError:
            raise FileNotFoundError(error_msg)
    else:
        # inside a folder so we need to filter by the prefix, e.g.,
        # s3://dd-stage2/004828796b994741c4466f59a8c7e9a4
        s3resource = boto3.resource('s3')
        bucket = s3resource.Bucket(bucket_name)
        prefix = '{}/'.format('/'.join(split_path[2:]))

        try:
            for obj in bucket.objects.filter(Prefix=prefix):
                yield obj.key.replace(prefix, '')
        except ClientError:
            raise FileNotFoundError(error_msg)


def gendir(dirpath, as_abspath=False, as_file_obj=False, exclude_hidden=True):
    """like os.listdir but returns an iterator, opens s3 paths, and optionally
    returns abspaths or file-like objects
    """
    is_s3 = dirpath.startswith('s3://')
    stripped_path = dirpath.rstrip('/')

    if is_s3 and not boto3:
        msg = 'Warning! You must install boto3 before calling this function.'
        raise ImportError(msg)
    elif is_s3:
        names = gen_s3_dir_names(stripped_path)
    else:
        names = iter(listdir(dirpath))

    if exclude_hidden:
        names = (name for name in names if not name.startswith('.'))

    if is_s3 and as_file_obj and not s3fs:
        msg = 'Warning! You must install s3fs before calling this function.'
        raise ImportError(msg)
    elif is_s3 and (as_abspath or as_file_obj):
        paths = (path.join(stripped_path, name) for name in names)
        entries = map(s3fs.open, paths) if as_file_obj else paths
    elif as_abspath or as_file_obj:
        paths = (path.join(path.abspath(stripped_path), name) for name in names)
        entries = (open(p, mode='rb') for p in paths) if as_file_obj else paths
    else:
        entries = names

    return entries


class Unpickler(pickle.Unpickler):
    def __init__(self, *args, encoding=None, **kwargs):
        # https://stackoverflow.com/a/28218598/408556
        super(Unpickler, self).__init__(*args, encoding='latin1', **kwargs)


def torch_loader(filepath):
    """Used for loading torch models in Py3 that were saved in Py2"""
    try:
        checkpoint = torch.load(filepath)
    except UnicodeDecodeError:
        pickle_module = pickle
        pickle_module.Unpickler = Unpickler

        with open(filepath, 'rb') as f:
            checkpoint = torch.load(f, pickle_module=pickle_module)

    return checkpoint
