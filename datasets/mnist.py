import os
import gzip
import numpy as np
import torch as th
import torch.utils.data


def get_dataset(path: str, kind='train'):
    """
    Load MNIST data from `path`
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def get_loader(path: str, batchsize: int, kind='train'):
    """
    Load MNIST data and turn them into dataloaders
    """
    if kind == 'train':
        x_train, y_train = get_dataset(path, kind='train')
        x_train = th.from_numpy(x_train).float().view(-1, 1, 28, 28) / 255.
        y_train = th.from_numpy(y_train).long().view(-1, 1)
        data_train = th.utils.data.TensorDataset(x_train, y_train)
        loader_train = th.utils.data.DataLoader(data_train,
                                                batch_size=batchsize, shuffle=True,
                                                pin_memory=True)
        return loader_train
    else:
        x_test, y_test = get_dataset(path, kind='t10k')
        x_test = th.from_numpy(x_test).float().view(-1, 1, 28, 28) / 255.
        y_test = th.from_numpy(y_test).long().view(-1, 1)
        data_test = th.utils.data.TensorDataset(x_test, y_test)
        loader_test = th.utils.data.DataLoader(data_test,
                                               batch_size=batchsize, shuffle=False,
                                               pin_memory=True)
        return loader_test
