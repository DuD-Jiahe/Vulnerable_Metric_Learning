"""
# -*-coding:utf-8-*-
Theme: lenet triplet for triple loss
Author: JiaHe Du
Date: 2023.04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml, os
import collections
import datasets
from utils import get_embeddings_from_dataloader, get_embeddings_from_images, get_poi_emb_from_images
from utils import get_poison_embeddings_from_dataloader, get_pgd_emb_from_dataloader
from utils import knn_acc, knn_pred
import random

class lenet_model(nn.Module):

    def to(self, device):
        self.device = device
        return self._apply(lambda x: x.to(device))

    def __init__(self):
        super(lenet_model, self).__init__()
        # self.conv1 = nn.Conv2d(3, 20, 5, stride=1)
        self.conv1 = nn.Conv2d(1, 20, 5, stride=1) #input, output, kernel, stride
        self.conv2 = nn.Conv2d(20, 50, 5, stride=1)
        self.fc1 = nn.Linear(800, 500)

    def forward(self, x, *, l2norm=False):
        # -1, 1, 28, 28
        x = self.conv1(x)
        # -1, 20, 24, 24
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 20, 12, 12
        x = self.conv2(x)
        # -1, 50, 8, 8
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 50, 4, 4
        x = x.view(-1, 4*4*50)
        # -1, 800
        x = nn.functional.relu(self.fc1(x))
        # -1, 500
        # x = self.fc2(x)
        if l2norm:
            x = x / x.norm(2, dim=1, keepdim=True).expand(*x.shape)
        return x

    def getloader(self, kind:str='train', batchsize:int=1):
        config = yaml.load(open('config.yml', 'r').read(),
                Loader=yaml.SafeLoader)
        if kind == 'train':
            return datasets.mnist.get_loader(
                    os.path.expanduser(config['mnist']['path']),
                    batchsize, 'train')
        else:
            return datasets.mnist.get_loader(
                    os.path.expanduser(config['mnist']['path']),
                    batchsize, 't10k')

    def split_train_val(self, loader, batchsize:int=1, split_ratio: float=0.9):
        total_len = len(loader.dataset)
        split_len = int(split_ratio*total_len)
        train = loader.dataset[0:split_len]
        val = loader.dataset[split_len:total_len]
        train = torch.utils.data.TensorDataset(train[0], train[1])
        loader_train = torch.utils.data.DataLoader(train, batch_size=batchsize, shuffle=True, num_workers=2)
        val = torch.utils.data.TensorDataset(val[0], val[1])
        loader_val = torch.utils.data.DataLoader(val, batch_size=batchsize, shuffle=True, num_workers=2)
        return loader_train, loader_val

    # TRIPLE LOSS OF CLEAN MODEL
    def loss(self, x, y, *, margin=0.5, hard=True):
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        output = self.forward(x, l2norm=False)
        # print('output.shape:', output.shape)
        # print('output.shape[0]:', output.shape[0])

        # idxp=sam class,idxn,diff class
        Y = y.cpu().numpy()
        idxa, idxp, idxn = [], [], []
        for i in range(10):  # 10 classes
            # 0414 ValueError: 'a' cannot be empty unless no samples are taken
            try:
                ida = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idp = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idn = np.random.choice(np.where(Y != i)[0],
                                       int(output.shape[0] / 10), replace=True)
                # print('ida:', ida, 'ida.shape:', ida.shape)
                idxa.extend(list(ida))
                idxp.extend(list(idp))
                idxn.extend(list(idn))
                # print('idxa:', idxa,'idxa.len:', len(idxa))
            except ValueError:
                print(f'No labels = {i}')
                continue

        XA = output[torch.LongTensor(idxa)]
        # print('XA:', XA, 'XA.shape:', XA.shape)
        # XA:100X500
        XP = output[torch.LongTensor(idxp)]
        XN = output[torch.LongTensor(idxn)]

        # loss = torch.nn.TripletMarginLoss(XA, XP, XN, margin=margin, reduction='mean')
        loss = F.triplet_margin_loss(XA, XP, XN, margin=margin, reduction='mean')
        return output, loss

    # REPORT OF TRAINING EPOCH
    def report(self, epoch, iteration, total, output, labels, loss):
        M, K = output.shape[0], output.shape[1]
        pdist = (output.view(M,1,K).expand(M,M,K) - output.view(1,M,K).expand(M,M,K)).norm(dim=2)
        offdiagdist = pdist + 1e9*torch.eye(pdist.shape[0]).to(output.device)
        #offdiagdist = pdist + 1e9 * torch.eye(pdist.shape[0]).to(self.device)
        #knnsearch = labels[torch.min(offdiagdist, dim=1)[1]].cpu()
        # print(output.shape)
        # print(output.view(1,M,K).expand(M,M,K).shape)
        # print(pdist.shape)
        # print(offdiagdist.shape)
        # print(torch.min(offdiagdist, dim=1)[1])
        knnsearch = labels[torch.min(offdiagdist, dim=1)[1]].cpu()
        acc = 100.*knnsearch.eq(labels.cpu()).sum().item() / len(labels)
        print(f'Eph[{epoch}][{iteration}/{total}]',
                collections.namedtuple('Result', ['loss', 'r_1'])(
                    '%.4f'%loss.item(), '%.3f'%acc))

    #  BEFORE TRAINING
    def val(self, model, test_loader, device: torch.device):
        model.eval()
        embeddings_test, labels_test = get_embeddings_from_dataloader(test_loader, model, device)
        correct = []
        for i in range(embeddings_test.shape[0]):
            pdist = (embeddings_test - embeddings_test[i].view(1, embeddings_test.shape[1])).norm(dim=1)

            knnsrch = labels_test[pdist.topk(k=2, largest=False)[1][1]]
            # topk：将高维数组沿某一维度（该维度共N项），选出最大（最小）的K项并排序。返回排序结果和index信息
            correct.append((knnsrch == labels_test[i]).item())
        correct = np.sum(correct)
        total = len(labels_test)
        #result = f'rank@1= {100. * correct / total} (/100)'
        result = 100. * correct / total
        return result

    # VALIDATION FOR CLEAN DATA
    def new_val(self, model, train_loader, test_loader, device: torch.device):
        model.eval()
        ### get train & test emb ################################
        embeddings_test, labels_test = get_embeddings_from_dataloader(test_loader, model, device)
        embeddings_train, labels_train = get_embeddings_from_dataloader(train_loader, model, device)
        # embeddings_train, labels_train = embeddings_train[0:10000], labels_train[0:10000]
        total_len = len(embeddings_train)
        index = random.sample(range(0, total_len), 5000)
        embeddings_train, labels_train = embeddings_train[index], labels_train[index]
        # correct = []
        # for i in range(embeddings_test.shape[0]):
        #     # pdist = (embeddings_test - embeddings_test[i].view(1, embeddings_test.shape[1])).norm(dim=1)
        #     # test sample的label未知，所以找到最近的5个train sample
        #     pdist = (embeddings_train - embeddings_test[i].view(1, embeddings_test.shape[1])).norm(dim=1)
        #     # knnsrch = labels_test[pdist.topk(k=5, largest=False)[1][1]]
        #     # topk：将高维数组沿某一维度（该维度共N项），选出最大（最小）的K项并排序。返回排序结果和index信息
        #     train_topk = pdist.topk(k=5, largest=False)[1]
        #
        #     nearest_n = []
        #     for label in train_topk:
        #         # print(f'train[{label.item()}] = {labels_train[label].item()}', end='   ')
        #         nearest_n.append(labels_train[label].item())
        #     count_max = np.bincount(nearest_n)  # count=[0 0 0 0 0 0 2 2]
        #     num_max = np.bincount(count_max)  # 2 number of max value
        #     # if only one label has largest votes
        #     if num_max[max(count_max)] == 1:
        #         pre_label = np.argmax(count_max)
        #         # print(f'[=1],label={np.argmax(count_max)}')
        #     # if more than one labels has largest, output the nearest label
        #     else:
        #         pre_label = labels_train[train_topk[0]].item()
        #         # print(f'[>1],label={labels_train[train_topk[0]].item()}')
        #     correct.append((pre_label == labels_test[i]).item())
        # correct = np.sum(correct)
        # total = len(labels_test)
        # result = 100. * correct / total
        acc = knn_acc(embeddings_train, labels_train, embeddings_test, labels_test)
        return acc

    # VALIDATION FOR POISONED DATA
    def new_val_pgd(self, model, train_loader, test_loader, attack, device: torch.device):
        model.eval()
        ### calcalate train embedding ##########################
        embeddings_train, labels_train = get_embeddings_from_dataloader(train_loader, model, device)
        total_len = len(embeddings_train)
        index = random.sample(range(0, total_len), 5000)
        embeddings_train, labels_train = embeddings_train[index], labels_train[index]
        # print('DONE calcalate train embedding')

        test_acc_pgd = 0.0
        for i, (images, labels) in enumerate(test_loader):
            # print('max: {}, min: {}'.format(torch.max(images), torch.min(images)))
            ### Predict Test and generate poisoned Test ##########################
            emb_test, label_test = get_embeddings_from_images(images, labels, model, device)
            pred_test = knn_pred(embeddings_train, labels_train, emb_test)
            # pred_test = torch.squeeze(pred_test, dim=1)
            pred_test = torch.tensor(pred_test)
            #todo: attack放进去的label应该是knn预测得出的 ———— done
            adv_images = attack(images, pred_test)

            ### calcalate pgd embedding ##########################
            embeddings_pgd, labels_pgd = get_embeddings_from_images(adv_images, pred_test, model, device)
            # print('Calculating knn')
            # correct = []
            # for k in range(embeddings_test.shape[0]):
            #     pdist = (embeddings_train - embeddings_test[k].view(1, embeddings_test.shape[1])).norm(dim=1)
            #     train_topk = pdist.topk(k=5, largest=False)[1]
            #
            #     nearest_n = []
            #     for label in train_topk:
            #         nearest_n.append(labels_train[label].item())
            #     count_max = np.bincount(nearest_n)  # count=[0 0 0 0 0 0 2 2]
            #     num_max = np.bincount(count_max)  # 2 最大值的个数
            #     if num_max[max(count_max)] == 1:
            #         pre_label = np.argmax(count_max)
            #     else:
            #         pre_label = labels_train[train_topk[0]].item()
            #     correct.append((pre_label == labels_test[k]).item())
            # correct = np.sum(correct)
            # total = len(labels_test)
            test_acc_pgd += knn_acc(embeddings_train, labels_train, embeddings_pgd, label_test)
        # print(f'i={i}')
        test_acc_pgd = test_acc_pgd / i
        return test_acc_pgd

    # POISON TRAINING DATA
    def poison_v2(self, x, y, x_adv, *, margin=0.5):
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        x_adv = x_adv.to(self.fc1.weight.device)

        output = self.forward(x, l2norm=False) # 获得模型预测值/输出
        output_adv = self.forward(x_adv, l2norm=False)
        Y = y.cpu().numpy()

        idxa, idxp, idxn, idxadv = [], [], [], []
        for i in range(10):  # 10 classes
            try:
                ida = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idp = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idn = np.random.choice(np.where(Y != i)[0],
                                       int(output.shape[0] / 10), replace=True)
                # idadv = np.random.choice(np.where(Y == i)[0],
                #                        int(output_adv.shape[0] / 10), replace=True)
                idxa.extend(list(ida))
                idxp.extend(list(idp))
                idxn.extend(list(idn))
                # idxadv.extend(list(idadv))
                idxadv.extend(list(ida)) # 0508 anchor和anchor_adv是同一个图像
            except ValueError:
                print(f'No labels = {i}')
                break

        XA = output[torch.LongTensor(idxa)]
        # XA_ori = XA.clone().detach()
        # XA_delta = XA_ori + delta * 2 * (0.5 - torch.rand(XA_ori.shape)).to(XA_ori.device)
        XAdv = output_adv[torch.LongTensor(idxadv)]
        XP = output[torch.LongTensor(idxp)]
        XN = output[torch.LongTensor(idxn)]

        loss_1 = F.triplet_margin_loss(XA, XP, XN, margin=margin, reduction='mean')
        loss_2 = F.triplet_margin_loss(XAdv, XN, XP, margin=margin, reduction='mean')
        loss = loss_1 + loss_2
        return output, loss


###############################################
####### NOT USE IN FINAL VERSION ##############


    # NOT USE
    def getloader_cifar(self, batchsize:int=1):
        print('datasets.cifar.load_cifar')
        return datasets.cifar.load_cifar(batchsize)

   # NOT USE
    def print_acc(self, epoch, iteration, total, output, labels, loss, top_k):
        M, K = output.shape[0], output.shape[1]
        pdist = (output.view(M, 1, K).expand(M, M, K) - output.view(1, M, K).expand(M, M, K)).norm(dim=2)
        #pdist.to(self.device)
        _, indices = pdist.topk(k=top_k, dim=1, largest=False)
        #indices.to(self.device)
        labels_test = labels.clone().detach()
        #labels_test.to(self.device)
        y_pred =[]
        for i in range(top_k):
            indices_at_k: torch.Tensor = indices[:, i]  # [M]
            #indices_at_k.to(labels_test.device)
            y_pred_at_k: torch.Tensor = labels_test[indices_at_k].unsqueeze(dim=1).cpu()  # [M x 1]
            y_pred.append(y_pred_at_k)

        y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
        labels_test = torch.hstack((labels_test,) * top_k)  # [M x k]

        n_predictions: int = y_pred.shape[0]
        n_true_predictions: int = ((y_pred == labels_test).sum(dim=1) > 0).sum().item()
        topk_accuracy: float = n_true_predictions / n_predictions * 100
        print(f'Eph[{epoch}][{iteration}/{total}]',
              collections.namedtuple('Result', ['loss', 'r_1'])(
                  '%.4f' % loss.item(), '%.3f' % topk_accuracy))

    # NOT USE
    def report_poison(self, epoch, iteration, total, output, labels, loss):
        M, K = output.shape[0], output.shape[1]
        pdist = (output.view(M,1,K).expand(M,M,K) - output.view(1,M,K).expand(M,M,K)).norm(dim=2)
        offdiagdist = pdist + 1e9*torch.eye(pdist.shape[0]).to(output.device)
        #knnsearch = labels[torch.min(offdiagdist, dim=1)[1]].cpu()
        knnsearch = labels[torch.min(offdiagdist, dim=1)[1]]
        acc = 100.*knnsearch.eq(labels.cpu()).sum().item() / len(labels)
        print(f'Eph[{epoch}][{iteration}/{total}]',
                collections.namedtuple('poison_Result', ['loss', 'r_1'])(
                    '%.4f'%loss.item(), '%.3f'%acc))

    # NOT USE
    def poison(self, x, y, *, margin=0.5, delta=0.1):
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        output = self.forward(x, l2norm=False)
        output_orig = output.clone().detach()
        # print('output.shape:', output.shape)
        #100x500
        # print('output.shape[0]:', output.shape[0])

        #idxp=sam class,idxn,diff class
        Y = y.cpu().numpy()
        idxa, idxp, idxn = [], [], []
        # targeted = [6, 7, 7, 5, 9, 3, 5, 9, 2, 4]
        for i in range(10):  # 10 classes
            try:
                ida = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idp = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idn = np.random.choice(np.where(Y != i)[0],
                                       int(output.shape[0] / 10), replace=True)
                # idn = np.random.choice(np.where(Y == targeted[i])[0],
                #                        int(output.shape[0] / 10), replace=True)
                #print('ida:', ida, 'ida.shape:', ida.shape)
                idxa.extend(list(ida))
                idxp.extend(list(idp))
                idxn.extend(list(idn))
            except ValueError:
                print(f'No labels = {i}')
                break
            #print('idxa:', idxa,'idxa.len:', len(idxa))
        XA = output[torch.LongTensor(idxa)]
        #print('XA:', XA, 'XA.shape:', XA.shape)
        #XA:100X500
        XA_ori = XA.clone().detach()
        #images = images + delta * 2 * (0.5 - torch.rand(images.shape)).to(images.device)
        #XA_delta = XA_ori + delta * 2 * (0.5 - torch.rand(XA_ori.shape)).to(XA_ori.device)
        # 为什么要在output了之后加，不能在images的时候加？
        XA_delta = XA_ori + delta * 2 * (0.5 - torch.rand(XA_ori.shape)).to(XA_ori.device)
        #XA_delta = output[torch.LongTensor(XA_delta)]
        #XA_delta.to(self.device)
        XP = output[torch.LongTensor(idxp)]
        XN = output[torch.LongTensor(idxn)]
        #loss = torch.nn.TripletMarginLoss(XA, XP, XN, margin=margin, reduction='mean')
        loss_1 = F.triplet_margin_loss(XA, XP, XN, margin=margin, reduction='mean')
        loss_2 = F.triplet_margin_loss(XA_delta, XN, XP, margin=margin, reduction='mean')
        loss = loss_1 + loss_2
        return output, loss

    # NOT USE
    # train_v1在val的时候也加delta
    def new_poi_val(self, model, train_loader, test_loader, delta, device: torch.device):
        model.eval()
        # embeddings_test, labels_test = get_poi_emb_from_dataloader(test_loader, model, device)
        embeddings_train, labels_train = get_embeddings_from_dataloader(train_loader, model, device)
        total_len = len(embeddings_train)
        index = random.sample(range(0, total_len), 10000)
        embeddings_train, labels_train = embeddings_train[index], labels_train[index]
        correct = []

        test_acc = 0.0
        for i, (images, labels) in enumerate(test_loader):
            labels = torch.squeeze(labels, dim=1)
            ### calcalate noise test embedding
            embeddings_test, labels_test = get_poi_emb_from_images(images, labels, model, device, delta)

            # print('Calculating knn')
            correct = []
            for k in range(embeddings_test.shape[0]):
                pdist = (embeddings_train - embeddings_test[k].view(1, embeddings_test.shape[1])).norm(dim=1)
                train_topk = pdist.topk(k=5, largest=False)[1]

                nearest_n = []
                for label in train_topk:
                    nearest_n.append(labels_train[label].item())
                count_max = np.bincount(nearest_n)  # count=[0 0 0 0 0 0 2 2]
                num_max = np.bincount(count_max)  # 2 最大值的个数
                if num_max[max(count_max)] == 1:
                    pre_label = np.argmax(count_max)
                else:
                    pre_label = labels_train[train_topk[0]].item()

                correct.append((pre_label == labels_test[k]).item())
            correct = np.sum(correct)
            total = len(labels_test)
            test_acc += 100. * correct / total
        # print(f'i={i}')
        test_acc = test_acc / i
        return test_acc

    # NOT USE
    def poison_val(self, model, test_loader, device: torch.device):
        model.eval()
        #embeddings_test, labels_test = get_embeddings_from_dataloader(test_loader, model, device)
        embeddings_test, labels_test = get_poison_embeddings_from_dataloader(test_loader, model, device)

        correct = []
        for i in range(embeddings_test.shape[0]):
            pdist = (embeddings_test - embeddings_test[i].view(1, embeddings_test.shape[1])).norm(dim=1)
            knnsrch = labels_test[pdist.topk(k=2, largest=False)[1][1]]
            correct.append((knnsrch == labels_test[i]).item())
        correct = np.sum(correct)
        total = len(labels_test)
        #result = f'rank@1= {100. * correct / total} (/100)'
        result = 100. * correct / total
        return result

    def load_poison(self,  batchsize:int=1):
        x_ = np.load('./MNIST/poison/x_train_mnist_ntga_cnn_best.npy',
                          encoding='bytes')
        y_ = np.load('./MNIST/poison/y_train_mnist.npy', encoding='bytes')
        #print(f'original label shape:{y_.shape}')
        x = np.swapaxes(x_, 3, 1)
        x = np.swapaxes(x, 3, 2)
        y = np.where(y_ == np.max(y_))[1]
        y = np.expand_dims(y, axis=1) # (50000, )改成(50000, 1)
        #print(f'poison label shape:{y.shape}')
        # split train and validation
        data_len = len(x)
        split_len = int(0.9*data_len)
        # print(data_len, split_len)
        x_train = x[0:split_len,:]
        y_train = y[0:split_len]
        x_test = x[split_len:data_len, :]
        y_test = y[split_len:data_len]
        #data loader
        poison_train = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        loader_poison_train = torch.utils.data.DataLoader(poison_train, batch_size=batchsize,
                                                          shuffle=True, num_workers=2)
        poison_test = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        loader_poison_test = torch.utils.data.DataLoader(poison_test, batch_size=batchsize,
                                                          shuffle=True, num_workers=2)
        print( f'x_train{x_train.shape},  y_train{y_train.shape}')
        print( f'x_test {x_test.shape},  y_test {y_test.shape}')
        return loader_poison_train, loader_poison_test
        # if kind == 'train':
        #     x_train = np.load('./MNIST/poison/x_train_mnist_ntga_cnn_best.npy',
        #                       encoding='bytes')
        #     y_train = np.load('./MNIST/poison/y_train_mnist.npy', encoding='bytes')
        #     x_train = np.swapaxes(x_train, 3, 1)
        #     x_train = np.swapaxes(x_train, 3, 2)
        #     poison_train = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        #     loader_poison_train = torch.utils.data.DataLoader(poison_train, batch_size=batchsize,
        #                                               shuffle=True, num_workers=2)
        #     print(x_train.shape, y_train.shape)
        #     return loader_poison_train
        # else:
        #     x_test = np.load('./MNIST/poison/x_val_mnist.npy',encoding='bytes')
        #     y_test = np.load('./MNIST/poison/y_val_mnist.npy', encoding='bytes')
        #     x_test = np.swapaxes(x_test, 3, 1)
        #     x_test = np.swapaxes(x_test, 3, 2)
        #     poison_test = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        #     loader_poison_test = torch.utils.data.DataLoader(poison_test, batch_size=batchsize,
        #                                                       shuffle=True, num_workers=2)
        #     print(x_test.shape, y_test.shape)
        #     return loader_poison_test


    def val_poi_v2(self, loader, model, attack, device):
        adv_acc = 0.0
        for i, (images, labels) in enumerate(loader):
            labels = torch.squeeze(labels, dim=1)
            ## show the images
            # imshow(images, labels, 'clean')
            # test_acc = model.val_predict(model, images, labels, device=ag.device)
            # print(f'Clean accuracy: {test_acc}%')
            adv_images = attack(images, labels)
            # if i==0: imshow(adv_images, labels, 'metric_adv', False)
            adv_acc += model.val_predict(model, adv_images, labels, device=device)
        # print(f'Adversarial accuracy: {adv_acc/i}%')
        return (adv_acc/i)

    # TRAIN_V2
    def val_predict(self, model, images, labels, device: torch.device):
        model.eval()
        embeddings_test, labels_test = get_embeddings_from_images(images, labels, model, device)
        correct = []
        predict = []
        for i in range(embeddings_test.shape[0]):
            pdist = (embeddings_test - embeddings_test[i].view(1, embeddings_test.shape[1])).norm(dim=1)
            knnsrch = labels_test[pdist.topk(k=2, largest=False)[1][1]]
            correct.append((knnsrch == labels_test[i]).item())
            predict.append(knnsrch.item())
        # print(f'predict= {predict}')
        # print(f'correct= {correct}')
        correct = np.sum(correct)
        total = len(labels_test)
        # result = f'rank@1= {100. * correct / total} (/100)'
        result = 100. * correct / total
        return result

    def new_val_predict(self, model, train_loader, images, labels, device: torch.device):
        model.eval()
        embeddings_test, labels_test = get_embeddings_from_images(images, labels, model, device)
        embeddings_train, labels_train = get_embeddings_from_dataloader(train_loader, model, device)
        total_len = len(embeddings_train)
        index = random.sample(range(0, total_len), 10000)
        embeddings_train, labels_train = embeddings_train[index], labels_train[index]

        correct = []
        print('Calculating knn')
        for i in range(embeddings_test.shape[0]):
            pdist = (embeddings_train - embeddings_test[i].view(1, embeddings_test.shape[1])).norm(dim=1)
            train_topk = pdist.topk(k=5, largest=False)[1]

            nearest_n = []
            for label in train_topk:
                # print(f'train[{label.item()}] = {labels_train[label].item()}', end='   ')
                nearest_n.append(labels_train[label].item())
            count_max = np.bincount(nearest_n)  # count=[0 0 0 0 0 0 2 2]
            num_max = np.bincount(count_max)  # 2 最大值的个数
            if num_max[max(count_max)] == 1:
                pre_label = np.argmax(count_max)
            else:
                pre_label = labels_train[train_topk[0]].item()
            correct.append((pre_label == labels_test[i]).item())
        correct = np.sum(correct)
        total = len(labels_test)
        result = 100. * correct / total
        return result

    #  USE IN TRAIN_V2
    def loss_poi(self, x, y, x_poi, y_poi, *, margin1=0.5, margin2=1):
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        x_poi = x_poi.to(self.fc1.weight.device) #try2 注释 不能去掉
        y_poi = y_poi.to(self.fc1.weight.device).view(-1) #try1 注释
        output = self.forward(x, l2norm=False)
        #output_orig = output.clone().detach()
        output_poi = self.forward(x_poi, l2norm=False)
        Y = y.cpu().numpy()
        Y_poi = y_poi.cpu().numpy()

        idxa, idxp, idxn, idxa_poi = [], [], [], []
        for i in range(10):  # 10 classes
            try:
                ida = np.random.choice(np.where(Y == i)[0],int(output.shape[0] / 10), replace=True)
                idp = np.random.choice(np.where(Y == i)[0],int(output.shape[0] / 10), replace=True)
                idn = np.random.choice(np.where(Y != i)[0],int(output.shape[0] / 10), replace=True)
                ida_poi = np.random.choice(np.where(Y_poi == i)[0],int(output_poi.shape[0] / 10), replace=True)
                #print('ida:', ida, 'ida.shape:', ida.shape)
                idxa.extend(list(ida))
                idxp.extend(list(idp))
                idxn.extend(list(idn))
                idxa_poi.extend(list(ida_poi))
            except ValueError:
                print(f'No labels = {i}')
                break
            #print('idxa:', idxa,'idxa.len:', len(idxa))
        #print('XA:', XA, 'XA.shape:', XA.shape)
        #XA:100X500
        #XA_ori = XA.clone().detach()
        # 为什么要在output了之后加，不能在images的时候加？
        #XA_delta = XA_ori + delta * 2 * (0.5 - torch.rand(XA_ori.shape)).to(XA_ori.device)
        XA = output[torch.LongTensor(idxa)]
        XP = output[torch.LongTensor(idxp)]
        XN = output[torch.LongTensor(idxn)]
        # 不太确定这个output的作用
        XA_poi = output_poi[torch.LongTensor(idxa_poi)] #torch.Size([0, 500])
        #XA_poi = output[torch.LongTensor(idxa_poi)]
        #print(XA_poi.shape,XN.shape)
            #torch.Size([0, 500]) torch.Size([60, 500])
        #loss = torch.nn.TripletMarginLoss(XA, XP, XN, margin=margin, reduction='mean')
        loss_1 = F.triplet_margin_loss(XA, XP, XN, margin=margin1, reduction='mean')
        # print(len(XA_poi),len(XN),len(XP)) # 120 120 120 出错是60 120 120
        #loss_2 = F.triplet_margin_loss(XA_poi, XN, XP, margin=margin2, reduction='mean')
        loss_2 = F.triplet_margin_loss(XA_poi, XN, XA, margin=margin2, reduction='mean') # 0419 try
        #print(f'loss1={loss_1}, loss2={loss_2}', end='   ')
        loss =  loss_1 + loss_2
        return output, loss

    def print_image(self,image, delta=0.1):
        image = image.to(self.fc1.weight.device)
        print(f'\n>>> image {image.shape}')
        print(image)
        output = self.forward(image, l2norm=False)
        print(f'\n>>> output  {output.shape}')
        print(output)
        output_delta = output + delta * 2 * (0.5 - torch.rand(output.shape)).to(output.device)
        print(f'\n>>> output_delta  {output_delta.shape}')
        print(output_delta)


'''
    def poison_test(self, testset):
        i = 0
        labels_original_images = torch.tensor(np.zeros(0, dtype=np.int64))
        labels_pertubed_images = torch.tensor(np.zeros(0, dtype=np.int64))
        for batch_index, (inputs, _) in enumerate(testset):
            i += inputs.shape[0]
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs, l2norm=False)
            _, predicted = outputs.max(1)
            labels_original_images = torch.cat((labels_original_images, predicted.cpu()))
        torch.cuda.empty_cache()
        correct = 0
        # Finding labels for perturbed images
        for batch_index, (inputs, labels) in enumerate(testset):
            inputs = inputs.to(self.device)
            # v = v.to(device)
            # inputs += transformer(v).float().to(device)
            # outputs = net(inputs)
            _, predicted = outputs.max(1)
            labels_pertubed_images = torch.cat((labels_pertubed_images, predicted.cpu()))
            correct += (predicted.cpu() == labels.cpu()).sum().item()
        torch.cuda.empty_cache()

    


    def poison_lo(self,  x, y, *, margin=0.5, delta=0.1):
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        output = self.forward(x, l2norm=False)
        images_ori = x.clone().detach()
        #x.requires_grad = True

        Y = y.cpu().numpy()
        idxa, idxp, idxn = [], [], []

        # first forword
        with torch.no_grad():
            # output, loss_ori = self.loss(x, y)
            for i in range(10):  # 10 classes
                ida = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idp = np.random.choice(np.where(Y == i)[0],
                                       int(output.shape[0] / 10), replace=True)
                idn = np.random.choice(np.where(Y != i)[0],
                                       int(output.shape[0] / 10), replace=True)
                # print('ida:', ida, 'ida.shape:', ida.shape)
                idxa.extend(list(ida))
                idxp.extend(list(idp))
                idxn.extend(list(idn))
                XA = output[torch.LongTensor(idxa)]
                # print('XA:', XA, 'XA.shape:', XA.shape)
                # XA:100X500
                XP = output[torch.LongTensor(idxp)]
                XN = output[torch.LongTensor(idxn)]

                # loss = torch.nn.TripletMarginLoss(XA, XP, XN, margin=margin, reduction='mean')
                loss_1 = F.triplet_margin_loss(XA, XP, XN, margin=margin, reduction='mean')
            output_ori = output.clone().detach()

        # start poison
        XA_delta = XA + delta * 2 * (0.5 - torch.rand(XA.shape)).to(XA.device)
        XA_delta.to(self.device)
        XA_delta = output[torch.LongTensor(XA_delta)]
        loss_2 = F.triplet_margin_loss(XA_delta, XN, XP, margin=margin, reduction='mean')
        loss = loss_1 + loss_2
        return output_ori, loss_1, output, loss

    def poison_demo1(self, x, y, *, margin=0.5, delta=0.1):
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        output = self.forward(x, l2norm=False)

        # idxp=sam class,idxn,diff class
        Y = y.cpu().numpy()
        idxa, idxp, idxn = [], [], []
        for i in range(10):  # 10 classes
            ida = np.random.choice(np.where(Y == i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idp = np.random.choice(np.where(Y == i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idn = np.random.choice(np.where(Y != i)[0],
                                   int(output.shape[0] / 10), replace=True)
            # print('ida:', ida, 'ida.shape:', ida.shape)
            idxa.extend(list(ida))
            idxp.extend(list(idp))
            idxn.extend(list(idn))
            print(ida.dtype)
            # print('idxa:', idxa,'idxa.len:', len(idxa))
        XA = output[torch.LongTensor(idxa)]
        # print('XA:', XA, 'XA.shape:', XA.shape)
        # XA:100X500

        XP = output[torch.LongTensor(idxp)]
        XN = output[torch.LongTensor(idxn)]
        XA_ori = XA.clone().detach()
        # images = images + delta * 2 * (0.5 - torch.rand(images.shape)).to(images.device)
        XA_delta = XA_ori + delta * 2 * (0.5 - torch.rand(XA_ori.shape)).to(XA_ori.device)
        #XA_delta = XA + delta * 2 * (0.5 - torch.rand(XA.shape)).to(XA.device)
        # XA_delta.numpy()
        #XA_delta.to(self.device)
        XA_delta.int()
        XA_delta = output[torch.LongTensor(XA_delta.cpu())]

        # loss = torch.nn.TripletMarginLoss(XA, XP, XN, margin=margin, reduction='mean')
        loss_1 = F.triplet_margin_loss(XA, XP, XN, margin=margin, reduction='mean')
        loss_2 = F.triplet_margin_loss(XA_delta, XN, XP, margin=margin, reduction='mean')
        loss = loss_1 + loss_2
        return output, loss





    def poison_loss(self, x, y, *, margin=0.5, delta=0.1):
        images = x.clone().detach().to(self.device)
        images_orig = images.clone().detach()
        images.requires_grad = True
        labels = y.to(self.device).view(-1)

        #step1:
        self.eval()
        with torch.no_grad():
            output = self.forward(x, l2norm=False)
            output_orig = output.clone().detach()

        #step2:attack
        images = images + delta * 2 * (0.5 - torch.rand(images.shape)).to(images.device)
        images = torch.clamp(images, min=0., max=1.)
        images = images.detach()
        images.requires_grad = True
        self.train()
        output = self.forward(x, l2norm=False)
        distance  = F.pairwise_distance(output, output_orig, p=2)
        loss = -distance.sum()



        # images = x.to(self.fc1.weight.device)
        # labels = y.to(self.fc1.weight.device).view(-1)
        # images_orig = images.clone().detach()
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        output = self.forward(x, l2norm=False)
        output_orig = output.clone().detach()


        #idxp=sam class,idxn,diff class
        Y = y.cpu().numpy()
        idxa, idxp, idxn = [], [], []
        for i in range(10):  # 10 classes
            ida = np.random.choice(np.where(Y == i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idp = np.random.choice(np.where(Y == i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idn = np.random.choice(np.where(Y != i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idxa.extend(list(ida))
            idxp.extend(list(idp))
            idxn.extend(list(idn))
        XA = output[torch.LongTensor(idxa)]
        XP = output[torch.LongTensor(idxp)]
        XN = output[torch.LongTensor(idxn)]
        #loss = torch.nn.TripletMarginLoss(XA, XP, XN, margin=margin, reduction='mean')
        loss_1 = F.triplet_margin_loss(XA, XP, XN, margin=margin, reduction='mean')
        return output, loss_1



    def loss_adversary(self, x, y, delta=0.1, margin=0.5):
        x = x.to(self.fc1.weight.device)
        y = y.to(self.fc1.weight.device).view(-1)
        output = self.forward(x, l2norm=False)

        #idxp=sam class,idxn,diff class
        Y = y.cpu().numpy()
        idxa, idxp, idxn = [], [], []
        for i in range(10):  # 10 classes
            ida = np.random.choice(np.where(Y == i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idp = np.random.choice(np.where(Y == i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idn = np.random.choice(np.where(Y != i)[0],
                                   int(output.shape[0] / 10), replace=True)
            idxa.extend(list(ida))
            idxp.extend(list(idp))
            idxn.extend(list(idn))
        XA = output[torch.LongTensor(idxa)]
        XP = output[torch.LongTensor(idxp)]
        XN = output[torch.LongTensor(idxn)]
        #loss = torch.nn.TripletMarginLoss(XA, XP, XN, margin=margin, reduction='mean')
        loss = F.triplet_margin_loss(XA, XP, XN, margin=margin, reduction='mean')
        return output, loss
'''

