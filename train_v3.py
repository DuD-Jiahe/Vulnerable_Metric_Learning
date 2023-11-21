"""
# -*-coding:utf-8-*- 
Theme: PGD with triplet loss
Author: JiaHe Du
Date: 2023.04.25
"""

import sys, os, yaml
import matplotlib.pyplot as plt
import numpy as np
import torch as torch
import argparse
import models
from models import lenet_triplet
from models.dbg import _fgcG, _bgcV
from attack.attack_pgd import PGD_triplet
import time
import torchvision
from utils import knn_pred


### TRAIN THE POISONED MODEL
def Train_v3_poison(argv, is_trian):
    print('>>> BUILDING POISONED MODEL\n')
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
            type=str, help='computational device')
    ag.add_argument('-M', '--model', type=str, default='lenet_triplet', required=False)
    ag.add_argument('--train', default=is_trian) # Train the model or not
    ag.add_argument('-A', '--attack', type=float, default=True, required=False) # default = None or True
    ag.add_argument('--overfit', action='store_true')
    ag.add_argument('--report', type=int, default=20) # output printed interval in training
    ag.add_argument('--validate', action='store_true')
    ag = ag.parse_args(argv)


    print('>>> Parsing arguments')
    for x in yaml.dump(vars(ag)).split('\n'): print(_fgcG(x))
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)


    print('\n>>> Setting up model and optimizer')
    model = lenet_triplet
    model = model.lenet_model().to(device=ag.device)
    optim = torch.optim.Adam(model.parameters(),
                             lr=config[ag.model]['lr'], weight_decay=1e-7)
    print(model)
    print(optim)

    print('\n>>> Loading datasets')
    loader_train_1 = model.getloader('train', config[ag.model]['batchsize_v2'])
    loader_test = model.getloader('test', config[ag.model]['batchsize_v2'])
    loader_train, loader_val = model.split_train_val(loader_train_1, config[ag.model]['batchsize_v2'], 0.9)
    print(f'Train:{len(loader_train.dataset)}, Test:{len(loader_test.dataset)}, Val:{len(loader_val.dataset)}')


    print('\n>>> Start training')
    test_acc = model.val(model, loader_test, device=ag.device)
    print(f'Validate[-1] {test_acc}')

    val_top1 = [0]
    # test_top1_ntga = [0]
    val_top1_pgd = [0]
    total_epoch = [0]
    ### Training Model #################################################

    if ag.train is True:

        for epoch in range(config[ag.model]['epoch_v3']):
            ### Training Stage #################################################
            for iteration, (images, labels) in enumerate(loader_train):
                # print(f'\n>>> PGD attack for training[{epoch}]')
                ### PGD attack -> poisoned data ################################
                atk_train = PGD_triplet(model, eps=EPS, alpha=ALPHA, steps=STEPS, random_start=True)
                # if iteration==0: print(atk_train)
                labels_atk = torch.squeeze(labels, dim=1) ### 这里可以直接用Training Label
                adv_images = atk_train(images, labels_atk)

                model.train() # 0425 在生成PGD毒数据之后再用train？
                ### poisoned triplet loss ################################
                output, loss = model.poison_v2(images, labels, adv_images)

                optim.zero_grad()
                with torch.no_grad():
                    loss.backward()
                    optim.step()

                if (iteration % ag.report == 0) or ag.overfit:
                    model.report(epoch, iteration, len(loader_train), output, labels, loss)
                if ag.overfit:
                        break

            ### MODEL VALIDATION
            ### clean validation ################################
            # see the model performance on clean data
            val_acc = model.new_val(model, loader_train, loader_val, device=ag.device)
            print(f'Clean Validate[{epoch}]{val_acc}%')

            ### PGD validation ##############
            # see the model performance on poison data
            # print(f'\n>>> PGD attack for validation')
            atk = PGD_triplet(model, eps=EPS, alpha=ALPHA, steps=STEPS, random_start=True)
            val_acc_pgd = model.new_val_pgd(model, loader_train, loader_val, atk, ag.device)
            print(f'Poison Validate[{epoch}] {val_acc_pgd}%')

            val_top1.append(val_acc)
            val_top1_pgd.append(val_acc_pgd)
            total_epoch.append(epoch)

            ### SAVE MODEL
            print('>>> Saving the network to:', 'trained/' + ag.model + '_poisoned_v3.pt')
            torch.save(model.state_dict(), 'trained/' + ag.model + '_poisoned_v3.pt')

    ### Loading Model #################################################
    else:
        sdpath = 'trained/' + ag.model + '_poisoned_v3.pt'
        print('\n>>> Loading clean model from', sdpath)
        model = getattr(models, ag.model).lenet_model().to(ag.device)
        if (torch.cuda.is_available() is False):
            model.load_state_dict(torch.load(sdpath, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(sdpath))
        print(model)

    ### MODEL TESTING
    model.eval()
    ### clean testing ###########################################
    print('\n >>>>> TESTING ')
    test_acc = model.new_val(model, loader_train, loader_test, device=ag.device)
    print(f'>>>Clean testing: {test_acc}%')
    ### PGD testing ##############################################
    print(f'>>> PGD attack for testing')
    atk_test = PGD_triplet(model, eps=EPS, alpha=ALPHA, steps=STEPS, random_start=True)
    print(atk_test)
    test_acc_pgd = model.new_val_pgd(model, loader_train, loader_test, atk_test, ag.device)
    print(f'>>>Poison testing: {test_acc_pgd:.2f}%')

    ### PLOT ##############################################
    if ag.train is True and config[ag.model]['epoch_v3']>1:
        plt.xlabel('epoch') #TODO 移到 train is True里面 ———— done
        plt.ylabel('rank-1')
        plt.plot(total_epoch, val_top1, label='Clean', color='#1f77b4')
        plt.plot(total_epoch, val_top1_pgd, label='PGD', color='#d62728')
        # plt.plot(total_epoch, test_top1_ntga, label='ntga', color='#17becf')
        plt.legend(loc=4)
        plt.title(f'V3_Attack=True, PGD_para: eps={EPS:.3f},alpha={ALPHA:.3f},steps={STEPS}' , loc='center')
        plt.savefig('plot/rank_poison_v3.png')
        plt.show()

### TRAIN THE CLEAN MODEL
def Train_v3_clean(argv, is_train):
    print('>>> BUILDING CLEAN MODEL\n')
    ag = argparse.ArgumentParser()
    ag.add_argument('-D', '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
            type=str, help='computational device')
    ag.add_argument('-M', '--model', type=str, default='lenet_triplet', required=False)
    ag.add_argument('--train', default=is_train)
    ag.add_argument('--overfit', action='store_true')
    ag.add_argument('--report', type=int, default=20)
    ag.add_argument('--validate', action='store_true')
    ag = ag.parse_args(argv)


    print('>>> Parsing arguments')
    for x in yaml.dump(vars(ag)).split('\n'): print(_fgcG(x))
    config = yaml.load(open('config.yml', 'r').read(), Loader=yaml.SafeLoader)


    print('>>> Setting up model and optimizer')
    model = lenet_triplet
    model = model.lenet_model().to(device=ag.device)
    optim = torch.optim.Adam(model.parameters(),lr=config[ag.model]['lr'], weight_decay=1e-7)
    print(model)
    print(optim)


    print('\n>>> Loading datasets')
    loader_train_1 = model.getloader('train', config[ag.model]['batchsize_v2'])
    loader_test = model.getloader('test', config[ag.model]['batchsize_v2'])
    loader_train, loader_val = model.split_train_val(loader_train_1, config[ag.model]['batchsize_v2'], 0.9)
    print(f'Train:{len(loader_train.dataset)}, Test:{len(loader_test.dataset)}, Val:{len(loader_val.dataset)}')


    val_top1 = [0]
    val_top1_pgd = [0]
    total_epoch = [0]
    ### Training Model #################################################
    if ag.train is True:
        print('\n>>> Start training')
        test_acc = model.val(model, loader_test, device=ag.device)
        print(f'Validate[-1] {test_acc}')

        for epoch in range(config[ag.model]['epoch_v3']):
            ### Training Stage #################################################
            for iteration, (images, labels) in enumerate(loader_train):
                model.train()
                # clean loss
                output, loss = model.loss(images, labels)

                optim.zero_grad()
                with torch.no_grad():
                    loss.backward()
                    #loss_clean.backward()
                    optim.step()

                if (iteration % ag.report == 0) or ag.overfit:
                    model.report(epoch, iteration, len(loader_train), output, labels, loss)
                if ag.overfit:
                        break

            ### clean validation ################################
            val_acc = model.new_val(model, loader_train, loader_val, device=ag.device)
            print(f'Clean Validate[{epoch}]{val_acc}%')

            ### PGD validation ##################################
            # print(f'\n>>> PGD attack for validation')
            atk = PGD_triplet(model, eps=EPS, alpha=ALPHA, steps=STEPS, random_start=True)
            val_acc_pgd = model.new_val_pgd(model, loader_train, loader_val, atk, ag.device)
            print(f'Poison Validate[{epoch}] {val_acc_pgd}%')

            val_top1.append(val_acc)
            # test_top1_ntga.append(test_acc_ntga)
            val_top1_pgd.append(val_acc_pgd)
            total_epoch.append(epoch)

            print('>>> Saving the network to:', 'trained/' + ag.model + '_clean_v3.pt')
            torch.save(model.state_dict(), 'trained/' + ag.model + '_clean_v3.pt')
    ### Loading Model #################################################
    else:
        sdpath = 'trained/' + ag.model + '_clean_v3.pt'
        print('\n>>> Loading clean model from', sdpath)
        model = getattr(models, ag.model).lenet_model().to(ag.device)
        if (torch.cuda.is_available() is False):
            model.load_state_dict(torch.load(sdpath, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(sdpath))
        print(model)

    model.eval()
    ### clean testing ##################################
    print('\n >>>>> TESTING ')
    test_acc = model.new_val(model, loader_train, loader_test, device=ag.device)
    print(f'>>>Clean testing: {test_acc}%')
    ### PGD testing ##################################
    print(f'>>> PGD attack for testing')
    atk_test = PGD_triplet(model, eps=EPS, alpha=ALPHA, steps=STEPS, random_start=True)
    print(atk_test)
    test_acc_pgd = model.new_val_pgd(model, loader_train, loader_test, atk_test, ag.device)
    print(f'>>>Poison testing: {test_acc_pgd:.2f}%')
    #

    ### plot validation and epoch ##############
    if ag.train is True and config[ag.model]['epoch_v3']>1:
        plt.xlabel('epoch')
        plt.ylabel('rank-1')
        plt.plot(total_epoch, val_top1, label='Clean', color='#1f77b4')
        # plt.axhline(y=test_acc_pgd, label='Poison', color='#e377c2')
        plt.plot(total_epoch, val_top1_pgd, label='pgd', color='#d62728')
        plt.legend(loc=4)
        plt.title(f'V2_Attack=None, PGD_para: eps={EPS:.3f},alpha={ALPHA:.3f},steps={STEPS}', loc='center')
        plt.savefig('plot/rank_clean_v2.png')
        plt.show()

EPS = 8 / 255
ALPHA = 2 / 225
STEPS = 10

if __name__ == '__main__':
    start = time.time()

    is_attack = False
    is_train = True
    print(f'is_attack={is_attack} \nis_train={is_train}')
    if is_attack:
        Train_v3_poison(sys.argv[1:], is_train)
    else:
        Train_v3_clean(sys.argv[1:], is_train)

    end = time.time()
    runTime = str((end - start)/60)
    print('\n>>> Running Time:', runTime, 'min')