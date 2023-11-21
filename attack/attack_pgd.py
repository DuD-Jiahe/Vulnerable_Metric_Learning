"""
# -*-coding:utf-8-*- 
Theme: modification of pgd from torchattacks
Author: JiaHe Du
Date: 2023.04.25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attack.attack_class import  Attack

class PGD_triplet(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # 暂时用不到
        # if self.targeted:
        #     target_labels = self.get_target_label(images, labels)
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            output = self.get_logits(adv_images)
            # generate triple
            Y = labels.cpu().numpy()
            idxa, idxp, idxn = [], [], []
            for i in range(10):  # 10 classes
                try:
                    ida = np.random.choice(np.where(Y == i)[0],int(output.shape[0] / 10), replace=True)
                    idp = np.random.choice(np.where(Y == i)[0],int(output.shape[0] / 10), replace=True)
                    idn = np.random.choice(np.where(Y != i)[0],int(output.shape[0] / 10), replace=True)
                    idxa.extend(list(ida))
                    idxp.extend(list(idp))
                    idxn.extend(list(idn))
                except ValueError:
                    print(f'No labels = {i}')
                    continue
            XA = output[torch.LongTensor(idxa)]
            XP = output[torch.LongTensor(idxp)]
            XN = output[torch.LongTensor(idxn)]

            cost = F.triplet_margin_loss(XA, XP, XN, margin=0.5, reduction='mean')
            # Calculate loss
            # 暂时用不到
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            # cost对adv_images求导

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

