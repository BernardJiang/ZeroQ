#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from utils import *
from torch.utils.data import Dataset, DataLoader

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from PIL import Image
from numpy import asarray


def own_loss(A, B):
    """
	L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
	"""
    return (A - B).norm()**2 / B.size(0)
    # return ((B - A -1e-3)/(A+1e-3)).norm()**2


class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


def getDistilData(teacher_model,
                  dataset,
                  batch_size,
                  num_batch=1,
                  for_inception=False):
    sumwriter = SummaryWriter()
    """
	Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
	Currently only support a single GPU.

	teacher_model: pretrained single-precision model
	dataset: the name of the dataset
	batch_size: the batch size of generated distilled data
	num_batch: the number of batch of generated distilled data
	for_inception: whether the data is for Inception because inception has input size 299 rather than 224
	"""

    # initialize distilled data with random noise according to the dataset
    dataloader = getRandomData(dataset=dataset,
                               batch_size=batch_size,
                               for_inception=for_inception)

    images = next(iter(dataloader))
    i3 = images + 0.5
    grid = torchvision.utils.make_grid(i3)
    sumwriter.add_image('images_before', grid, 0)
    
    std, mean = torch.std_mean(images, unbiased=False)
    print("Before training min/max= {} {} std/mean {} {} ".format(torch.min(images), torch.max(images), std, mean))

    eps = 1e-6
    # initialize hooks and single-precision model
    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    # get number of BatchNorm layers in the model
    layers = sum([
        1 if isinstance(layer, nn.BatchNorm2d) else 0
        for layer in teacher_model.modules()
    ])

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.Conv2d) and len(hook_handles) < layers:
            # register hooks on the convolutional layers to get the intermediate output after convolution and before BatchNorm.
            hook = output_hook()
            hooks.append(hook)
            hook_handles.append(m.register_forward_hook(hook.hook))
        if isinstance(m, nn.BatchNorm2d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var +
                            eps).detach().clone().flatten().cuda()))
    assert len(hooks) == len(bn_stats)

    target = torch.zeros(64, 1000, dtype=torch.long)
    b = torch.tensor([*range(64)]).reshape(-1,1)
    target.scatter_(1, b, value=1)
    target = target.cuda()
    for i, gaussian_data in enumerate(dataloader):
        if i == num_batch:
            break
        # initialize the criterion, optimizer, and scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.075, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         factor=0.95,
                                                         verbose=False,
                                                         patience=10)

        input_mean = torch.zeros(1, 3).cuda()
        input_std = torch.ones(1, 3).cuda()

        for it in range(5000):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            target_loss = crit(output, torch.max(target,1)[1]) * 10000
            mean_loss = 0
            std_loss = 0
            min_loss = 0
            max_loss = 0

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                tmp_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                # bn_min = bn_mean - 3 * bn_std
                # bn_max = bn_mean + 3 * bn_std

                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1),
                                      dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
                # tmp_min = tmp_mean - 3 * tmp_std
                # tmp_max = tmp_mean + 3 * tmp_std

                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
                # min_loss += own_loss(bn_min, tmp_min)
                # max_loss += own_loss(bn_max, tmp_max)

            tmp_mean = torch.mean(gaussian_data.view(gaussian_data.size(0), 3,
                                                     -1),
                                  dim=2)
            tmp_std = torch.sqrt(
                torch.var(gaussian_data.view(gaussian_data.size(0), 3, -1),
                          dim=2) + eps)
            # tmp_min = tmp_mean - 3 * tmp_std
            # tmp_max = tmp_mean + 3 * tmp_std
            mean_loss += own_loss(input_mean, tmp_mean)
            std_loss += own_loss(input_std, tmp_std)
            # min_loss += own_loss(bn_min, tmp_min)
            # max_loss += own_loss(bn_max, tmp_max)

            subtotal_loss = mean_loss + std_loss
            total_loss = subtotal_loss + target_loss
            if i==0:
                if it % 100 == 0:
                    # lr = scheduler.get_last_lr()[0] # latest pytorch 1.5+ uses get_last_lr,  previously it was get_lr iirc;
                    lr1 = optimizer.param_groups[0]["lr"] # either the above line or this, both should do the same thing
                    std, mean = torch.std_mean(gaussian_data, unbiased=False)
                    gf = gaussian_data.flatten()
                    k = int(gf.nelement() * 0.005)
                    if k==0:
                        k = 1
                    ka, _ = torch.topk(gf, k, largest=False)
                    min2 = ka[-1]
                    ka, _ = torch.topk(gf, k, largest=True)
                    max2 = ka[-1]
                    print("it {} total_loss {:.2f} {:.2f} mean_loss {:.2f} std_loss {:.2f}, target_loss {:.2f}. min_min2/max_2_max= {:.2f} {:.2f} {:.2f} {:.2f} std/mean {:.2f} {:.2f} .LR {:.5f}".format(it, subtotal_loss, total_loss, mean_loss, std_loss, target_loss, torch.min(gaussian_data), min2, max2, torch.max(gaussian_data), std, mean, lr1))

                    sumwriter.add_scalars('epoch', {'total_loss': total_loss,
                                                    'mean_loss': mean_loss,
                                                    'std_loss': std_loss}, it)
                    sumwriter.add_histogram("epoch {}".format(it), gaussian_data, bins='auto')


            # update the distilled data
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        tensor = gaussian_data.detach().clone()
        refined_gaussian.append(tensor)

    for handle in hook_handles:
        handle.remove()

    tensor2 = tensor.cpu()    
    i3 = tensor2 + 0.5
    grid = torchvision.utils.make_grid(i3)
    sumwriter.add_image('images_after', grid, 0)
    std, mean = torch.std_mean(tensor2, unbiased=False)
    print("after training {} min/max= {} {} std/mean {} {} ".format(it, torch.min(tensor2), torch.max(tensor2), std, mean))

    sumwriter.close()
    return refined_gaussian
