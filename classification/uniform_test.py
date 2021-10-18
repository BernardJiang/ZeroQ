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

import argparse
import torch
import numpy as np
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import *
from distill_data import *
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes, TrainingMode


# model settings
def arg_parse():
    parser = argparse.ArgumentParser(
        description='This repository contains the PyTorch implementation for the paper ZeroQ: A Novel Zero-Shot Quantization Framework.')
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        choices=['imagenet', 'cifar10'],
                        help='type of dataset')
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet50', 'inceptionv3',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'sqnxt23_w2'
                        ],
                        help='model to be quantized')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    sys.path.insert(0, '/workspace/develop/ZeroQ/classification')
    os.chdir("/workspace/develop/ZeroQ/classification")
    result_folder='./data/img_dir/'

    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Load pretrained model
    model = ptcv_get_model(args.model, pretrained=True)
    print('****** Full precision model loaded ******')
    
    #save the model.
    torch.save(model.state_dict(), result_folder+'mobilenetv2.pt',_use_new_zipfile_serialization=False)
    try:
        import onnx
        onnx_export_file = result_folder+'mobilenetv2_zeroq.onnx'
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        print('****onnx file****',onnx_export_file)
        model.eval()
        img = torch.zeros((1, 3, 224, 224))
        y = model(img)  # dry run
        # torch.onnx.export(  output_names=['classes', 'boxes'] if y is None else ['output'])
        torch.onnx.export(model,               # model being run
                            img,                         # model input (or a tuple for multiple inputs)
                            onnx_export_file,   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=11,          # the ONNX version to export the model to
                            do_constant_folding=False,  # whether to execute constant folding for optimization
                            input_names = ['images'],   # the model's input names
                            output_names = ['classes', 'boxes'] if y is None else ['output'], # the model's output names
                            training=TrainingMode.PRESERVE,
                            keep_initializers_as_inputs=True,
                            verbose=False
        )        # Checks
        onnx_model = onnx.load(onnx_export_file)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % onnx_export_file)
    except Exception as e:
        print('ONNX export failure: %s' % e)
    
    # Load validation data
    test_loader = getTestData(args.dataset,
                              batch_size=args.test_batch_size,
                              path='./data/imagenet/',
                              for_inception=args.model.startswith('inception'))

    # # save 1000 true images.
    # for batch_idx, (inputs, targets) in enumerate(test_loader):
    #     x1 = inputs.cpu().numpy()
    #     x2 = np.moveaxis(x1, 1, -1)  #move from cxhxw to hxwxc
    #     for j in range(x2.shape[0]):
    #         x3 = np.reshape(inputs[j], (-1))
    #         img_path = os.path.join(result_folder+'/trueimages', "IMG{:04d}.txt".format(targets[j]))
    #         np.savetxt(img_path, x3, delimiter=",", fmt='%f')

    # Generate distilled data
    # dataloader = getDistilData(
    #     model.cuda(),
    #     args.dataset,
    #     batch_size=args.batch_size,
    #     for_inception=args.model.startswith('inception'))
    # print('****** Data loaded ******')

    # save all distilled data
    # for idx, x in enumerate(dataloader):
    #     x1 = x.cpu().numpy()
    #     x2 = np.moveaxis(x1, 1, -1)  #move from cxhxw to hxwxc
    #     # x3 = np.reshape(x2, (-1))
    #     i = 0
    #     for i in range(x2.shape[0]):
    #         x3 = np.reshape(x2[i], (-1))
    #         img_path = os.path.join(result_folder+'zeroqdata', "IMG{:04d}.txt".format(i))
    #         np.savetxt(img_path, x3, delimiter=",", fmt='%f')
            


    # Quantize single-precision model to 8-bit model
    quantized_model = model # quantize_model(model)
    # Freeze BatchNorm statistics
    quantized_model.eval()
    quantized_model = quantized_model.cuda()

    # print("Test model without distilled data")
    # model_0 = copy.deepcopy(quantized_model)
    # freeze_model(model_0)
    # model_0 = nn.DataParallel(model_0).cuda()
    # test(model_0, test_loader)

    # Update activation range according to distilled data
    # update(quantized_model, dataloader)
    # print('****** Zero Shot Quantization Finished ******')
    
    # print_model_range(quantized_model)

    # Freeze activation range during test
    # freeze_model(quantized_model)
    quantized_model = nn.DataParallel(quantized_model).cuda()

    # Test the final quantized model
    test(quantized_model, test_loader)

    torch.save(model.state_dict(), result_folder + 'mobilenetv2_quan.pt', _use_new_zipfile_serialization=False)


