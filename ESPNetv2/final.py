import torch
import sys
sys.path.append('F:/My_Stuff/GP/Last Mile Delivery/Semantic Segmentation/EdgeNets - Copy')
import os
from PIL import Image
import glob
from torchvision.transforms import functional as F
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser


def main(args):

    if args.model == 'espnetv2':
        from model.segmentation.espnetv2 import espnetv2_seg
        args.classes = 20
        model = espnetv2_seg(args)


    if args.weights_test:
        print('Loading model weights')
        weight_dict = torch.load(args.weights_test, map_location=torch.device('cpu'))
        model.load_state_dict(weight_dict)
        print('Weight loaded successfully')
    else:
        print("ERRORRRR")
    #model = model.cuda()
    #input = torch.Tensor(1, 3, 1024, 512)
    #out = model(input.cuda())
    #print(out.shape)
    x,y = process_img("data/000000_10.png",[512,1024],'cuda',model.cuda())
    print(y.shape)
    cv2.imshow("image2d",x)
    cv2.imshow("image", y)
    cv2.waitKey(0)



if __name__ == '__main__':
    segmentation_datasets = ['pascal', 'city']
    segmentation_models = ['espnetv2', 'dicenet']

    parser = ArgumentParser()
    # mdoel details
    parser.add_argument('--model', default="espnetv2", choices=segmentation_models, help='Model name')
    parser.add_argument('--weights-test', default='', help='Pretrained weights directory.')
    parser.add_argument('--s', default=2.0, type=float, help='scale')
    # dataset details
    parser.add_argument('--dataset', default='city', choices=segmentation_datasets, help='Dataset name. '
                                                                                           'This is required to retrieve the correct segmentation model weights')
    parser.add_argument('--data-path', default='./sample_images', type=str, help='Image folder location')
    # input details
    parser.add_argument('--im-size', type=int, nargs="+", default=[1024, 512], help='Image size for testing (W x H)')
    parser.add_argument('--model-width', default=224, type=int, help='Model width')
    parser.add_argument('--model-height', default=224, type=int, help='Model height')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')

    args = parser.parse_args()

    if not args.weights_test:
        from weight_segmentation import model_weight_map

        model_key = '{}_{}'.format(args.model, args.s)
        dataset_key = '{}_{}x{}'.format(args.dataset, args.im_size[0], args.im_size[1])
        assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
        assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
        args.weights_test = model_weight_map[model_key][dataset_key]['weights']

    # This key is used to load the ImageNet weights while training. So, set to empty to avoid errors
    args.weights = ''

    main(args)
