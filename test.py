import argparse
import pickle
import random
import time
from os import path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import data_orig
from data_orig.semantic_keypoints_datasets import ArrayToTensor
from models.encoder import NeMF
from utils_training.evaluation import Evaluator
import utils_training.optimize as optimize
from utils_training.utils import parse_list, log_args

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='NeMF Test Script')
    
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--benchmark', type=str, choices=['pfpascal', 'spair', 'pfwillow'])
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./eval')
    parser.add_argument('--pretrained', dest='pretrained',
                       help='path to pre-trained model')
    parser.add_argument('--pretrained_file_name',
                       help='path to pre-trained model')

    parser.add_argument('--seed', type=int, default=2022,
                        help='Pseudo-RNG seed')
    parser.add_argument('--alpha', type=float, default=0.1)

    # Seed
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    with open(osp.join(args.pretrained, 'args.pkl'), 'rb') as f:
        args_model = pickle.load(f)
    log_args(args_model)
    
    # Dataloader
    co_transform = None
    target_transform = transforms.Compose([ArrayToTensor()]) 
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])  

    if args.benchmark == 'pfpascal':
        test_dataset = data_orig.PFPascalDataset( args.datapath + '/PF-PASCAL', source_image_transform=input_transform,
                                                    target_image_transform=input_transform, split='test',
                                                    flow_transform=target_transform)
    if args.benchmark == 'pfwillow':
        test_dataset = data_orig.PFWillowDataset( args.datapath + '/PF-WILLOW', source_image_transform=input_transform,
                                                    target_image_transform=input_transform, split='test',
                                                    flow_transform=target_transform)
    if args.benchmark == 'spair':
        test_dataset = data_orig.SPairDataset( args.datapath + '/SPair-71k', source_image_transform=input_transform,
                                                    target_image_transform=input_transform, split='test',
                                                    flow_transform=target_transform)
                                                    
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    model = NeMF(
                feature_size=args_model.feature_size, feature_proj_dim=args_model.feature_proj_dim,
                depth=args_model.depth, num_heads=args_model.num_heads, mlp_ratio=args_model.mlp_ratio,
                hyperpixel_ids=parse_list(args_model.hyperpixel), freeze=True, mlp_image_size=512)

    if args.pretrained:
        checkpoint = torch.load(osp.join(args.pretrained, args.pretrained_file_name))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise NotImplementedError()

    model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    train_started = time.time()
    
    # For fast optimization
    val_mean_pck = optimize.test_epoch_fast(model,
                                            test_dataloader,
                                            device) 
    # val_mean_pck = optimize.test_epoch_nemf(model,
    #                                         test_dataloader,
    #                                         device) 
   
    print('mean PCK is {}'.format(val_mean_pck))
    print(args.seed, 'Test took:', time.time()-train_started, 'seconds')
