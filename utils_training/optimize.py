from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange

from utils_training.utils import flow2kps_unbalanced
from utils_training.evaluation import Evaluator

r'''
    loss function implementation from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

def test_epoch_nemf(net,
                   val_loader,
                   device,
                   alpha_list=[0.1, 0.15, 0.07, 0.05, 0.03, 0.01]):
   
    net.eval()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    pck_array = {}
    for alpha in alpha_list:
        pck_array[alpha] = []

    for i, mini_batch in pbar:
        src_imsize = mini_batch['src_img'].shape[-2:]
        trg_imsize = mini_batch['trg_img'].shape[-2:]
        src_resized_enc = F.interpolate(mini_batch['src_img'].float(), size=256, mode='bicubic', align_corners=True)
        trg_resized_enc = F.interpolate(mini_batch['trg_img'].float(), size=256, mode='bicubic', align_corners=True)

        pred_flow, _ = net.module.forward_inference(
            trg_resized_enc.to(device),
            src_resized_enc.to(device),
            mini_batch['trg_kps'].squeeze(0).t()[:, :mini_batch['n_pts'][0]].to(device),
            mini_batch['src_kps'].squeeze(0).t()[:, :mini_batch['n_pts'][0]].to(device),
            trg_imsize, src_imsize
        )

        trg_kps = rearrange(mini_batch['trg_kps'], 'B N C -> B C N')
        estimated_kps = flow2kps_unbalanced(trg_kps.to(device), pred_flow, mini_batch['n_pts'].to(device), upsample_size=(trg_imsize[0], trg_imsize[1]))
            
        for alpha in alpha_list:
            eval_results = Evaluator.eval_kps_transfer_test(estimated_kps.to(device), mini_batch, alpha)
            pck_array[alpha].extend(eval_results['pck'][:])
    
        pbar.set_description(
            'Test: PCK:%.3f' % (sum(pck_array[0.1]) / (len(pck_array[0.1]))))
    
    mean_pck = {}
    for alpha in alpha_list:
        mean_pck[alpha] = sum(pck_array[alpha]) / len(pck_array[alpha])
        print(f'PCK@{alpha}:{mean_pck[alpha]}')

    return mean_pck[0.1]

def test_epoch_fast(net,
                   val_loader,
                   device,
                   alpha_list=[0.1, 0.15, 0.07, 0.05, 0.03, 0.01]):
    net.eval()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    pck_array = {}
    for alpha in alpha_list:
        pck_array[alpha] = []
    
    cls_array = {}
    cls_id_list = list(range(18))
    for cls_id in cls_id_list:
        cls_array[cls_id] = []
        
    for i, mini_batch in pbar:
     
        cls_id = mini_batch['category_id']
        src_imsize = mini_batch['src_img'].shape[-2:]
        trg_imsize = mini_batch['trg_img'].shape[-2:]
        src_resized_enc = F.interpolate(mini_batch['src_img'].float(), size=256, mode='bicubic', align_corners=True) 
        trg_resized_enc = F.interpolate(mini_batch['trg_img'].float(), size=256, mode='bicubic', align_corners=True)

        estimated_kps, _ = net.module.forward_inference(
            trg_resized_enc.to(device),
            src_resized_enc.to(device),
            mini_batch['trg_kps'].squeeze(0).t()[:, :mini_batch['n_pts'][0]].to(device),
            mini_batch['src_kps'].squeeze(0).t()[:, :mini_batch['n_pts'][0]].to(device),
            trg_imsize, src_imsize)

        for alpha in alpha_list:
            eval_results = Evaluator.eval_kps_transfer_test(estimated_kps.to(device), mini_batch, alpha)
            pck_array[alpha].extend(eval_results['pck'][:])
            
            if alpha == 0.1:
                cls_array[cls_id.item()].extend(eval_results['pck'][:])
    
        pbar.set_description(
            'Test: PCK:%.3f' % (sum(pck_array[0.1]) / (len(pck_array[0.1]))))
        
    mean_pck = {}
    for alpha in alpha_list:
        mean_pck[alpha] = sum(pck_array[alpha]) / len(pck_array[alpha])
        print(f'PCK@{alpha}:{mean_pck[alpha]}')
    
    mean_pck_cls = {}
    for cls_id in cls_id_list:
        if len(cls_array[cls_id]):
            mean_pck_cls[cls_id] = sum(cls_array[cls_id]) / len(cls_array[cls_id])
            print(f'PCK_per_class@{cls_id}:{mean_pck_cls[cls_id]}')
        else: 
            print(f'Dataset doesnt have {cls_id} category!')

    return mean_pck[0.1]
    