from random import sample
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def simple_sampling(anchor_kps, gt_kps, N):
    """
    Arguments:
        anchor_kps: B, 2, 40 
        gt_kps: B, 2, 40 
            Should be normalized in advance
    Returns:
        Tensor shape of B, 40, N, 2 (where GT is located at 0 of last dimension)
    """
    B, _, n_kps = anchor_kps.shape

    sampled_points = (torch.rand(B, n_kps, N - 1, 2) * 2 - 1).to(anchor_kps.device)
    anchor_kps = repeat(anchor_kps, 'B C K -> B K N C', N=N)

    gt_kps = rearrange(gt_kps, 'B C K -> B K () C')
    gt_with_sampled = torch.cat((gt_kps, sampled_points), dim=2) # B K N C
    return anchor_kps, gt_with_sampled

@torch.no_grad()
def simple_sampling(anchor_kps, gt_kps, N, img_size=512):
    print('here')
    """
    Arguments:
        anchor_kps: B, 2, 40 
        gt_kps: B, 2, 40 
            Should be normalized in advance
    Returns:
        Tensor shape of B, 40, N, 2 (where GT is located at 0 of last dimension)
    """
    B, _, n_kps = anchor_kps.shape


    sampled_points = (torch.randint(0, img_size, (B, n_kps, N-1, 2)) / (img_size - 1) * 2 -1).to(anchor_kps.device)
    anchor_kps = repeat(anchor_kps, 'B C K -> B K N C', N=N)

    gt_kps = rearrange(gt_kps, 'B C K -> B K () C')
    gt_with_sampled = torch.cat((gt_kps, sampled_points), dim=2) # B K N C
    return anchor_kps, gt_with_sampled
