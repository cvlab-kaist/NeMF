import torch
import torch.nn.functional as F
import re

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
    
def log_args(args):
    r"""Log program arguments"""
    print('\n+================================================+')
    for arg_key in args.__dict__:
        print('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    print('+================================================+\n')

def parse_list(list_str):
    r"""Parse given list (string -> int)"""
    return list(map(int, re.findall(r'\d+', list_str)))

def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices

def flow2kps_unbalanced(trg_kps, flow, n_pts, upsample_size=(256, 256)):
    _, _, h, w = flow.size()
    flow = F.interpolate(flow, upsample_size, mode='bilinear')
    flow[:, 0, :, :] = flow[:, 0, :, :] * (upsample_size[1] / w)
    flow[:, 1, :, :] = flow[:, 1, :, :] * (upsample_size[0] / h)
    
    src_kps = []
    for trg_kps, flow, n_pts in zip(trg_kps.long(), flow, n_pts):
        size = trg_kps.size(1)
        kp_x = torch.clamp(trg_kps[0].unsqueeze(0).narrow_copy(1, 0, n_pts), 0, upsample_size[1] - 1)
        kp_y = torch.clamp(trg_kps[1].unsqueeze(0).narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
        kp = torch.cat((kp_x, kp_y), dim=0)

        estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
        estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts).cuda() * -1), dim=1)
        src_kps.append(estimated_kps)

    return torch.stack(src_kps)

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):
    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    b,_,h,w = corr.size()
    
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = torch.linspace(-1, 1, w).expand(b,w).to(corr.device)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = torch.linspace(-1, 1, h).expand(b,h).to(corr.device)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return torch.cat((grid_x, grid_y), dim=1)
