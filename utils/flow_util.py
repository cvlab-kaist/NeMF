import torch

def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow
