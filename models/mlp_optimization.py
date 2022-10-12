import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import numpy as np
from einops import rearrange, repeat

from utils.harmonic_embedding import HarmonicEmbedding
from utils.sampling import simple_sampling
from models.base.mlp import LocalDecoder
from models.base.resnet import ResNet101
from models.base.cost_embedding import CostEmbedding4d
from utils_training.utils import soft_argmax
from utils.flow_util import unnormalise_and_convert_mapping_to_flow

class MLPTrainer(nn.Module):
    def __init__(self, hidden_dim=256, skips=(), depth=3, N=100, enc_feature_size=16, img_size=512):
        super().__init__()
        
        self.img_size = img_size
        self.enc_feature_size = enc_feature_size
        self.embedding = HarmonicEmbedding(n_harmonic_functions=10, logspace=False, append_input=True)

        self.feature_dim = np.array([256, 512, 1024, 2048])
        input_dim = [42 + 42]

        self.mlp = nn.ModuleList([LocalDecoder(dim=i,
                                        c_dim=64,
                                        hidden_size=hidden_dim,
                                        n_blocks=depth,
                                        leaky=False,
                                        sample_mode='bilinear',
                                        padding=0.1
                                       ) 
                                        for i in input_dim])

        self.cost_embedding = CostEmbedding4d()
        self.backbone = ResNet101((3, 7, 30, 33))
        self.N = 100
        self.N_hard_sample = N//2
        self.hardness = 1.
        self.std_alpha = 2
        self.feat_dropout = nn.Dropout(p=0.5)

    
    def MLP_score(self, cost, corr, src_coord, trg_coord, mlp, embedding):
        """
        Arguments:
            cost: B H_s W_s H_t W_t
            src_feat: B C H W
            trg_feat: B C H W
            src_coord: B K N C
            trg_coord: B K N C
        Returns:
            ... D
        """
  
        B, N, K, _ = src_coord.shape

        src_coord_embedded = embedding(src_coord).detach()
        trg_coord_embedded = embedding(trg_coord).detach()
        cost = torch.cat([cost, corr], dim=1)
        cost = self.cost_embedding(cost)
        _, C, _, _, H_t, W_t = cost.shape
        
        cost = rearrange(cost, 'B C H_s W_s H_t W_t -> B (C H_t W_t) H_s W_s')
        cost = F.grid_sample(cost, src_coord, mode='bilinear', align_corners=True)
        cost = rearrange(cost, 'B (C H_t W_t) N K -> (B N K) C H_t W_t', C=C, H_t=H_t, W_t=W_t)
        trg_coord = rearrange(trg_coord, 'B N K C -> (B N K) () () C')
        cost = F.grid_sample(cost, trg_coord, mode='bilinear', align_corners=True)
        cost = cost.view(B, N, K, C)
      
        cost = mlp(p=torch.cat([src_coord_embedded, trg_coord_embedded], dim=-1), c_plane=cost)

        return cost
 
    def MLP_score_batch(self, refined_corr, projected_corr, src_coord, trg_coord, mlp, embedding, batch_size=100000):
        """
        Arguments:
            cost: B H_s W_s H_t W_t
            src_feat: B C H W
            trg_feat: B C H W
            src_coord: B K N C
            trg_coord: B K N C
        Returns:
            ... D
        """
        B, K, N, _ = src_coord.shape
        src_coord = rearrange(src_coord, 'B K N C -> () () (B K N) C')
        trg_coord = rearrange(trg_coord, 'B K N C -> () () (B K N) C')
        
        cost_batched = []
        
        for src_c, trg_c in zip(src_coord.split(batch_size, dim=2), trg_coord.split(batch_size, dim=2)):
            cost_batched.append(self.MLP_score(refined_corr, projected_corr, src_c, trg_c, mlp, embedding))

        return torch.cat(cost_batched, dim=-1).view(B, K, N)

    def forward_MLP(self, cost, corr, mlp, trg_kps, src_kps):
        
        B = cost.size(0)
        cost = cost.view(B, -1, *((self.enc_feature_size,) * 4))
        corr = corr.view(B, -1, *((self.enc_feature_size,) * 4))
        
        src_sampled, trg_sampled = simple_sampling(src_kps, trg_kps, N=self.N)
        trg_sampled_reverse, src_sampled_reverse = simple_sampling(trg_kps, src_kps, N=self.N)
        
        c = self.MLP_score(cost, corr, src_sampled, trg_sampled, mlp, self.embedding)
        c_reverse = self.MLP_score(cost, corr, src_sampled_reverse, trg_sampled_reverse, mlp, self.embedding)

        return torch.cat((c, c_reverse), dim=1)
    
    def forward(self, corr_enc, corr, src_img, trg_img, trg_kps, src_kps, beta=0.2):
        B = src_kps.size(0)
        src_kps = src_kps / (self.img_size - 1) * 2 - 1 # Normalize to [-1, 1]
        trg_kps = trg_kps / (self.img_size - 1) * 2 - 1

        logits = self.forward_MLP(
            corr_enc, corr, self.mlp[0], trg_kps, src_kps)

        return logits
    
    def generate_grid(self, size):
        return torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, size[1]), torch.linspace(-1, 1, size[0]), indexing='xy'), dim=-1)

    def optimize_fast(self, refined_corr, projected_corr, src_kps, trg_kps, src_imsize, trg_imsize, src_coord=None, epochs=20, lr=3e-4, alpha=0.1):
        device = refined_corr.device

        projected_corr = projected_corr.view(1, -1, *((self.enc_feature_size,) * 4)).detach()
        refined_corr = refined_corr.view(1, -1, *((self.enc_feature_size,) * 4)).detach()
    
        trg_kps = trg_kps.squeeze(0).float()
        trg_kps[0] = trg_kps[0] / (trg_imsize[1] - 1) * 2 - 1
        trg_kps[1] = trg_kps[1] / (trg_imsize[0] - 1) * 2 - 1
        trg_kps = rearrange(trg_kps, 'C n_pts -> () () n_pts C')
        
        if src_coord == None:
            grid = soft_argmax(projected_corr.mean(1).flatten(1, 2))
        else:
            grid = src_coord
        
        grid = F.grid_sample(grid, trg_kps, mode='bilinear', align_corners=True) # 1 2 1 n_pts
        grid = rearrange(grid, '() C () n_pts -> () () n_pts C')
        src_coord = nn.Parameter(grid.detach().clone()).to(device) # learnable coordinate. shape: 1 H_s W_s 2
        trg_coord = trg_kps.clone()

        optimizer = optim.AdamW(params=[src_coord], lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
          
            score = self.MLP_score(refined_corr, projected_corr, src_coord, trg_coord, self.mlp[0], self.embedding).squeeze(0)
            Loss = -torch.log(torch.sigmoid(score)).mean()

            Loss.backward()
            optimizer.step()

            with torch.no_grad():
                src_coord.data.copy_(torch.clamp(src_coord, grid - alpha, grid + alpha))
                # src_coord.data.copy_(src_coord.clip(-1, 1))
            scheduler.step(epoch)
        
        pred_kps = src_coord.data.detach().clone()
        pred_kps = rearrange(pred_kps, '() () n_pts C -> C n_pts')

        pred_kps[0] = (pred_kps[0] + 1) / 2 * (src_imsize[1] - 1)
        pred_kps[1] = (pred_kps[1] + 1) / 2 * (src_imsize[0] - 1)
        return pred_kps[None]

    def optimize(self, refined_corr, projected_corr, src_imsize, trg_imsize, src_coord=None, last=True, epochs=20, lr=3e-4, batch_size=100000, loss_weights=[1, 1, 1, 1], alpha=0.1):
        device = refined_corr.device
        
        projected_corr = projected_corr.view(1, -1, *((self.enc_feature_size,) * 4)).detach()
        refined_corr = refined_corr.view(1, -1, *((self.enc_feature_size,) * 4)).detach()

        # initialize coordinates
        grid_enc =  soft_argmax(projected_corr.mean(1).flatten(1, 2))
        grid_enc = F.interpolate(grid_enc, size=trg_imsize, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        
        if src_coord == None:
            grid = soft_argmax(projected_corr.mean(1).flatten(1, 2))
        else:
            grid = src_coord
        
        if last:
            grid = F.interpolate(grid, size=trg_imsize, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
        else:
            grid = grid.permute(0, 2, 3, 1)
        
        src_coord = nn.Parameter(grid.detach().clone()).to(device)  # learnable coordinate. shape: 1 H_s W_s 2
        trg_coord = self.generate_grid(size=grid.shape[1:3])[None].to(device) # 1 H_t W_t 2

        # initialize optimizer
        optimizer = optim.AdamW(params=[src_coord], lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(epochs):
            src_batches = src_coord.flatten(0, -2)
            trg_batches = trg_coord.flatten(0, -2)
            B = src_batches.size(0)
            for i in range(0, B, batch_size):
                src_batch = src_batches[i:min(i + batch_size, B)]
                trg_batch = trg_batches[i:min(i + batch_size, B)]
                src_batch = rearrange(src_batch, 'P C -> () () P C')
                trg_batch = rearrange(trg_batch, 'P C -> () () P C')

                optimizer.zero_grad()
                scores_loss = []
                score = self.MLP_score(refined_corr, projected_corr, src_batch, trg_batch, self.mlp[-1], self.embedding).squeeze(0)
                loss = -torch.log(torch.sigmoid(score)).mean()
                scores_loss.append(loss)
                
                Loss = sum([l * w for l, w in zip(scores_loss, loss_weights)])
                Loss.backward()
                optimizer.step()

                with torch.no_grad():
                    # src_coord.data.copy_(src_coord.clip(-1, 1))
                    src_coord.data.copy_(torch.clamp(src_coord, grid - alpha, grid + alpha))
            scheduler.step(epoch)
        
        # 1, H, W, 2
        src_coord_mlp = rearrange(src_coord.data.detach().clone(), 'B H W C -> B C H W')
        src_coord_enc = rearrange(grid_enc, 'B H W C -> B C H W')
        
        flow_mlp = unnormalise_and_convert_mapping_to_flow(src_coord_mlp)
        flow_enc = unnormalise_and_convert_mapping_to_flow(src_coord_enc)
        flow_mlp = rearrange(flow_mlp, '() C H W -> H W C')
        flow_enc = rearrange(flow_enc, '() C H W -> H W C')

        src_img = rearrange(src_img, '() C H W -> H W C')
        trg_img = rearrange(trg_img, '() C H W -> H W C')

        if not last:
            return src_coord_mlp
        else:
            return rearrange(flow_mlp, 'H W C -> 1 C H W')
    
    def forward_patchmatch(self, refined_corr, projected_corr, src_kps, trg_kps, src_imsize, trg_imsize, iterations=20, alpha=0.1):

        projected_corr = projected_corr.view(1, -1, *((self.enc_feature_size,) * 4)).detach()
        refined_corr = refined_corr.view(1, -1, *((self.enc_feature_size,) * 4)).detach()
        
        grid = projected_corr.mean(1)
        grid = soft_argmax(grid.view(1, -1, self.enc_feature_size, self.enc_feature_size))
                
        corr = grid.clone()
        mlp = self.mlp[-1]
  
        prop_directions = 5 # current position, up, down, left, right
        res = (trg_imsize[0], trg_imsize[1])

        with torch.no_grad():
            corr = F.interpolate(corr, size=res, mode='bilinear', align_corners=True)
            corr_prev = corr.clone()
             
            for it in range(iterations):
                ########### Step 1. Propagation ###########
                horizontal = F.pad(corr, pad=(1, 1, 0, 0), mode='reflect')
                vertical = F.pad(corr, pad=(0, 0, 1, 1), mode='reflect')

                up = torch.roll(horizontal, shifts=1, dims=3)[:, :, :, 1:-1]
                left = torch.roll(vertical, shifts=1, dims=2)[:, :, 1:-1, :]
                down = torch.roll(horizontal, shifts=-1, dims=3)[:, :, :, 1:-1]
                right = torch.roll(vertical, shifts=-1, dims=2)[:, :, 1:-1, :]
        
                trg_coord = self.generate_grid(res).to(corr.device)[None]
                src_coord = torch.cat((corr, up, down, left, right), dim=0) 
                src_coord = rearrange(src_coord, 'B C H W -> B H W C')

                score = self.MLP_score_batch(refined_corr, projected_corr, src_coord, trg_coord.repeat(prop_directions, 1, 1, 1), mlp, self.embedding)
                score, score_index = repeat(score, 'B H W -> B H W C', C=2).max(dim=0, keepdim=True)

                ########### Step 2. Random Search ###########
                corr = torch.gather(src_coord, dim=0, index=score_index)
                search = corr + (torch.rand_like(corr) - 0.5) * 2 * alpha

                search_score = self.MLP_score_batch(refined_corr, projected_corr, search, trg_coord, mlp, self.embedding)
                search_score = repeat(search_score, 'B H W -> B H W C', C=2)
                corr = torch.where(score < search_score, search, corr) # select random searched position if score is higher
                corr = rearrange(corr, 'B H W C -> B C H W')
                corr.data.copy_(torch.clamp(corr, corr_prev - alpha, corr_prev + alpha))
        
        # For fast optimization
        flow = self.optimize_fast(refined_corr, projected_corr, src_kps, trg_kps, src_imsize, trg_imsize, src_coord=corr)

        # flow = self.optimize(refined_corr, projected_corr, src_imsize, trg_imsize, src_coord=corr, last=True)
           
        return flow