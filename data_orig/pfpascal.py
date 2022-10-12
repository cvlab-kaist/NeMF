r"""PF-PASCAL dataset"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from .semantic_keypoints_datasets import SemanticKeypointsDataset, random_crop
import scipy.io as sio
import random
import cv2

def define_mask_zero_borders(image, epsilon=1e-6):
    """Computes the binary mask, equal to 0 when image is 0 and 1 otherwise."""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            if image.shape[1] == 3:
                # image b, 3, H, W
                image = image.transpose(0, 2, 3, 1)
            # image is b, H, W, 3
            occ_mask = np.logical_and(np.logical_and(image[:, :, :, 0] < epsilon,
                                                     image[:, :, :, 1] < epsilon),
                                      image[:, :, :, 2] < epsilon)
        else:
            if image.shape[0] == 3:
                # image 3, H, W
                image = image.transpose(1, 2, 0)
            # image is H, W, 3
            occ_mask = np.logical_and(np.logical_and(image[:, :, 0] < epsilon,
                                                     image[:, :, 1] < epsilon),
                                      image[:, :, 2] < epsilon)
        mask = ~occ_mask
        mask = mask.astype(np.bool) if float(torch.__version__[:3]) >= 1.1 else mask.astype(np.uint8)
    else:
        # torch tensor
        if len(image.shape) == 4:
            if image.shape[1] == 3:
                # image b, 3, H, W
                image = image.permute(0, 2, 3, 1)
            occ_mask = image[:, :, :, 0].le(epsilon) & image[:, :, :, 1].le(epsilon) & image[:, :, :, 2].le(epsilon)
        else:
            if image.shape[0] == 3:
                # image 3, H, W
                image = image.permute(1, 2, 0)
            occ_mask = image[:, :, 0].le(epsilon) & image[:, :, 1].le(epsilon) & image[:, :, 2].le(epsilon)
        mask = ~occ_mask
        mask = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()
    return mask

def pad_to_same_shape(im1, im2):
    # pad to same shape
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)
    shape = im1.shape
    return im1, im2


def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj


class PFPascalDataset(SemanticKeypointsDataset):
    """
    Proposal Flow image pair dataset (PF-Pascal).
    There is a certain number of pairs per category and the number of keypoints per pair also varies
    """
    def __init__(self, root, split, thres='img', source_image_transform=None,
                 target_image_transform=None, flow_transform=None, output_image_size=None, training_cfg=None):
        """
        Args:
            root:
            split: 'test', 'val', 'train'
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            flow_transform: flow transformations to apply to ground-truth flow fields
            output_image_size: size if images and annotations need to be resized, used when split=='test'
            training_cfg: training config
        Output in __getittem__  (for split=='test'):
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
        """
        super(PFPascalDataset, self).__init__('pfpascal', root, thres, split, source_image_transform,
                                              target_image_transform, flow_transform, training_cfg=training_cfg,
                                              output_image_size=output_image_size)

        self.train_data = pd.read_csv(self.spt_path)
        self.src_imnames = np.array(self.train_data.iloc[:, 0])
        self.trg_imnames = np.array(self.train_data.iloc[:, 1])
        self.cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.cls_ids = self.train_data.iloc[:, 2].values.astype('int') - 1

        if split == 'train':
            self.flip = self.train_data.iloc[:, 3].values.astype('int')
        self.src_kps = []
        self.trg_kps = []
        self.src_bbox = []
        self.trg_bbox = []
        # here reads bounding box and keypoints information from annotation files. Also in most of the csv files.
        for src_imname, trg_imname, cls in zip(self.src_imnames, self.trg_imnames, self.cls_ids):
            src_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(src_imname))[:-4] + '.mat'
            trg_anns = os.path.join(self.ann_path, self.cls[cls],
                                    os.path.basename(trg_imname))[:-4] + '.mat'

            src_kp = torch.tensor(read_mat(src_anns, 'kps')).float()
            trg_kp = torch.tensor(read_mat(trg_anns, 'kps')).float()
            src_box = torch.tensor(read_mat(src_anns, 'bbox')[0].astype(float))
            trg_box = torch.tensor(read_mat(trg_anns, 'bbox')[0].astype(float))

            src_kps = []
            trg_kps = []
            for src_kk, trg_kk in zip(src_kp, trg_kp):
                if torch.isnan(src_kk).sum() > 0 or torch.isnan(trg_kk).sum() > 0:
                    continue
                else:
                    src_kps.append(src_kk)
                    trg_kps.append(trg_kk)
            self.src_kps.append(torch.stack(src_kps).t())
            self.trg_kps.append(torch.stack(trg_kps).t())
            self.src_bbox.append(src_box)
            self.trg_bbox.append(trg_box)

        self.src_imnames = list(map(lambda x: os.path.basename(x), self.src_imnames))
        self.trg_imnames = list(map(lambda x: os.path.basename(x), self.trg_imnames))

        # if need to resize the images, even for testing
        if output_image_size is not None:
            if not isinstance(output_image_size, tuple):
                output_image_size = (output_image_size, output_image_size)
        self.output_image_size = output_image_size

    def __getitem__(self, idx):
        """
        Args:
            idx:
        Returns: If split is test, dictionary with fieldnames:
            source_image
            target_image
            source_image_size
            target_image_size
            flow_map
            correspondence_mask: valid correspondences (which are originally sparse)
            source_kps
            target_kps
        """
        batch = super(PFPascalDataset, self).__getitem__(idx)

        batch['sparse'] = True
        batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'])
        batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'])

        if self.split != 'test':
            # for training, might want to have different output flow sizes
            if self.training_cfg['augment_with_crop']:
                batch['src_img'], batch['src_kps'], batch['src_bbox'] = random_crop(
                    batch['src_img'], batch['src_kps'].clone(), batch['src_bbox'].int(),
                    size=self.training_cfg['crop_size'], p=self.training_cfg['proba_of_crop'])

                batch['trg_img'], batch['trg_kps'], batch['trg_bbox'] = random_crop(
                    batch['trg_img'], batch['trg_kps'].clone(), batch['trg_bbox'].int(),
                    size=self.training_cfg['crop_size'], p=self.training_cfg['proba_of_crop'])

            if self.training_cfg['augment_with_flip']:
                if random.random() < self.training_cfg['proba_of_batch_flip']:
                    self.horizontal_flip(batch)
                else:
                    if random.random() < self.training_cfg['proba_of_image_flip']:
                        batch['src_img'], batch['src_bbox'], batch['src_kps'] = self.horizontal_flip_img(
                            batch['src_img'], batch['src_bbox'], batch['src_kps'])
                    if random.random() < self.training_cfg['proba_of_image_flip']:
                        batch['trg_img'], batch['trg_bbox'], batch['trg_kps'] = self.horizontal_flip_img(
                            batch['trg_img'], batch['trg_bbox'], batch['trg_kps'])

            '''
            # Horizontal flipping of both images and key-points during training
            if self.split == 'train' and self.flip[idx]:
                self.horizontal_flip(batch)
                batch['flip'] = 1
            else:
                batch['flip'] = 0
            '''

            batch = self.recover_image_pair_for_training(batch)
            batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'],
                                              output_image_size=self.training_cfg['output_image_size'])
            batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'],
                                              output_image_size=self.training_cfg['output_image_size'])
            batch['pckthres'] = self.get_pckthres(batch, batch['source_image_size'])

            if self.source_image_transform is not None:
                batch['src_img'] = self.source_image_transform(batch['src_img'])
            if self.target_image_transform is not None:
                batch['trg_img'] = self.target_image_transform(batch['trg_img'])

            flow = batch['flow_map']
            if self.flow_transform is not None:
                if type(flow) in [tuple, list]:
                    # flow field at different resolution
                    for i in range(len(flow)):
                        flow[i] = self.flow_transform(flow[i])
                else:
                    flow = self.flow_transform(flow)
            batch['flow_map'] = flow

            if self.training_cfg['compute_mask_zero_borders']:
                mask_valid = define_mask_zero_borders(batch['target_image'])
                batch['mask_zero_borders'] = mask_valid
        else:
            batch['src_bbox'] = self.get_bbox(self.src_bbox, idx, batch['src_imsize_ori'],
                                              output_image_size=self.output_image_size)
            batch['trg_bbox'] = self.get_bbox(self.trg_bbox, idx, batch['trg_imsize_ori'],
                                              output_image_size=self.output_image_size)
            batch['pckthres'] = self.get_pckthres(batch, batch['source_image_size'])

            batch['src_img'], batch['trg_img'] = pad_to_same_shape(batch['src_img'], batch['trg_img'])
            h_size, w_size, _ = batch['trg_img'].shape

            flow, mask = self.keypoints_to_flow(batch['src_kps'][:batch['n_pts']],
                                                batch['trg_kps'][:batch['n_pts']],
                                                h_size=h_size, w_size=w_size)

            if self.source_image_transform is not None:
                batch['src_img'] = self.source_image_transform(batch['src_img'])
            if self.target_image_transform is not None:
                batch['trg_img'] = self.target_image_transform(batch['trg_img'])
            if self.flow_transform is not None:
                flow = self.flow_transform(flow)

            batch['flow_map'] = flow
            batch['correspondence_mask'] = mask.bool() if float(torch.__version__[:3]) >= 1.1 else mask.byte()

        return batch

    def get_bbox(self, bbox_list, idx, original_image_size=None, output_image_size=None):
        r"""Returns object bounding-box"""
        bbox = bbox_list[idx].clone()
        if self.output_image_size is not None or output_image_size is not None:
            if output_image_size is None:
                bbox[0::2] *= (self.output_image_size[1] / original_image_size[1])  # w
                bbox[1::2] *= (self.output_image_size[0] / original_image_size[0])
            else:
                bbox[0::2] *= (float(output_image_size[1]) / float(original_image_size[1]))
                bbox[1::2] *= (float(output_image_size[0]) / float(original_image_size[0]))
        return bbox