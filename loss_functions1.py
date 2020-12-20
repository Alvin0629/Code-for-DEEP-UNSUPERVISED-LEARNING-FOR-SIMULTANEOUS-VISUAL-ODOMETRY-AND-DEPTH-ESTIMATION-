from __future__ import division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import configparser 
from builtins import open
from builtins import str


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from inverse_warp import inverse_warp
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, intrinsics_inv, depth, explainability_mask, pose, rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
        intrinsics_scaled_inv = torch.cat((intrinsics_inv[:, :, 0:2]*downscale, intrinsics_inv[:, :, 2:]), dim=2)

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped = inverse_warp(ref_img, depth[:,0], current_pose, intrinsics_scaled, intrinsics_scaled_inv, rotation_mode, padding_mode)
            out_of_bound = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * out_of_bound

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss += one_scale(d, mask)
    return loss






def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = Variable(torch.ones(1)).expand_as(mask_scaled).type_as(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss
    
    
    
    
    
    
    


grad_kernel = torch.FloatTensor([[ 1, 2, 1],
                                 [ 0, 0, 0],
                                 [-1,-2,-1]]).view(1,1,3,3).to(device)/4
grad_img_kernel = grad_kernel.expand(3,1,3,3).contiguous()
lapl_kernel = torch.FloatTensor([[-1,-2,-1],
                                 [-2,12,-2],
                                 [-1,-2,-1]]).view(1,1,3,3).to(device)/12

def texture_aware_smooth_loss(pred_map, image=None):
    global grad_img_kernel, lapl_kernel

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.
    eps = 0.1

    for scaled_map in pred_map:
        if image is not None:
            b, _, h, w = scaled_map.size()
            scaled_image = F.adaptive_avg_pool2d(image.detach(), (h, w))
            grad_y = F.conv2d(scaled_image, grad_img_kernel, groups=3)
            grad_x = F.conv2d(scaled_image, grad_img_kernel.transpose(2,3).contiguous(), groups=3)
            textureness = (grad_x.abs() + grad_y.abs()).sum(dim=1, keepdim=True) + eps
        else:
            textureness = 1

        disp_lapl = F.conv2d(scaled_map, lapl_kernel.type_as(scaled_map))
        loss_map = disp_lapl / textureness

        loss += loss_map.abs().mean()*weight / scaled_map.detach().mean()
        weight /= 4
    return loss






##@torch.no_grad()
def compute_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()       ####-------------------thresh<1.25^i
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]
