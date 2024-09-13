import numpy as np
import torch
from utils.datasets import SpeckleDataset


@torch.no_grad()
def validate_speckle(model,
                     attn_splits_list=False,
                     corr_radius_list=False,
                     prop_radius_list=False,
                     ):
    model.eval()

    val_dataset = SpeckleDataset('./dataset/eval', 320)
    print('Number of validation image pairs: %d' % len(val_dataset))

    epe_list = []
    results = {}

    s00_05_list = []
    s05_10_list = []
    s10plus_list = []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        results_dict = model(image1, image2,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             )

        # useful when using parallel branches
        flow_pr = results_dict['flow_preds'][-1]
        flow = flow_pr[0].cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        flow_gt_speed = mag

        valid_mask = (flow_gt_speed < 0.5)
        if valid_mask.max() > 0:
            s00_05_list.append(epe[valid_mask].cpu().numpy())

        valid_mask = (flow_gt_speed >= 0.5) * (flow_gt_speed <= 1)
        if valid_mask.max() > 0:
            s05_10_list.append(epe[valid_mask].cpu().numpy())

        valid_mask = (flow_gt_speed > 1)
        if valid_mask.max() > 0:
            s10plus_list.append(epe[valid_mask].cpu().numpy())

        epe = epe.view(-1)
        val = valid_gt.view(-1) >= 0.5

        epe_list.append(epe[val].cpu().numpy())

    epe_list = np.concatenate(epe_list)

    epe = np.mean(epe_list)

    print("Validation dataset AEE: %.3f" % epe)
    results['dataset_AEE'] = epe

    s00_05 = np.mean(np.concatenate(s00_05_list))
    s05_10 = np.mean(np.concatenate(s05_10_list))
    s10plus = np.mean(np.concatenate(s10plus_list))

    print("Validation dataset AEE, s0_0.5: %.3f, s0.5_1: %.3f, s1+: %.3f" % (
        s00_05,
        s05_10,
        s10plus))

    results['dataset_AEE_s0_0.5'] = s00_05
    results['dataset_AEE_s0.5_1'] = s05_10
    results['dataset_AEE_s1+'] = s10plus

    return results
