import torch
import os
import numpy as np
from torch.utils import data

from torch_lib.Dataset import Dataset
from torchvision.models import vgg
from torch_lib.Model import Model, OrientationLoss

def calculate_iou_3d(pred_box, target_box):
    intersection = torch.min(pred_box[:, 3:], target_box[:, 3:]).clamp(0).prod(dim=1)
    pred_volume = pred_box[:, 3:].prod(dim=1)
    target_volume = target_box[:, 3:].prod(dim=1)
    union = pred_volume + target_volume - intersection
    iou = intersection / union
    return iou

def calculate_angle_difference(pred_orient, target_orient, conf):
    indexes = torch.max(conf, dim=1)[1]
    target_orient = target_orient[torch.arange(target_orient.size(0)), indexes]
    pred_orient = pred_orient[torch.arange(pred_orient.size(0)), indexes]

    theta_diff = torch.atan2(target_orient[:, 1], target_orient[:, 0])
    estimated_theta_diff = torch.atan2(pred_orient[:, 1], pred_orient[:, 0])

    angle_diff = theta_diff - estimated_theta_diff
    return angle_diff

def validate(model, generator):
    ious = []
    angle_diffs = []

    model.eval()

    with torch.no_grad():
        for local_batch, local_labels in generator:
            truth_orient = local_labels['Orientation'].float().cuda()
            truth_conf = local_labels['Confidence'].long().cuda()
            truth_dim = local_labels['Dimensions'].float().cuda()

            local_batch = local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)


            iou = calculate_iou_3d(dim, truth_dim)
            ious.append(iou)

            angle_diff = calculate_angle_difference(orient, truth_orient,conf)
            angle_diffs.append(angle_diff)

    ious = torch.cat(ious, dim=0)
    angle_diffs = torch.cat(angle_diffs, dim=0)

    mean_iou = ious.mean().item()
    std_iou = ious.std().item()

    mean_angle_diff = angle_diffs.mean().item()
    std_angle_diff = angle_diffs.std().item()

    return mean_iou, std_iou, mean_angle_diff, std_angle_diff

def main():
    batch_size = 8

    print("Loading validation dataset...")

    val_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    val_dataset = Dataset(val_path, mode='val')

    val_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 6}
    val_generator = data.DataLoader(val_dataset, **val_params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features).cuda()

    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    latest_model = None

    if os.path.isdir(model_path):
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass

    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])

        print('Found previous checkpoint: %s'%(latest_model))
        print('Loading weights....')

    mean_iou, std_iou, mean_angle_diff, std_angle_diff = validate(model, val_generator)
    print(f"Mean IoU: {mean_iou}, Std IoU: {std_iou}")
    print(f"Mean Angle Difference: {mean_angle_diff}, Std Angle Difference: {std_angle_diff}")

if __name__ == '__main__':
    main()
