import torch
import torch.nn.functional as F
import torch.nn as nn

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def focal_loss(inputs, targets, alpha=.25, gamma=2) : 
    inputs = F.sigmoid(inputs)       
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP)**gamma * BCE
    return loss

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def IOU_loss(inputs, targets, smooth=1) : 
    inputs = F.sigmoid(inputs)      
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    return 1 - IoU

def combine_loss(pred, target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

_criterion_entrypoints = {
    'bce': nn.BCEWithLogitsLoss(),
    'focal': focal_loss,
    'dice': dice_loss,
    'iou': IOU_loss,
    'combine': combine_loss
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints

def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion