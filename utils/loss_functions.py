import torch
from torch.nn import functional as F


def get_loss_function(cfg, device):
    if cfg.MODEL.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif cfg.MODEL.LOSS_TYPE == 'WeightedBCEWithLogitsLoss':
        positive_weight = torch.tensor([cfg.MODEL.POSITIVE_WEIGHT]).float().to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceLoss':
        criterion = soft_dice_loss
    elif cfg.MODEL.LOSS_TYPE == 'SoftDiceBalancedLoss':
        criterion = soft_dice_loss_balanced
    elif cfg.MODEL.LOSS_TYPE == 'JaccardLikeLoss':
        criterion = jaccard_like_loss
    elif cfg.MODEL.LOSS_TYPE == 'ComboLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'WeightedComboLoss':
        criterion = lambda pred, gts: 2 * F.binary_cross_entropy_with_logits(pred, gts) + soft_dice_loss(pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'FrankensteinLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + jaccard_like_balanced_loss(
            pred, gts)
    elif cfg.MODEL.LOSS_TYPE == 'WeightedFrankensteinLoss':
        positive_weight = torch.tensor([cfg.MODEL.POSITIVE_WEIGHT]).float().to(device)
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts, pos_weight=positive_weight) + 5 *\
                                      jaccard_like_balanced_loss(pred, gts)
    else:
        criterion = soft_dice_loss
    return criterion


def soft_dice_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection) / (iflat.sum() + tflat.sum() + eps))


def soft_dice_loss_multi_class(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def soft_dice_loss_multi_class_debug(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    loss_components = 1 - 2 * intersection/denom
    return loss, loss_components


def generalized_soft_dice_loss_multi_class(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-12

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width
    ysum = y.sum(dim=sum_dims)
    wc = 1 / (ysum ** 2 + eps)
    intersection = ((y * p).sum(dim=sum_dims) * wc).sum()
    denom =  ((ysum + p.sum(dim=sum_dims)) * wc).sum()

    loss = 1 - (2. * intersection / denom)
    return loss


def jaccard_like_loss_multi_class(input: torch.Tensor, y: torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y ** 2 + p ** 2).sum(dim=sum_dims) + (y*p).sum(dim=sum_dims) + eps

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def jaccard_like_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - ((2. * intersection) / denom)


def jaccard_like_balanced_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps
    piccard = (2. * intersection)/denom

    n_iflat = 1-iflat
    n_tflat = 1-tflat
    neg_intersection = (n_iflat * n_tflat).sum()
    neg_denom = (n_iflat**2 + n_tflat**2).sum() - (n_iflat * n_tflat).sum()
    n_piccard = (2. * neg_intersection)/neg_denom

    return 1 - piccard - n_piccard


def soft_dice_loss_balanced(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    dice_pos = ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

    negatiev_intersection = ((1-iflat) * (1 - tflat)).sum()
    dice_neg =  (2 * negatiev_intersection) / ((1-iflat).sum() + (1-tflat).sum() + eps)

    return 1 - dice_pos - dice_neg
