import torch
from torch import nn, Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    A custom loss function that extends PyTorch's CrossEntropyLoss to incorporate a dynamic modulation based on a
    distance map and epoch-based smoothing.
    """
    def forward(self, input: Tensor, target: Tensor, disMap=None, current_epoch=None) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
            if disMap is not None:
                disMap = disMap[:, 0]

        if disMap is None:
            return super().forward(input, target.long())

        Tauo_st = 0  # Start epoch for smooth transition
        st_epoch = 1000  # Duration of the smooth transition
        smooth_trans = True

        # Distance map weighted loss without Smooth transition
        if smooth_trans == False:
            temp = super().forward(input, target.long())
            if disMap.shape == temp.shape:
                disMap = disMap / torch.mean(disMap)
                temp = torch.mul(temp, disMap)
            return temp

        # Distance map weighted loss with Smooth transition
        else:
            temp = super().forward(input, target.long())
            if current_epoch < Tauo_st:
                return temp
            elif Tauo_st <= current_epoch < Tauo_st + st_epoch:
                if disMap.shape == temp.shape:
                    disMap = disMap / torch.mean(disMap)
                    warm_start_matrix = torch.ones_like(disMap)
                    warm_para = float(Tauo_st + st_epoch - current_epoch) / st_epoch
                    disMap_ = warm_para * warm_start_matrix + (1 - warm_para) * disMap
                    return torch.mul(temp, disMap_)
                return temp
            elif current_epoch >= Tauo_st + st_epoch:
                if disMap.shape == temp.shape:
                    disMap = disMap / torch.mean(disMap)
                    return torch.mul(temp, disMap)
                return temp