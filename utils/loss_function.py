import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor
from typing import Optional

import warnings

class CustomLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float
    
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0,
                 cross_entropy_factor: float = 1.0, l2_factor: float = 1.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        # Check that at least one of the loss weights is non-zero
        if cross_entropy_factor == 0 and l2_factor == 0:
            raise ValueError("At least one of cross_entropy_weight and l2_weight must be non-zero")

        self.cross_entropy_factor = cross_entropy_factor
        self.l2_factor = l2_factor
        
        # Mapping classes to 2d grid
        self.grid_coords = torch.tensor([[i // 3, i % 3] for i in range(9)], dtype=torch.float)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss_cross_entropy = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)
        
        soft_input_coords = torch.matmul(input, self.grid_coords.to(input.device))
        target_coords = self.grid_coords.to(target.device)[target]

        # L2 Norm
        distances = torch.sqrt(torch.sum((soft_input_coords - target_coords) ** 2, dim=1))

        # Apply class weights
        if self.weight is not None:
            sample_weights = self.weight[target]
            weighted_distances = distances * sample_weights
        else:
            weighted_distances = distances

        # Reduction
        if self.reduction == 'none':
            loss_l2 = weighted_distances
        elif self.reduction == 'mean':
            loss_l2 = torch.mean(weighted_distances)
        elif self.reduction == 'sum':
            loss_l2 = torch.sum(weighted_distances)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
        
        if self.label_smoothing > 0.0: 
            warnings.warn('Label smoothing is not implemented for the L2 loss component')

        loss_total = (self.cross_entropy_factor * loss_cross_entropy + self.l2_factor * loss_l2) / (self.cross_entropy_factor + self.l2_factor)

        return loss_total