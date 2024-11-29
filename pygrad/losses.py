
"""
Module storing class-defined loss functions.
"""

import numpy as np

from .tensor import Tensor


class BCELoss:
    """
    Binary Cross Entropy Loss.
    """
    def __init__(self, label:str="BCELoss"):
        # expects (B, 1, 1)
        assert isinstance(label, str), "label must be a str"
        self.label  = label

    def __repr__(self) -> str:
        return f"BCELoss=(name={self.label})"

    def __call__(self, pred:Tensor, target:Tensor)->Tensor:
        """
        Computes the BCE on pred and target, summing over the batch dimension.

        :param pred: A Tensor of shape (batch_size, 1, 1)
                     Values in [0,1]
        :param target: A Tensor of shape (batch_size, 1, 1)
                     Values in {0,1}
        """

        assert isinstance(pred, (Tensor)), "Ensure the prediction is of type Tensor"
        assert isinstance(target, (Tensor)), "Ensure the target is of type Tensor"
        assert len(pred.shape)      == 3, "Ensure the prediction shape is 3, (batch_size, 1, 1)"
        assert len(target.shape)    == 3, "Ensure the target shape is 3, (batch_size, 1, 1)"
        assert pred.shape[-1]       == 1 and target.shape[-1] == 1, "Ensure pred and target shapes have 1 as their last dimension"
        assert pred.shape           == target.shape, "Ensure pred and target have the same shape"
        assert np.max(pred.value)   <= 1 and np.min(pred.value) >= 0, "Ensure predictions have value in [0,1]"
        assert np.max(target.value) <= 1 and np.min(target.value) >= 0, "Ensure targets have values in {0,1}"
        outp          = target * pred.log() + (1-target) * ((1-pred).log())       # is batched
        loss          = (-1/pred.shape[0]) * outp.sum(axis=(0,1,2))               # sums in batch dimension also
        return loss


class CCELoss:
    """
    Categorical Cross Entropy Loss.
    """
    def __init__(self, label="CCELoss"):
        assert isinstance(label, str), "label must be a str."
        self.label = label
    
    def __repr__(self):
        return f"CCELoss=(name={self.label})"
    
    def __call__(self, pred:Tensor, target:Tensor, mask:bool=False)->Tensor:
        """
        Performs CCE on pred and target, with an optional mask.

        :param pred: A Tensor of shape (batch_size, 1, w) with values in [0,1]
        :param target: A Tensor of shape (batch_size, 1, w) with values in {0,1}
        :param mask: A boolean. If true, the CCE is only computed across values 
                     where the target has an output in dimension -1.
        """
        assert isinstance(pred, (Tensor)),          "Ensure the prediction is of type Tensor"
        assert isinstance(target, (Tensor)),        "Ensure the target is of type Tensor"
        assert len(pred.shape)      == 3,           f"Ensure the prediction shape is 3, (batch_size, 1, w), got: {pred.shape}"
        assert len(target.shape)    == 3,           f"Ensure the target shape is 3, (batch_size, 1, w), got: {target.shape}"
        assert pred.shape           == target.shape,"Ensure pred and target have the same shape"
        assert np.max(target.value) <= 1 and np.min(target.value) >= 0, "Ensure targets have values in {0,1}"
        if pred.shape[1] != 1:
            pred = pred.reshape((-1,1,pred.shape[-1]))
            target = target.reshape((-1,1,target.shape[-1]))
        
        if not mask:
            loss             = (target * (pred.softmax_log())).sum(axis=(0,1,2))
            loss_multiplier  = Tensor(value=np.ones_like(loss.value)*(-1/pred.shape[0]), dtype=pred.dtype)
            loss             = loss * loss_multiplier
        else:
            mask_np          = (target[:,:,0] != 1)[...,None]
            eos_no_mask      = ((target[:,:,0] == 1).cumsum(axis=1) == 1)[...,None]
            mask             = Tensor(mask_np | eos_no_mask, learnable=False, leaf=True, dtype=pred.dtype)
            loss             = ((target*mask)*(pred.softmax_log())).sum(axis=(0,1,2))
            loss_multiplier  = Tensor(value=np.ones_like(loss.value)*(-1/np.sum(mask[:])), dtype=pred.dtype)
            loss             = loss * loss_multiplier
        return loss
