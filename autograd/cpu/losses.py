
from .tensor import Tensor
import numpy as np


class BCELoss:
    def __init__(self, label="BCELoss"):
        # expects (B, 1, 1)
        self.label  = label

    def __repr__(self) -> str:
        return f"BCELoss=(name={self.label})"

    def __call__(self, pred, target):
        assert isinstance(pred, (Tensor)), "Ensure the prediction is of type Tensor"
        assert isinstance(target, (Tensor)), "Ensure the target is of type Tensor"
        assert len(pred.shape)      == 3, "Ensure the prediction shape is 3, (batch_size, 1, 1)"
        assert len(target.shape)    == 3, "Ensure the target shape is 3, (batch_size, 1, 1)"
        assert pred.shape[-1]       == 1 and target.shape[-1] == 1, "Ensure pred and target shapes have 1 as their last dimension"
        assert pred.shape           == target.shape, "Ensure pred and target have the same shape"
        assert np.max(pred.value)   <= 1 and np.min(pred.value) >= 0, "Ensure predictions have value in [0,1]"
        assert np.max(target.value) <= 1 and np.min(target.value) >= 0, "Ensure targets have values in {0,1}"
        outp          = target * pred.log() + (1-target) * ((1-pred).log())       # is batched
        loss          = (-1/pred.shape[0]) * outp.sum(over_batch=True)            # sums in batch dimension also
        return loss


class CCELoss:
    def __init__(self, label="CCELoss"):
        self.label = label
    
    def __repr__(self):
        return f"CCELoss=(name={self.label})"
    
    def __call__(self, pred, target, mask:bool=False):
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
            loss             = (target * (pred.softmax().log())).sum(over_batch=True)
            loss_multiplier  = Tensor(value=np.ones_like(loss.value)*(-1/pred.shape[0]))
            loss             = loss * loss_multiplier
        else:
            # fix scalar issue
            mask_np          = (target[:,:,0] != 1)[...,None]
            eos_no_mask      = ((target[:,:,0] == 1).cumsum(axis=1) == 1)[...,None]
            mask             = Tensor(mask_np | eos_no_mask, learnable=False, leaf=True)
            loss             = ((target*mask)*(pred.softmax().log())).sum(over_batch=True)
            loss_multiplier  = Tensor(value=np.ones_like(loss.value)*(-1/np.sum(mask[:])))
            loss             = loss * loss_multiplier
        return loss
