
"""
Module storing class-defined activation functions.
"""
from __future__ import annotations

from typing import Union

from .tensor import Tensor


class ReLU:
    """
    Performs ReLU activation, defined as a Class.
    """
    def __init__(self, label:Union[None,int,str]="ReLU"):
        assert isinstance(label, (int, str)) or label is None, "Ensure label is None, an int, or a str."
        self.label  = label
        self.value  = None
        self.tensor = None

    def __repr__(self) -> str:
        return f"ReLU Layer=(name={self.label})"

    def __call__(self, x:Tensor)->Tensor:
        assert isinstance(x, Tensor), "x must be of type Tensor."
        out = x.relu()
        out.label = self.label
        self.tensor = out
        return self.tensor
