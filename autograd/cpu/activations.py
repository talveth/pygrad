
from typing import Union

class ReLU:
    """
    Performs ReLU.
    """
    def __init__(self, label:Union[None,int,str]="ReLU"):
        self.label  = label
        self.value  = None
        self.tensor = None

    def __repr__(self) -> str:
        if self.tensor is None:
            raise "Unable to __repr__, do a forward pass first."
        return f"ReLU=(name={self.label})"

    def __call__(self, x):
        out = x.relu()
        out.label = self.label
        self.tensor = out
        return self.tensor
