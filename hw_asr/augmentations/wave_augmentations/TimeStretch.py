from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug_ = None

    def __call__(self, data: Tensor) -> Tensor:
        pass
