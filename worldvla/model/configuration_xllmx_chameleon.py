import logging
from typing import List

from .chameleon import ChameleonConfig

logger = logging.getLogger(__name__)


class ChameleonXLLMXConfig(ChameleonConfig):

    def __init__(
        self,
        z_loss_weight: float = 0.0,
        **kwargs,
    ):
        self.z_loss_weight = z_loss_weight
        super().__init__(
            **kwargs,
        )
