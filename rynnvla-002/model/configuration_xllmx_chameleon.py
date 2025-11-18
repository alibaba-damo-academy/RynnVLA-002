import logging
from typing import List

from .chameleon import ChameleonConfig

logger = logging.getLogger(__name__)


class ChameleonXLLMXConfig(ChameleonConfig):

    def __init__(
        self,
        z_loss_weight: float = 0.0,
        action_dim: int = 7,
        time_horizon: int = 5,
        **kwargs,
    ):
        self.z_loss_weight = z_loss_weight
        self.action_dim = action_dim
        self.time_horizon = time_horizon
        super().__init__(
            **kwargs,
        )
