from enum import Enum
from dataclasses import dataclass
import logging
import vector

log = logging.getLogger(__name__)

class Mode(Enum):
    TRAIN = "train"
    PREDICT = "predict"
    BATCH_PREDICT = "batch_predict"
    PLOT = "plot"
    TEST_FLOW = "test_flow"
    CHECKPOINT_EVOLUTION = "checkpoint_evolution"

@dataclass
class EventNumpy:
    q: vector.MomentumNumpy4D
    p1: vector.MomentumNumpy4D
    p2: vector.MomentumNumpy4D
    k1: vector.MomentumNumpy4D
    k2: vector.MomentumNumpy4D
