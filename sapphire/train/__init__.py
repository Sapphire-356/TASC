
from .Schedule import InvLR
from .Schedule import build_lr_scheduler
from .loss import NNConsistencyLoss

LRSchedule = {
    'InvLR': InvLR,
}