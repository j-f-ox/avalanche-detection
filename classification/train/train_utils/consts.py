from enum import Enum
from typing import Any, Callable, TypedDict, Union

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

LabelScores = TypedDict('WandbMulticlassDict',
                        {'glide/f1': float,
                         'glide/precision': float,
                         'glide/recall': float,
                         'loose/f1': float,
                         'loose/precision': float,
                         'loose/recall': float,
                         'none/f1': float,
                         'none/precision': float,
                         'none/recall': float,
                         'slab/f1': float,
                         'slab/precision': float,
                         'slab/recall': float, }
                        )
'''Scores available when the standard avalanche labels are used'''


class WandbMulticlassDict(LabelScores):
    '''Dict with wandb multiclass scores.'''
    epoch: int
    accuracy: float
    f1: float
    precision: float
    recall: float


WandbBinaryDict = TypedDict('WandbBinaryDict',
                            {'epoch': int,
                             'binary/accuracy': float,
                             'binary/precision': float,
                             'binary/recall': float,
                             'binary/f1': float, }
                            )
'''Dict with wandb binary scores.'''


class WandbScoreDict(WandbMulticlassDict, WandbBinaryDict):
    '''Dict containing all wandb scores'''
    loss: float


class EarlyStoppingInfo(TypedDict):
    '''Dict with early stopping information.'''
    best_valid_acc: int
    '''The validation accuracy of the best model so far'''
    early_stopping_path: str
    '''The path to save the best model at'''
    early_stopping_epoch: int
    '''The training epoch of the best model so far'''


BEST_VALID_ACC = 'best_valid_acc'
ES_PATH = 'early_stopping_path'
ES_EPOCH = 'early_stopping_epoch'

# Transforms constants
RANDOM_CROP = 'random_crop'
HORIZONTAL_FLIP = 'horizontal_flip'
COLOUR_JITTER = 'colour_jitter'
AFFINE_TRANSFORM = 'affine_transform'

ImageLoader = Callable[[Any, str], Any]
'''Type alias for an image loading function to load images for training. Expects (self, path) as input and returns a numpy image.'''


class ScoreKind(Enum):
    """Supported score logging types"""
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "validation"


CriterionType = nn.CrossEntropyLoss

# Supported optimizer types
OptimizerType = optim.Adam

# Supported model types
ModelType = Union[models.ResNet, models.VGG, models.VisionTransformer]


NONE = "none"
'''NONE label constant'''
