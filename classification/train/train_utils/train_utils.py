from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from classification.train.train_utils.consts import (ImageLoader, ModelType,
                                                     ScoreKind, WandbScoreDict)
from classification.train.train_utils.log_scores import (
    log_scores_binaryclass, log_scores_multiclass)


def calculate_mean_std(batch_size: int, image_loader: ImageLoader = None, full_size: int = None, input_size: int = None,
                       train_dir: str = None, n_image_channels: int = 3, num_workers: int = 4) -> Tuple[Tensor, Tensor]:
    '''Calculate image set mean and standard deviation for normalization.
    Process images in batches to avoid timeout errors.

    See: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
        and https://www.youtube.com/watch?v=y6IEcEBRZks

    Args:
        batch_size (int): the size of image batches to load. Can be higher than the training batch_size without causing memory issues.

    Returns:
        (Tensor, Tensor): a tuple of pytorch tensors (mean, standard deviation) with dimension n_image_channels.
    '''
    normalize_trans = transforms.Compose([
        transforms.Resize(full_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])
    normalize_set = ImageFolder(
        root=train_dir, transform=normalize_trans, loader=image_loader)

    # Split train set into batches
    loader = DataLoader(
        normalize_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialise batch variables
    n_images = len(normalize_set)
    n_batches = round(np.ceil(n_images/batch_size))
    batch_means = torch.zeros((n_image_channels, n_batches))
    batch_means_sq = torch.zeros((n_image_channels, n_batches))
    batch_sizes = torch.zeros(n_batches)
    batch_idx = 0

    # Calculate pixel value sum and squared sum over each batch, storing batch size in the 4th column
    for _, data in tqdm(enumerate(loader), total=len(loader)):
        # images has dimension torch.Size([batch_size, c, input_size, input_size])
        images, _ = data
        # Save batch size
        batch_size = len(images)
        batch_sizes[batch_idx] = batch_size

        # Calculate batch mean and square mean
        batch_means[:, batch_idx] = torch.mean(images, dim=[0, 2, 3])
        batch_means_sq[:, batch_idx] = torch.mean(
            images ** 2, dim=[0, 2, 3])

        batch_idx += 1

    # Calculate mean and standard deviation over the entire data set
    norm_batch_sizes = batch_sizes/n_images
    mean = torch.matmul(batch_means, norm_batch_sizes)

    norm_mean_squared = torch.matmul(batch_means_sq, norm_batch_sizes)
    std = torch.sqrt(norm_mean_squared - mean ** 2)

    print(f'----------- mean, std: {mean}, {std}')
    return mean, std


def test_model(device: str, model, testloader, criterion, epoch: int = 0, kind: Literal[ScoreKind.TEST, ScoreKind.VALIDATION] = None,
               suppress_logging: bool = False, none_label: int = 2, num_classes: int = 0,
               idx_to_class: Dict[int, str] = None, calculate_weighted_scores: bool = False) -> WandbScoreDict:
    """Run a round of testing/validation with the current model state

    Returns:
        (WandbScoreDict) a dict of the calculated scores.
    """
    model.eval()
    if not suppress_logging:
        print(kind.value)
    test_running_loss: float = 0.0
    test_running_correct: float = 0.0
    test_running_correct_binary: float = 0.0
    counter = 0

    epoch_labels_test = torch.tensor([]).to(device)
    epoch_preds_test = torch.tensor([]).to(device)

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(image)

            # Calculate the loss
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            # Calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()

            # Calculate the binary accuracy (snow avalanche yes/no)
            labels_binary = (labels == none_label)
            preds_binary = (preds == none_label)
            test_running_correct_binary += (preds_binary ==
                                            labels_binary).sum().item()

            epoch_labels_test = torch.cat((epoch_labels_test, labels), 0)
            epoch_preds_test = torch.cat((epoch_preds_test, preds), 0)

    # To do: if counter == 0, then do not calculate scores but return something that makes sense

    # Create binary labels (0 = Avalanche, 1 = None Avalanche)
    epoch_labels_binary_test = (
        epoch_labels_test == none_label).type(torch.uint8)
    epoch_preds_binary_test = (
        epoch_preds_test == none_label).type(torch.uint8)

    # Get logging dicts for multi/binary class
    wandb_dict = log_scores_multiclass(epoch_labels_test.tolist(), epoch_preds_test.tolist(),
                                       kind=kind, epoch=epoch, idx_to_class=idx_to_class,
                                       suppress_logging=suppress_logging, calculate_weighted_scores=calculate_weighted_scores)

    if num_classes > 2:
        wandb_dict_binary = log_scores_binaryclass(epoch_labels_binary_test.tolist(),
                                                   epoch_preds_binary_test.tolist(), kind=kind, epoch=epoch,
                                                   suppress_logging=suppress_logging, calculate_weighted_scores=calculate_weighted_scores)
    else:
        wandb_dict_binary = {}

    # Calculate loss and accuracy for the complete epoch
    epoch_loss = test_running_loss / counter

    score_dict: WandbScoreDict = {**wandb_dict, **wandb_dict_binary}
    score_dict['loss'] = epoch_loss
    return score_dict


def get_pytorch_model(architecture: str, weights=None, num_classes: int = 4) -> ModelType:
    '''Fetch the correct pytorch model from the PytorchModel architecture param.'''
    if architecture.startswith("ResNet"):
        if architecture == "ResNet152":
            model = models.resnet152(weights=weights)
        elif architecture == "ResNet101":
            model = models.resnet101(weights=weights)
        elif architecture == "ResNet50":
            model = models.resnet50(weights=weights)
        elif architecture == "ResNet34":
            model = models.resnet34(weights=weights)
        elif architecture == "ResNet18":
            model = models.resnet18(weights=weights)
        else:
            raise ValueError(f"ResNet {architecture} not supported")

        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=num_classes)
    elif architecture.startswith("vgg"):
        if architecture == "vgg19":
            model = models.vgg19(weights=weights)
        elif architecture == "vgg16":
            model = models.vgg16(weights=weights)
        elif architecture == "vgg13":
            model = models.vgg13(weights=weights)
        else:
            raise ValueError(f"VGG {architecture} not supported")
        # Replace last layer of classifier with self.num_classes output classes
        model.classifier[-1] = nn.Linear(
            in_features=model.classifier[-1].in_features, out_features=num_classes)
    elif architecture.startswith("vit"):
        if architecture == "vit_b_16":
            model = models.vit_b_16(weights=weights)
        else:
            raise ValueError(f"ViT {architecture} not supported")
        # Replace head with the correct number of output classes
        model.heads[-1] = nn.Linear(
            in_features=model.heads[-1].in_features, out_features=num_classes)
    else:
        raise ValueError(f"Model {architecture} is not supported!")

    return model
