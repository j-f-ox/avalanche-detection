import os
from typing import Any, Callable, Dict, List, Tuple, Union

from torch import DoubleTensor, Generator, Tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from classification.train.train_utils.consts import (AFFINE_TRANSFORM,
                                                     COLOUR_JITTER,
                                                     HORIZONTAL_FLIP,
                                                     RANDOM_CROP, ImageLoader)


def create_data_loaders(mean: Tensor = None, std:  Tensor = None, train_transforms: Dict[str, bool] = None,
                        train_dir: str = '', test_dir: str = '', num_workers: int = 8,
                        batch_size: int = None, image_loader: ImageLoader = None,
                        full_size: int = None, input_size: int = None, train_valid_seed=100,
                        label_mapping: Union[Dict[str, str], None] = None):
    """Split and weights train set, and create and return a tuple Pytorch data loaders.

    Optionally override trans transform preprocessing by speciying dict `label_mapping`.

    Returns:
        (train_data_loader, valid_data_loader, test_data_loader, class_to_idx) as a tuple.
    """

    # Create transformations for the train and test sets
    train_trans = _get_train_transforms_list(mean=mean, std=std, train_transforms=train_transforms,
                                             full_size=full_size, input_size=input_size)

    test_trans = transforms.Compose([
        transforms.Resize(full_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create label overrides if label_mapping is not None
    target_transform, new_class_to_idx = _get_label_overrides(
        label_mapping, train_dir=train_dir, test_dir=test_dir)

    # Load the train and test datasets into an ImageFolder
    train_data_set = ImageFolder(
        root=train_dir, transform=train_trans,
        loader=image_loader, target_transform=target_transform)
    test_data = ImageFolder(
        root=test_dir, transform=test_trans,
        loader=image_loader, target_transform=target_transform)

    # Split train set into 90% train and 10% validation set, weighting sampling as the dataset is imbalanced
    train_set_size = int(len(train_data_set) * 0.9)
    valid_set_size = len(train_data_set) - train_set_size
    assert train_set_size >= batch_size, f'Not enough images: train set size {train_set_size} smaller than batch size {batch_size}.'
    assert valid_set_size >= batch_size, f'Not enough images: valid set size {valid_set_size} smaller than batch size {batch_size}.'

    seed = Generator().manual_seed(train_valid_seed)
    train_data_subset, valid_data_subset = random_split(
        train_data_set, [train_set_size, valid_set_size], generator=seed)

    # Create samplers for the train and validation subsets
    n_classes = len(train_data_set.classes)
    train_sampler = _get_train_sampler(
        data_subset=train_data_subset, n_classes=n_classes)
    valid_sampler = _get_train_sampler(
        data_subset=valid_data_subset, n_classes=n_classes)

    # Create and return data loaders
    train_data_loader = DataLoader(
        train_data_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
        num_workers=num_workers)

    valid_data_loader = DataLoader(
        valid_data_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=valid_sampler,
        num_workers=num_workers)

    test_data_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers)

    # Get class_to_idx to return, taking (optional) label overrides into account
    if new_class_to_idx is not None:
        class_to_idx = new_class_to_idx
    else:
        class_to_idx = train_data_set.class_to_idx

    return train_data_loader, valid_data_loader, test_data_loader, class_to_idx


def _get_label_overrides(label_mapping: Union[Dict[str, str], None],
                         train_dir: str = '', test_dir: str = '') -> Tuple[Callable[[int], int], Dict[str, int]]:
    '''Convert a label mapping function label_mapping into a target_transform function and class_to_idx dict
    which can be passed to an ImageFolder to override dataset labels.

    Args:
        label_mapping (Dict[str,str]) : a label mapping to overwrite dataset labels.
        train_dir (str)               : train folder from which labels are automatically extracted.
        test_dir (str)                : test folder from which labels are automatically extracted.

    Returns:
        (int->int,Dict[str,int]) a tuple (target_transform, new_class_to_idx)
    '''
    if label_mapping is None:
        return None, None
    else:
        train_labels = [f.path.split('/')[-1]
                        for f in os.scandir(train_dir) if f.is_dir()]
        test_labels = [f.path.split('/')[-1]
                       for f in os.scandir(test_dir) if f.is_dir()]
        assert train_labels == test_labels, f'Train and test labels must be equal to deduce a label mapping. Got {train_labels} and {test_labels}'

        # Sort labels to get the indices used by PyTorch
        train_labels.sort()

        # Get unique labels from label_mapping
        new_labels = list(set(label_mapping.values()))
        new_labels.sort()
        new_class_to_idx = {label: idx for idx, label in enumerate(new_labels)}

        def _target_transform(old_label_idx: int) -> int:
            old_label: str = train_labels[old_label_idx]
            new_label = label_mapping[old_label]
            return new_class_to_idx[new_label]

        return _target_transform, new_class_to_idx


def _get_train_transforms_list(mean: Tensor = None, std: Tensor = None, train_transforms: Dict[str, bool] = None,
                               full_size: int = 0, input_size: int = 0) -> List[Any]:
    '''
    Create a list of pytorch transforms based on the given train_transforms argument.
    '''
    if train_transforms is None:
        train_trans_list = [
            transforms.Resize(full_size),
            transforms.RandomAffine(
                degrees=5, scale=(0.95, 1.05)),
            transforms.RandomCrop(input_size),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.02),
            transforms.RandomHorizontalFlip()]
    else:
        train_trans_list = []
        # Apply colour transforms before any padding
        if COLOUR_JITTER in train_transforms:
            colour_cfg = train_transforms[COLOUR_JITTER]
            train_trans_list.append(transforms.ColorJitter(**colour_cfg))

        if AFFINE_TRANSFORM in train_transforms:
            affine_cfg = train_transforms[AFFINE_TRANSFORM]
            train_trans_list.append(transforms.RandomAffine(**affine_cfg))

        # Resize and crop the image to a square of size input_size x input_size
        train_trans_list.append(transforms.Resize(full_size))
        if train_transforms[RANDOM_CROP]:
            train_trans_list.append(transforms.RandomCrop(input_size))
        else:
            train_trans_list.append(transforms.CenterCrop(input_size))

        if train_transforms[HORIZONTAL_FLIP]:
            train_trans_list.append(transforms.RandomHorizontalFlip())

    train_trans = transforms.Compose(train_trans_list + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    del train_transforms
    del train_trans_list

    return train_trans


def _get_train_sampler(data_subset, n_classes: int):
    '''Return a torch WeightedRandomSampler for the train and validation subsets.
    Needed to balance classes in sampling as the dataset classes are unbalanced.
    '''
    imgs = _get_images_from_subset(data_subset)
    weights = _make_weights_for_balanced_classes(imgs, n_classes)
    weights = DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler


def _make_weights_for_balanced_classes(images, n_classes: int) -> List[int]:
    """Weight samples as our dataset is imbalanced
     See https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""
    count = [0] * n_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        if count[i] != 0:
            weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def _get_images_from_subset(subset):
    images = []
    for i in subset.indices:
        images.append(subset.dataset.imgs[i])
    return images
