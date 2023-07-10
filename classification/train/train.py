import os
from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch import Tensor
from tqdm import tqdm

from classification.train.train_utils.consts import (BEST_VALID_ACC, ES_EPOCH,
                                                     ES_PATH, NONE,
                                                     CriterionType,
                                                     EarlyStoppingInfo,
                                                     ImageLoader, ModelType,
                                                     OptimizerType, ScoreKind,
                                                     WandbScoreDict)
from classification.train.train_utils.dataloaders import create_data_loaders
from classification.train.train_utils.log_scores import (
    log_epoch_scores, log_scores_binaryclass, log_scores_multiclass)
from classification.train.train_utils.train_utils import (calculate_mean_std,
                                                          get_pytorch_model,
                                                          test_model)
from utils.np_image import RGB, NpImage


class PytorchModel(object):
    """Wrapper class for training and testing Pytorch models

        Example:
        ```
        # Create model and start training
        Model = PytorchModel()
        ```
    """
    class_to_idx: Dict[str, int] = None
    '''Mapping between model classes and indices.
    By default {'glide': 0, 'loose': 1, 'none': 2, 'slab': 3}.
    '''

    device: Literal['cuda', 'cpu']
    '''The device to train on (cuda if GPU is available)'''

    early_stopping_info: EarlyStoppingInfo = None
    '''Dict with early stopping information'''

    mean_std: Tuple[List, List]
    '''A tuple of the tensor mean and standard deviation values of the train set converted to a list of floats.'''

    idx_to_class: Dict[int, str]
    '''Mapping between model indixes and classes'''

    save_model_path: Union[str, None] = None
    '''If set, save model state dict at save_model_path after training is complete. Should end in .pth - see https://pytorch.org/tutorials/beginner/saving_loading_models.html for more information.'''

    def __init__(self, architecture: str = 'ResNet101', batch_size: int = 16,
                 early_stopping_path: str = None, epochs: int = 75, full_size: int = 950,
                 image_loader: ImageLoader = None, input_size: int = 224*4,
                 label_mapping: Union[Dict[str, str], None] = None,
                 load_model_path: str = None, lr: float = 3e-5,
                 mean_std: tuple[Tensor, Tensor] = None, num_classes: int = 4,
                 num_workers: int = 16, optimizer: str = 'Adam',
                 save_model_path: str = None, test_dir: str = '.data/test',
                 train_dir: str = '.data/train', train_transforms: Dict[str, bool] = None,
                 use_wandb: bool = False, wandb_init: Dict = {'project': 'avalanche_detection'},
                 weights=None):
        """

        Args:
            label_mapping (Dict[str,str]): label mapping function to reassign labels for training and testing.
        """

        # Set class variables
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.test_dir = test_dir
        self.train_dir = train_dir
        assert os.path.exists(test_dir), f'Could not find path {test_dir}'
        assert os.path.exists(train_dir), f'Could not find path {train_dir}'

        self.early_stopping_path = early_stopping_path
        self.epochs = epochs
        self.input_size = input_size
        self.use_wandb = use_wandb
        self.full_size = full_size
        self.lr = lr
        self.num_classes = num_classes
        self.num_workers = num_workers

        if save_model_path is not None:
            self.save_model_path = os.path.abspath(save_model_path)
            # Assert that absolute save_model_path directory exists
            save_dir, model_file = os.path.split(save_model_path)
            assert os.path.exists(
                save_dir), f"Must specify a model_path in a directory which exists. Got: {os.path.abs_path(save_model_path)}"
            assert model_file.endswith(
                '.pth'), f"Model path name should end in .pth - got {model_file}"

        # Optionally override image loader
        if image_loader is not None:
            self.image_loader = image_loader
        else:
            self.image_loader = default_image_loader

        # Initialise wandb and log all non-callable class variables
        if self.use_wandb:
            wandb.init(**wandb_init)
            wandb.config.update({k: v for k, v in self.__dict__.items(
            ) if not k.startswith('__') and not callable(k)})
            wandb.config.update({
                'architecture': architecture,
                'load_model_path': load_model_path,
                'optimizer': optimizer,
                'weights': weights,
            })

        # Check that paths to save models to are in directories which exist
        for model_path in [early_stopping_path, save_model_path]:
            if model_path is not None:
                save_dir, _ = os.path.split(save_model_path)
                assert os.path.exists(
                    save_dir), f"Must specify a model path in a directory which exists. Got: {model_path}"
                self.early_stopping_path = os.path.abspath(early_stopping_path)
                self.save_model_path = os.path.abspath(save_model_path)

        # Calculate train set mean and std, and save in state
        if mean_std is not None:
            mean, std = mean_std
        else:
            print("Calculating mean and standard deviation")
            mean, std = calculate_mean_std(batch_size=64, image_loader=self.image_loader,
                                           full_size=self.full_size, input_size=self.input_size,
                                           train_dir=self.train_dir, num_workers=self.num_workers)
        self.mean_std = (mean.tolist(), std.tolist())

        # print("Loading and weighting train/valid/test sets")
        self.train_data_loader, self.valid_data_loader, self.test_data_loader, self.class_to_idx = create_data_loaders(
            mean, std, train_transforms=train_transforms, train_dir=self.train_dir, test_dir=self.test_dir,
            num_workers=self.num_workers, batch_size=self.batch_size, image_loader=self.image_loader,
            full_size=self.full_size, input_size=self.input_size, label_mapping=label_mapping)

        self.idx_to_class = dict([(v, k)
                                 for k, v in self.class_to_idx.items()])

        # print(f"Computation device: {self.device}\n")
        self.model, self.optimizer, self.criterion = self._initialise_model(architecture=architecture,
                                                                            weights=weights,
                                                                            load_model_path=load_model_path,
                                                                            optimizer=optimizer)

    def _initialise_training_run(self):
        '''Calculate scores before training starts, log initial scores to wandb, and set up early stopping dict.'''
        # Measure before training starts
        valid_score_dict = self._eval_valid(epoch=-1, suppress_logging=False)
        test_score_dict = self._eval_test(epoch=-1, suppress_logging=False)

        if (self.use_wandb):
            # Initialise train scores to test scores to set a starting point
            train_score_dict = {
                k.replace(ScoreKind.TEST.value, ScoreKind.TRAIN.value): v for k, v in test_score_dict.items()}

            wandb_dict = {**train_score_dict, **
                          test_score_dict, **valid_score_dict}
            wandb.log(wandb_dict)

        # Initialise early stopping
        if self.early_stopping_path is not None:
            save_dir, _ = os.path.split(self.early_stopping_path)
            validation_accuracy = valid_score_dict['validation/accuracy']
            assert os.path.exists(
                save_dir), f"Must specify an early stopping path in a directory which exists. Got: {self.early_stopping_path}"
            self.early_stopping_info = {
                ES_PATH: os.path.abspath(self.early_stopping_path),
                BEST_VALID_ACC: float(validation_accuracy),
                ES_EPOCH: 0,
            }

    def _initialise_model(self, architecture: str = None, weights: str = None, load_model_path: str = None,
                          optimizer: str = None) -> Tuple[ModelType, OptimizerType, CriterionType]:
        """Set up model and optimizer, and return these.

        Args:
            load_model_path (str|None): if set, load model state dict from this path.

        Returns:
            (model, optimizer, criterion) tuple
        """
        model = get_pytorch_model(
            architecture, weights, num_classes=self.num_classes)

        model.to(self.device)

        # Load model state dict if load_model_path is set
        if load_model_path is not None:
            with torch.no_grad():
                model.load_state_dict(torch.load(load_model_path), strict=True)

        # Optimizer
        if optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {self.optimizer} is not supported!")

        # Loss function (cross-entropy loss)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def _training_epoch(self, model, trainloader, optimizer, criterion, epoch: int) -> Dict[str, Any]:
        """Run a single epoch of training and return epoch scores."""
        device = self.device

        model.train()
        print(ScoreKind.TRAIN.value)
        train_running_loss = 0.0
        train_running_correct = 0
        train_running_correct_binary = 0
        counter = 0

        epoch_labels_train = torch.tensor([]).to(device)
        epoch_preds_train = torch.tensor([]).to(device)

        for _, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            image, labels = data

            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(image)

            # Calculate the loss
            loss = criterion(outputs, labels)
            train_running_loss += float(loss.item())

            # Calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()

            epoch_labels_train = torch.cat((epoch_labels_train, labels), 0)
            epoch_preds_train = torch.cat((epoch_preds_train, preds), 0)

            # Calculate the binary accuracy (avalanche yes/no)
            labels_binary = (labels == self.class_to_idx[NONE])
            preds_binary = (preds == self.class_to_idx[NONE])
            train_running_correct_binary += (preds_binary ==
                                             labels_binary).sum().item()

            # Update weights
            loss.backward()
            optimizer.step()

        # Evaluate scores on the train set
        train_score_dict = self._eval_train(
            epoch, epoch_labels_train, epoch_preds_train)
        train_score_dict['train/loss'] = train_running_loss / counter

        return train_score_dict

    def _eval_train(self, epoch: int, true_labels, predicted_labels) -> Dict[str, Any]:
        """Geenerate score dict on the train set for current training epoch"""
        epoch_labels_binary_train = (
            true_labels == self.class_to_idx[NONE]).type(torch.uint8)
        epoch_preds_binary_train = (
            predicted_labels == self.class_to_idx[NONE]).type(torch.uint8)

        multiclass_scores = log_scores_multiclass(true_labels.tolist(), predicted_labels.tolist(),
                                                  kind=ScoreKind.TRAIN, epoch=epoch, idx_to_class=self.idx_to_class)
        if self.num_classes > 2:
            binary_scores = log_scores_binaryclass(epoch_labels_binary_train.tolist(),
                                                   epoch_preds_binary_train.tolist(), kind=ScoreKind.TRAIN, epoch=epoch)
        else:
            binary_scores = {}

        score_dict = {**multiclass_scores, **binary_scores}
        return prepend_split_kind(score_dict, 'train')

    def _eval_test(self, epoch: int, suppress_logging: bool = False, calculate_weighted_scores: bool = False) -> Dict[str, Any]:
        """Wraper function for model testing.
        If calculate_weighted_scores is True, then also calculates weighted scores in addition to (unweighted) scores.
        """
        score_dict = self._test_model(
            self.model, self.test_data_loader, self.criterion, epoch, kind=ScoreKind.TEST,
            suppress_logging=suppress_logging, calculate_weighted_scores=calculate_weighted_scores)

        return prepend_split_kind(score_dict, prefix='test')

    def _eval_valid(self, epoch: int, suppress_logging: bool = False, calculate_weighted_scores: bool = False) -> Dict[str, Any]:
        """Wraper function for validation testing during training"""
        score_dict: WandbScoreDict = self._test_model(
            self.model, self.valid_data_loader, self.criterion, epoch, kind=ScoreKind.VALIDATION,
            suppress_logging=suppress_logging, calculate_weighted_scores=calculate_weighted_scores)

        return prepend_split_kind(score_dict, prefix='validation')

    def _test_model(self, model, testloader, criterion, epoch: int, **kwargs) -> WandbScoreDict:
        """Run a round of testing/validation with the current model state"""
        return test_model(self.device, model, testloader, criterion, epoch=epoch, none_label=self.class_to_idx[NONE],
                          num_classes=self.num_classes, idx_to_class=self.idx_to_class, **kwargs)

    def run_training(self) -> Union[None, Dict[str, Any]]:
        """Run model training"""
        # Run pretraining functions
        self._initialise_training_run()

        torch.cuda.empty_cache()

        # Start the training
        for epoch in range(self.epochs):
            train_score_dict = self._training_epoch(
                self.model, self.train_data_loader, self.optimizer, self.criterion, epoch)
            valid_score_dict = self._eval_valid(epoch)
            test_score_dict = self._eval_test(epoch)

            # Log epoch scores to the console
            print(f"[INFO]: Epoch {epoch+1} of {self.epochs}")
            log_epoch_scores(train_scores=train_score_dict,
                             test_scores=test_score_dict, valid_scores=valid_score_dict)

            if (self.use_wandb):
                # Log scores to wandb
                wandb_dict = {**train_score_dict, **
                              test_score_dict, **valid_score_dict}
                wandb.log(wandb_dict)

            # If validation accuracy is a new best value, update early stopping information
            valid_accuracy = valid_score_dict['validation/accuracy']
            best_valid_acc = self.early_stopping_info[BEST_VALID_ACC]
            actual_epoch = epoch+1
            if valid_accuracy >= best_valid_acc:
                print(
                    f"New best model at epoch {actual_epoch} with validation accuracy {valid_accuracy}")
                # Convert best loss to a float to save memory: see here https://pytorch.org/docs/stable/notes/faq.html
                self.early_stopping_info[BEST_VALID_ACC] = float(
                    valid_accuracy)
                self.early_stopping_info[ES_EPOCH] = actual_epoch
                torch.save(self.model.state_dict(),
                           self.early_stopping_info[ES_PATH])
            elif (self.early_stopping_info[ES_EPOCH] < actual_epoch-5):
                # Stop training after 6 successive epochs with no improvement in validation accuracy
                print(
                    f'Running early stopping at epoch {epoch}. Best accuracy was {best_valid_acc} at epoch {self.early_stopping_info[ES_EPOCH]}')
                break
            print('-'*50)

        print('TRAINING COMPLETE')

        if self.save_model_path is not None:
            print(f"Saving model to {self.save_model_path}")
            torch.save(self.model.state_dict(), self.save_model_path)

        # Finish run (needed to train multiple models from the same script)
        wandb.finish()

        return self.mean_std, self.early_stopping_info


#####################################################
#### Helper functions for PytorchModel class ########
#####################################################

def renormalize(tensor):
    minFrom = tensor.min()
    maxFrom = tensor.max()
    minTo = 0
    maxTo = 1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


def default_image_loader(path: str):
    """Image loader (handles EXIF transformation metadata).
    Can optionally be overloaded by specifying kwarg image_loader when initialising the class."""
    NpIm = NpImage(path, colour=RGB)
    img = NpIm.get_PIL_image_exif()

    return img


def prepend_split_kind(score_dict: WandbScoreDict, prefix: str = None) -> Dict[str, Any]:
    '''Prepend "prefix/" to all keys of the score_dict except epoch.'''
    return {new_key: val for key, val in score_dict.items() if (
        new_key := key if key == 'epoch' else f'{prefix}/{key}')}
