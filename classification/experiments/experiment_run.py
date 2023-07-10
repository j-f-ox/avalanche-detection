import json
import os
import shutil
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from torch import tensor
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, VGG13_Weights,
                                VGG16_Weights, VGG19_Weights)

from classification.train.train import PytorchModel

RUN = 'run'
'''Key for run number in config dict.'''

RUNS = 'runs'
'''Key for run information in config.json'''

TRAIN_CONFIG = 'train_config'
'''Key for training config dict in config.json'''

EARLY_STOPPING_PATH = 'early_stopping_path'

FINAL_MODEL_PATH = 'model_path'

MEAN_STD = 'mean_std'
'''Key of mean and standard deviation tuple in run config dict.'''

IMAGE_LOADER = 'image_loader'
'''Key for image loader override in training config.'''

WEIGHTS = 'weights'
'''Key for weights in training config.'''


class ExperimentRun(object):
    """Pytorch class to train several models with the same config, and save the config, trained models, and results."""

    config_path: str
    '''(str): path to the experiment config and details.'''

    experiment_dir: str
    '''(str): the directory of the experiment subfolder and wandb run names.'''

    def __init__(self, run_dir: str = '', experiment_name: str = ''):
        """

        Args:
            experiment_name (str): the name of the experiment subfolder and wandb run names.
            run_dir (str): the directory where all run information will be saved in or loaded from. A folder will be created in this directory.
        """
        assert os.path.exists(
            run_dir), f"Directory not found {os.path.abspath(run_dir)}"

        self.experiment_dir = f'{run_dir}/{experiment_name}'

        self.config_path = os.path.join(
            self.experiment_dir, 'config.json')

        self.experiment_name = experiment_name

    def start_training(self, n_runs: int = 5, train_config: Dict[str, Any] = None, description: str = '', starting_idx: int = 0):
        """

        Args:
            n_runs (int): the total number of training runs to do.
            train_config (Dict[str, Any]): the kwargs to pass to PytorchModel.
            description (str): the training run description (will be saved in the config.json).
            starting_idx (int): if set to an integer, continue training models at this training run index (starting from 0). Does not overwrite previous experiments.
        """
        if starting_idx > 0:
            # If continuing experiment run
            assert isinstance(
                starting_idx, int), f"continue_from_idx must be an int. Got {starting_idx}"
            assert starting_idx < n_runs, f"Can only continue from a run index<n_runs. Got {starting_idx} >= {n_runs}"

            # Check that experiment dir and config exist if continuing from a specific run
            assert os.path.exists(
                self.experiment_dir), f"Experiment directory not found: {self.experiment_dir}"
            assert os.path.exists(
                self.config_path), f"Experiment config not found at {self.config_path}"
        else:
            # If starting run from scratch, create experiment dir
            try:
                shutil.rmtree(self.experiment_dir)
            except:
                pass

            # Create experiment_dir if needed
            os.makedirs(self.experiment_dir, exist_ok=False)

            # Save an initial experiment config
            self.save_experiment_config({
                'name': self.experiment_name,
                'description': description,
                'n_runs': n_runs,
                RUNS: {}
            })
        # Train models
        for i in range(starting_idx, n_runs):
            print(f"Training model {self.experiment_name} - {i+1}/{n_runs}")
            run_dump = {RUN: i+1, 'start': str(datetime.now())}
            model_name = f'{self.experiment_dir}/model_{i+1}'
            model_path = f'{model_name}.pth'
            early_stopping_path = f'{model_name}_earlystopping.pth'

            # Train model
            ExperimentModel = PytorchModel(
                **train_config, save_model_path=model_path, early_stopping_path=early_stopping_path)
            mean_std, early_stopping_info = ExperimentModel.run_training()
            del ExperimentModel
            run_dump.update({
                FINAL_MODEL_PATH: model_path,
                'end': str(datetime.now()),
                MEAN_STD: mean_std,
                **early_stopping_info
            })
            self._update_experiment_runs({f'model_{i+1}': run_dump})

        if MEAN_STD in train_config:
            # Convert tensor to JSON serializable array
            mean, std = train_config[MEAN_STD]
            mean = mean.tolist()
            std = std.tolist()
            train_config[MEAN_STD] = (mean, std)

        # Convert non-serialisable keys to strings
        for nonserialisable_key in [WEIGHTS, IMAGE_LOADER]:
            if nonserialisable_key in train_config:
                train_config[nonserialisable_key] = str(
                    train_config[nonserialisable_key])

        self._update_experiment_dump({TRAIN_CONFIG: train_config})

    def save_experiment_config(self, obj: Dict[str, Any]):
        """Save python dict to self.config_path."""
        with open(self.config_path, 'w') as f:
            json.dump(obj, f, indent=2)

    def _update_experiment_dump(self, obj):
        with open(self.config_path, 'rb') as f:
            old_config: Dict = json.load(f)

        old_config.update(obj)
        self.save_experiment_config(old_config)

    def _update_experiment_runs(self, run_obj):
        """Load config, update RUNS key, and save updated config."""
        old_config = self.load_config()
        old_runs = old_config.get(RUNS, {})

        old_runs.update(run_obj)
        old_config.update({RUNS: old_runs})
        with open(self.config_path, 'w') as f:
            json.dump(old_config, f, indent=2)

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'rb') as f:
            config: Dict[str, Any] = json.load(f)
        return config

    def evaluate(self, image_loader: Callable[[str], Any] = None, score_keys: List[str] = None,
                 early_stopping: bool = False, suppress_logging: bool = True,
                 test_dir: str = None, train_dir: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate training run models. Get mean and median of final and early stopping models.

        Args:
            image_loader (Callable[[str], Any]): the same image loader used for training if a custom loader was used.
            score_keys (List[str]): the score keys to calculate the mean, median, and standard deviation for.
            early_stopping (bool): if true, use the saved early stopping model. Otherwise, use the final model.
            test_dir (str): if set, load test images from this directory rather than the directory in the train config.
            train_dir (str): if set, load train images from this directory rather than the directory in the train config.
        Returns:
            ([dict], dict): a list of the score dictionaries of all models, and a dictionary of the mean, median and standard deviation of selected scores over all runs."""
        config = self.load_config()
        config_runs: Dict[str, Any] = config[RUNS]

        # Load and correct train config
        train_config = config[TRAIN_CONFIG]
        train_config['use_wandb'] = False  # Supress wandb logging

        # Optionally override the train and test directories
        if test_dir is not None:
            train_config['test_dir'] = test_dir
        if train_dir is not None:
            train_config['train_dir'] = train_dir

        # Check for image loader overwrite and fix None being saved as a string
        if IMAGE_LOADER in train_config and train_config[IMAGE_LOADER] != "None":
            assert image_loader is not None, f"Image loader was overridden. Expected {train_config[IMAGE_LOADER]}."
            train_config[IMAGE_LOADER] = image_loader
        else:
            train_config[IMAGE_LOADER] = None

        # Load model weights (needed for hidden layers)
        weight_str: str = train_config[WEIGHTS]
        train_config[WEIGHTS] = self._load_weights(weight_str)

        score_dicts = []
        for run_dict in config_runs.values():
            print(
                f'Calculating scores for {self.experiment_name} run {run_dict[RUN]}')
            # Load mean and standard deviation from config
            list_mean, list_std = run_dict[MEAN_STD]
            train_config[MEAN_STD] = (tensor(list_mean), tensor(list_std))

            if early_stopping:
                model_path = run_dict[EARLY_STOPPING_PATH]
            else:
                model_path = run_dict[FINAL_MODEL_PATH]

            test_scores = PytorchModel(
                **train_config, load_model_path=model_path)._eval_test(epoch=-1,
                                                                       suppress_logging=suppress_logging,
                                                                       calculate_weighted_scores=True)

            score_dicts.append(test_scores)

        if score_keys is None:
            score_keys = ['test/accuracy', 'test/f1', 'test/recall',
                          'test/precision', 'test/accuracy_binary', 'test/f1_binary']

        score_summaries = {}
        for score_key in score_keys:
            scores_arr = np.array([i[score_key] for i in score_dicts])
            # Calculate scores mean, median, standard deviation
            score_summaries[score_key] = {
                'mean': np.mean(scores_arr),
                'median': np.median(scores_arr),
                'std': np.std(scores_arr)
            }

        return score_dicts, score_summaries

    def _load_weights(self, weight_str: str):
        '''Takes a string of model weights and returns the weights object'''
        supported_weights = {
            'ResNet152_Weights.IMAGENET1K_V2': ResNet152_Weights.IMAGENET1K_V2,
            'ResNet101_Weights.IMAGENET1K_V2':  ResNet101_Weights.IMAGENET1K_V2,
            'ResNet50_Weights.IMAGENET1K_V2': ResNet50_Weights.IMAGENET1K_V2,
            'ResNet34_Weights.IMAGENET1K_V1': ResNet34_Weights.IMAGENET1K_V1,
            'ResNet18_Weights.IMAGENET1K_V1': ResNet18_Weights.IMAGENET1K_V1,
            'VGG19_Weights.IMAGENET1K_V1': VGG19_Weights.IMAGENET1K_V1,
            'VGG16_Weights.IMAGENET1K_V1': VGG16_Weights.IMAGENET1K_V1,
            'VGG13_Weights.IMAGENET1K_V1': VGG13_Weights.IMAGENET1K_V1,
        }
        if weight_str in supported_weights.keys():
            return supported_weights[weight_str]
        else:
            raise NotImplementedError(
                f'Weights {weight_str} not supported yet.')
