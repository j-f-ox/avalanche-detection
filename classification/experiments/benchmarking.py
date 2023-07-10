from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from experiment_run import ExperimentRun
from torchvision.models import (ResNet18_Weights, ResNet34_Weights,
                                ResNet50_Weights, ResNet101_Weights,
                                ResNet152_Weights, VGG13_Weights,
                                VGG16_Weights, VGG19_Weights)

IM_SIZE = 704
'''Network input size'''

RESULTS_FILE = f'./classification/experiments/benchmark_scores_{IM_SIZE}.csv'
'''File path to the output results'''

DATA_ROOT = '.data/images'
'''Folder containing /train and /test image folders'''

MODELS_DIR = f'./classification/experiments/models/benchmarks_{IM_SIZE}'
'''Output directory for saved models'''

BASE_CFG = {
    'batch_size': 10,
    'epochs': 40,
    'image_loader': None,
    'lr': 2.25e-5,
    'num_classes': 4,
    'num_workers': 16,
    'optimizer': 'Adam',
    'test_dir': DATA_ROOT+'/test/images',
    'train_dir': DATA_ROOT+'/train/images',
    'train_transforms': None,
    'use_wandb': True,
    'input_size': IM_SIZE,
    'full_size': int(IM_SIZE * 1.05),
}
'''Default config to use for training runs'''

BENCHMARK_MODELS: List[Tuple[str, Any]] = [
    ('ResNet152', ResNet152_Weights.IMAGENET1K_V2),
    ('ResNet101', ResNet101_Weights.IMAGENET1K_V2),
    ('ResNet50', ResNet50_Weights.IMAGENET1K_V2),
    ('ResNet34', ResNet34_Weights.IMAGENET1K_V1),
    ('ResNet18', ResNet18_Weights.IMAGENET1K_V1),
    ('vgg19', VGG19_Weights.IMAGENET1K_V1),
    ('vgg16', VGG16_Weights.IMAGENET1K_V1),
    ('vgg13', VGG13_Weights.IMAGENET1K_V1),
]


def _model_name(architecture: str) -> str:
    '''Model name for a given architecture'''
    return f'{architecture}_benchmark_{IM_SIZE}'


def train_model(architecture: str, weights, wandb_project: str = 'avalanche_benchmark'):
    '''Train a model for the given architecture and weights.

    Args:
        architecture (str):  the model base architecture.
        weights:             PyTorch weights for the given base architecture.
        wandb_project (str): name of the wandb project for logging results.
    '''
    model_name = _model_name(architecture)
    train_config = {
        **BASE_CFG,
        'architecture': architecture,
        'weights': weights,
        'wandb_init': {'project': wandb_project,
                       'name': model_name,
                       'tags': [architecture, 'Adam', str(weights)]}
    }
    Experiment = ExperimentRun(run_dir=MODELS_DIR, experiment_name=model_name)
    Experiment.start_training(
        n_runs=3, train_config=train_config, description=f'Benchmark experiment for {architecture} with weights {weights}')


def eval_benchmark_models(score_keys=['test/accuracy', 'test/f1', 'test/binary/accuracy',
                                      'test/binary/f1', 'test/weighted_accuracy', 'test/weighted_f1',
                                      'test/binary/weighted_accuracy', 'test/binary/weighted_f1']):
    '''Evaluate trained models and save scores to a .csv file

    Args:
        score_keys (list[str]): list of score keys to save values for'''
    model_key = 'model'

    scores: Dict[str, List[Union[float, str]]] = {
        model_key: []} | {f'{key}/mean': [] for key in score_keys} | {f'{key}/std': [] for key in score_keys}

    # Calculate scores for each model
    for (architecture, _) in BENCHMARK_MODELS:
        model_name = _model_name(architecture)
        _, score_summary = _get_score_from_model_name(
            model_name, score_keys=score_keys)
        print(score_summary)

        scores[model_key].append(architecture)
        for key in score_keys:
            scores[f'{key}/mean'].append(f'{score_summary[key]["mean"]:.1f}')
            scores[f'{key}/std'].append(f'{score_summary[key]["std"]:.1f}')

    # Save scores as a csv
    output_df = pd.DataFrame(scores)
    output_df.to_csv(RESULTS_FILE, index=False)


def _get_score_from_model_name(model_name: str, suppress_logging: bool = True,
                               test_dir=None, train_dir=None,
                               score_keys: List[str] = []) -> Dict[str, float]:
    '''Load a model and return the mean scores with keys in score_keys

    Args:
        model_name (str): the model to load.
        suppress_logging (bool): if false then print verbose output.
        test_dir (str): optionally override the test directory saved in the model config.
        train_dir (str): optionally override the train directory saved in the model config.
        score_keys (list[str]): the metrics to save for the model.
    '''
    Experiment = ExperimentRun(
        run_dir=MODELS_DIR, experiment_name=model_name)
    _, score_summary = Experiment.evaluate(score_keys=score_keys, early_stopping=True,
                                           suppress_logging=suppress_logging, test_dir=test_dir, train_dir=train_dir)
    print(score_summary)
    return ({key: f"{score_summary[key]['mean']:.2f}" for key in score_keys}), score_summary


if __name__ == '__main__':
    # Train models
    for (architecture, weights) in BENCHMARK_MODELS:
        train_model(architecture, weights)

    # Evaluate models
    eval_benchmark_models()
