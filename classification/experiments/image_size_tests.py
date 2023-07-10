from typing import Dict, List

import pandas as pd
from experiment_run import ExperimentRun
from torchvision.models import ResNet101_Weights, VGG19_Weights

DATA_ROOT = '.data'
'''Folder containing /train and /test image folders'''

MODELS_DIR = './classification/experiments/models'
'''Directory to save output models'''

BASE_CFG = {
    'batch_size': 7,  # 10 used for VGG_19 and 7 for ResNet101
    'epochs': 40,
    'image_loader': None,
    'lr': 2.25e-5,
    'num_classes': 4,
    'num_workers': 16,
    'optimizer': 'Adam',
    'test_dir': DATA_ROOT+'/test/images',
    'train_dir': DATA_ROOT+'/train/images',
    'train_transforms': None,
    'use_wandb': True
}

IM_SIZES = [1120, 896, 672, 448, 224]
'''Range of network input sizes for each model'''


def _resnet_model_name(input_size: int):
    return f'resnet101_{input_size}'


def _vgg_model_name(input_size: int):
    return f'vgg19_{input_size}'


def resnet_imsize_experiment(input_size: int, weights=ResNet101_Weights.IMAGENET1K_V2,
                             wandb_project: str = 'avalanche_segmentation_image_size'):
    '''Train ResNet{resnet_version} models for the given image input size (in px)'''
    model_name = _resnet_model_name(input_size)
    resnet_config = {
        **BASE_CFG,
        'input_size': input_size,
        'full_size': int(input_size*1.05),
        'architecture': f'ResNet101',
        'weights': weights,
        'wandb_init': {'project': wandb_project,
                       'name': model_name,
                       'tags': ['ResNet', 'IMAGENET1K_V2', 'Adam']}
    }

    Experiment = ExperimentRun(run_dir=MODELS_DIR, experiment_name=model_name)
    Experiment.start_training(
        n_runs=3, train_config=resnet_config, description=f'Image size experiment with input size {input_size}')


def vgg_imsize_experiment(input_size: int, weights=None):
    '''Train VGG-19 models for the given image input size (in px)'''
    model_name = _vgg_model_name(input_size)
    vgg_config = {
        **BASE_CFG,
        'input_size': input_size,
        'full_size': int(input_size*1.05),
        'architecture': f'vgg19',
        'weights': weights,
        'wandb_init': {'project': 'paper_initial_tests',
                       'name': model_name,
                       'tags': [f'vgg19', 'IMAGENET1K_V1', 'Adam']}
    }

    Experiment = ExperimentRun(run_dir=MODELS_DIR, experiment_name=model_name)
    Experiment.start_training(
        n_runs=3, train_config=vgg_config, description=f'Image size experiment with input size {input_size}')


def evaluate_imsize_models():
    '''Evaluate the models, format scores, and print them to the console.'''
    SCORE_KEY = 'test/accuracy'

    def _get_score_from_model_name(model_name: str) -> Dict[str, float]:
        '''Load a model and return the mean score with key SCORE_KEY'''
        Experiment = ExperimentRun(
            run_dir=MODELS_DIR, experiment_name=model_name)
        all_scores, score_summary = Experiment.evaluate(score_keys=[SCORE_KEY],
                                                        early_stopping=True, suppress_logging=True)

        print(score_summary)
        return score_summary[SCORE_KEY]
    resnet_scores: List[float] = []
    vgg_scores: List[float] = []
    for im_size in IM_SIZES:
        # Get ResNet results
        resnet_model_name = _resnet_model_name(im_size)
        resnet_scores.append(_get_score_from_model_name(resnet_model_name))

        # Get VGG results
        vgg_model_name = _vgg_model_name(im_size)
        vgg_scores.append(_get_score_from_model_name(vgg_model_name))

    resnet_mean = [d['mean'] for d in resnet_scores]
    vgg_mean = [d['mean'] for d in vgg_scores]
    output_df = pd.DataFrame(
        {'imsize': IM_SIZES, 'resnet_mean': resnet_mean, 'vgg_mean': vgg_mean, })
    output_df.to_csv(
        './classification/experiments/im_size_accuracy.csv', index=False)


if __name__ == '__main__':
    # Train models
    for input_size in IM_SIZES:
        vgg_imsize_experiment(input_size, weights=VGG19_Weights.IMAGENET1K_V1)
        resnet_imsize_experiment(
            input_size, weights=ResNet101_Weights.IMAGENET1K_V1)

    # Evaluate models
    evaluate_imsize_models()
