from typing import Dict, List, Union

import pandas as pd
from experiment_run import ExperimentRun
from torchvision.models import ResNet101_Weights, VGG19_Weights

IM_SIZE = 704
DATA_ROOT = '.data'
MODELS_DIR = f'./classification/experiments/models/labels_{IM_SIZE}'

BASE_CFG = {
    'batch_size': 10,
    'epochs': 40,
    'image_loader': None,
    'lr': 2.25e-5,
    'num_workers': 16,
    'optimizer': 'Adam',
    'test_dir': DATA_ROOT+'/test/images',
    'train_dir': DATA_ROOT+'/train/images',
    'train_transforms': None,
    'use_wandb': True,
    'input_size': IM_SIZE,
    'full_size': int(IM_SIZE * 1.05)
}

# Label mappings
AVALANCHE = 'avalanche'
NONE = 'none'

RESULTS_FILE = f'./classification/experiments/label_scores_{IM_SIZE}.csv'
'''.csv file where results are saved'''


def get_label_mapping(n_labels: int) -> Union[Dict[str, str], None]:
    if n_labels == 2:
        return {'glide': AVALANCHE, 'loose': AVALANCHE,
                'none': NONE, 'slab': AVALANCHE}


def _resnet_model_name(n_labels: int, architecture: str = 'ResNet101'):
    lowercase_architecture = architecture.lower()
    return f'{lowercase_architecture}_{n_labels}_labels_{IM_SIZE}'


def _vgg_model_name(n_labels: int):
    return f'vgg19_{n_labels}_labels_{IM_SIZE}'


def resnet_label_experiment(n_labels: int, architecture: str = 'ResNet101', weights=None, starting_idx: int = 0):
    '''Train ResNet models for on both 2 labels and 4 labels'''
    assert n_labels == 2 or n_labels == 4, f"Unexpected n_labels {n_labels}"
    model_name = _resnet_model_name(n_labels, architecture)
    resnet_config = {
        **BASE_CFG,
        'num_classes': n_labels,
        'label_mapping': get_label_mapping(n_labels),
        'architecture': architecture,
        'weights': weights,
        'wandb_init': {'project': 'paper_initial_tests',
                       'name': model_name,
                       'tags': ['ResNet', 'IMAGENET1K_V2', 'Adam']}
    }

    Experiment = ExperimentRun(run_dir=MODELS_DIR, experiment_name=model_name)
    Experiment.start_training(n_runs=3,
                              train_config=resnet_config,
                              description=f'Model label experiments with {n_labels} labels',
                              starting_idx=starting_idx)
    del Experiment
    del resnet_config


def vgg_label_experiment(n_labels: int, starting_idx: int = 0):
    '''Train VGG-19 models for on both 2 labels and 4 labels'''
    assert n_labels == 2 or n_labels == 4, f"Unexpected n_labels {n_labels}"
    model_name = _vgg_model_name(n_labels)
    vgg_config = {
        **BASE_CFG,
        'num_classes': n_labels,
        'label_mapping': get_label_mapping(n_labels),
        'architecture': 'vgg19',
        'weights': VGG19_Weights.IMAGENET1K_V1,
        'wandb_init': {'project': 'paper_initial_tests',
                       'name': model_name,
                       'tags': ['vgg19', 'IMAGENET1K_V1', 'Adam']}
    }

    Experiment = ExperimentRun(run_dir=MODELS_DIR, experiment_name=model_name)
    Experiment.start_training(
        n_runs=3, train_config=vgg_config, description=f'Model label experiments with {n_labels} labels',
        starting_idx=starting_idx)
    del Experiment
    del vgg_config


def evaluate_label_models(resnet_architecture: str = 'ResNet101'):
    '''Evaluate label models and save scores to a .csv file'''
    # Define score mappings as binary models have no need for explicit binary scores
    SCORE_MAPPINGS = {
        'accuracy': ('test/accuracy', 'test/binary/accuracy'),
        'f1': ('test/f1', 'test/binary/f1'),
        'precision': ('test/precision', 'test/binary/precision'),
        'recall': ('test/recall', 'test/binary/recall'),
        'weighted_accuracy': ('test/weighted_accuracy', 'test/binary/weighted_accuracy'),
        'weighted_f1': ('test/weighted_f1', 'test/binary/weighted_f1'),
    }
    two_label_scores = {key: s for (key, (s, _)) in SCORE_MAPPINGS.items()}
    four_label_scores = {key: s for (key, (_, s)) in SCORE_MAPPINGS.items()}
    MODEL_KEY = 'model'

    # Create initial scores dict
    scores: Dict[str, List[Union[float, str]]] = {
        MODEL_KEY: []} | {f'{key}/mean': [] for key in list(SCORE_MAPPINGS.keys())
                          } | {f'{key}/std': [] for key in list(SCORE_MAPPINGS.keys())}

    # Calculate scores for each model
    for n_labels, score_keys in [(2, two_label_scores), (4, four_label_scores)]:
        for model_name in [_resnet_model_name(n_labels, resnet_architecture), _vgg_model_name(n_labels)]:
            Experiment = ExperimentRun(
                run_dir=MODELS_DIR, experiment_name=model_name)
            _, score_summary = Experiment.evaluate(score_keys=list(score_keys.values()),
                                                   early_stopping=True, suppress_logging=True)

            # Add calculated model scores scores to scores dict
            scores[MODEL_KEY].append(model_name)
            for key_index, key in score_keys.items():
                scores[f'{key_index}/mean'].append(
                    f"{score_summary[key]['mean']:.1f}")
                scores[f'{key_index}/std'].append(
                    f"{score_summary[key]['std']:.1f}")

    # Save scores as a csv
    output_df = pd.DataFrame(scores)
    output_df.to_csv(RESULTS_FILE, index=False)


def print_latex_results_table():
    '''Print results with a set column order'''
    df = pd.read_csv(RESULTS_FILE)
    col_order = ['f1', 'precision', 'recall', 'accuracy']

    for col_name in col_order:
        mean_std_col_names = [f'{col_name}/mean', f'{col_name}/std']
        df[col_name] = df[mean_std_col_names].apply(
            lambda row: ' \\pm '.join(row.values.astype(str)), axis=1)

    # Reorder columns to desired output order
    df = df[['model'] + col_order]

    # Print output in latex format
    def formatter(x): return f"${x}$"
    print(df.style.format(formatter=formatter, precision=2).to_latex())


if __name__ == '__main__':

    # Train models
    for n_labels in [2, 4]:
        vgg_label_experiment(n_labels)
        resnet_label_experiment(
            n_labels, architecture="ResNet101", weights=ResNet101_Weights.IMAGENET1K_V2)

    evaluate_label_models(resnet_architecture='ResNet101')
    print_latex_results_table()
