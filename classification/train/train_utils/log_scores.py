from typing import Any, Dict, List, Tuple

import sklearn.metrics as metrics

from classification.train.train_utils.consts import (ScoreKind,
                                                     WandbBinaryDict,
                                                     WandbMulticlassDict)


def log_scores_binaryclass(targets: List[float], predicted: List[float], kind: ScoreKind,
                           epoch: int, suppress_logging: bool = False, calculate_weighted_scores: bool = False) -> WandbBinaryDict:
    """Log binary scores to console and return wandb dictionary
    N.B. expects avalanche classes to have label 0 and non-avalanche to have label 1

    Args:
        targets ([float])       : the true labels.
        predicted ([float])     : the predicted labels.
        kind (ScoreKind)        : the kind of score to log.
        epoch (int)             : the current epoch.
        suppress_logging (bool) : if True, do not print scores to console.
        calculate_weighted_scores (bool) : if True, also calculate scores weighted by support for each label.

    Returns:
        (WandbBinaryDict) A dict of score values.
    """
    accuracy = metrics.accuracy_score(targets, predicted) * 100
    precision, recall, f1score = _precision_recall_f1(
        targets, predicted, pos_label=0, average='binary')

    tp, fn, fp, tn = metrics.confusion_matrix(targets, predicted).ravel()

    bin_ = 'binary/'  # Prefix for binary scores
    wandb_dict = {"Epoch": epoch+1, f"{bin_}accuracy": accuracy,
                  f"{bin_}precision": precision, f"{bin_}recall": recall, f"{bin_}f1": f1score}

    if not suppress_logging:
        print(f"----- {kind.value}/Binary Scores ----- epoch: {epoch +1} -----")
        print(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1_Score: {f1score:.3f}")
        print("---------- Confusion Matrix ----------")
        print(" t\p         | Avalanche  No Avalanche")
        print("--------------------------------------")
        print(f"Avalanche    | {tp:9} {fn:12}")
        print(f"No Avalanche | {fp:9} {tn:12}")
        print("TP: ", tp, " FP: ", fp, " TN: ", tn, "FN: ", fn)
        print("--------------------------------------\n")

    if calculate_weighted_scores:
        w_accuracy = metrics.balanced_accuracy_score(targets, predicted) * 100
        w_precision, w_recall, w_f1score = _precision_recall_f1(
            targets, predicted,  average='weighted')
        wandb_dict.update({f"{bin_}weighted_accuracy": w_accuracy,
                           f"{bin_}weighted_f1": w_f1score,
                           f"{bin_}weighted_precision": w_precision,
                           f"{bin_}weighted_recall": w_recall})
    return wandb_dict


def log_scores_multiclass(targets: List[float] = [], predicted: List[float] = [],
                          kind: ScoreKind = None, epoch: int = None, idx_to_class: Dict[int, str] = {},
                          suppress_logging: bool = False, calculate_weighted_scores: bool = False) -> WandbMulticlassDict:
    """Log multiclass scores to console and return wandb dictionary

    Args:
        targets ([float])             : the true labels.
        predicted ([float])           : the predicted labels.
        kind (ScoreKind)              : the kind of score to log.
        epoch (int)                   : the current epoch.
        idx_to_class (Dict[int, str]) : dictionary of labels indexed by label index.
        suppress_logging (bool)       : if True, do not print scores to console.
        calculate_weighted_scores (bool) : if True, also calculate scores weighted by support for each label..

    Returns:
        (WandbMulticlassDict) A dict of score values.
    """
    if not suppress_logging:
        print(f"----- {kind.value} ----- epoch: {epoch + 1} -----")
        print("Number of labels: ", len(targets))
        print("Number of true labels per class: ")

    # Get labels from idx_to_class dictionary and log class labels and count to console
    labels = list(idx_to_class.keys())
    class_str = "  "
    for label_idx, label in enumerate(labels):
        if label_idx > 0:
            class_str += ", "
        class_str += f"class {label}/{idx_to_class[label]}: {targets.count(label)}"
    if not suppress_logging:
        print(class_str)

    accuracy = metrics.accuracy_score(targets, predicted) * 100
    # Calculate scores for each label
    precision, recall, f1score = _precision_recall_f1(
        targets, predicted, average=None, labels=labels)
    # Calculate overall (macro) scores
    um_precision, um_recall, um_f1score = _precision_recall_f1(
        targets, predicted, average='macro')

    conf_matrix = metrics.confusion_matrix(
        targets, predicted, labels=labels)

    wandb_dict = {"Epoch": epoch+1, "accuracy": accuracy,
                  "precision": um_precision, "recall": um_recall, "f1": um_f1score}
    if not suppress_logging:
        print(f"Accuracy: {accuracy:.3f}")
    for i, label in enumerate(labels):
        if not suppress_logging:
            print(f"------ {label}/{idx_to_class[label]} -----")
            print(
                f"Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1: {f1score[i]:.3f}")
        label_precision = f"{idx_to_class[label]}/precision"
        wandb_dict[label_precision] = precision[i]
        label_recall = f"{idx_to_class[label]}/recall"
        wandb_dict[label_recall] = recall[i]
        label_f1 = f"{idx_to_class[label]}/f1"
        wandb_dict[label_f1] = f1score[i]

    if not suppress_logging:
        print("----- Unweighted mean -----")
        print(
            f"Precision: {um_precision:.3f}, Recall: {um_recall:.3f}, F1: {um_f1score:.3f}")

        print("---------- Confusion Matrix ----------")
        print(f"  true/pred   |"+''.join([f" {i:5}" for i in labels]))
        print("--------------------------------------")
        for row_idx, row in enumerate(conf_matrix):
            row_label_idx, row_label = list(idx_to_class.items())[row_idx]
            print(f"  {row_label:9} {row_label_idx} | ",
                  ', '.join([f"{label_count:4}" for label_count in row]))

        print("--------------------------------------\n")

    if calculate_weighted_scores:
        w_accuracy = metrics.balanced_accuracy_score(targets, predicted) * 100
        w_precision, w_recall, w_f1score = _precision_recall_f1(
            targets, predicted, average='weighted')
        wandb_dict.update(
            {"weighted_accuracy": w_accuracy, "weighted_f1": w_f1score,
             "weighted_precision": w_precision, "weighted_recall": w_recall})

    return wandb_dict


def _precision_recall_f1(targets: List[float], predicted: List[float], **kwargs) -> Tuple[float, float, float]:
    """Helper function to return a tuple of precision, recall and f1 score multiplied by 100 for the given labels

    Args:
        targets ([float])   : the true labels
        predicted ([float]) : the predicted labels
        **kwargs            : other kwargs to pass directly to each metrics.[[score_type]] function

    Returns:
        (float, float, float) : tuple of (precision, recall, f1 score) for the given labels in range [0, 100]""
    """
    precision = metrics.precision_score(
        targets, predicted, zero_division=0, **kwargs) * 100
    recall = metrics.recall_score(
        targets, predicted, zero_division=0, **kwargs) * 100
    f1_score = metrics.f1_score(
        targets, predicted, zero_division=0, **kwargs) * 100
    return precision, recall, f1_score


def log_epoch_scores(train_scores: Dict[str, Any] = None, valid_scores: Dict[str, Any] = None, test_scores: Dict[str, Any] = None, ):
    '''Log scores for current epoch to the console'''
    train_loss = train_scores['train/loss']
    train_acc = train_scores['train/accuracy']
    valid_acc = valid_scores['validation/accuracy']
    test_loss = test_scores['test/loss']
    test_acc = test_scores['test/accuracy']
    train_str = f"Training loss: {train_loss:.3f}, Train acc: {train_acc:.3f}"
    test_str = f"Test loss: {test_loss:.3f}, Test acc: {test_acc:.3f}"

    # Binary scores are not always defined so only print if included in scores dicts
    if 'train/binary/accuracy' in train_scores and 'test/binary/accuracy' in test_scores:
        train_acc_binary = train_scores.get('train/binary/accuracy', '')
        test_acc_binary = test_scores.get('test/binary/accuracy', '')
        train_str += f", Train acc bin: {train_acc_binary:.3f}"
        test_str += f", Test acc bin: {test_acc_binary:.3f}"

    print(train_str)
    print(test_str)
    print(f"Validation accuracy: {valid_acc:.3f}")
