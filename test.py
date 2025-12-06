# Modified for multitask regression + classification support.

from __future__ import print_function
import os
import os.path as osp
import json
import numpy as np
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser

def plot_confusion_matrix(cls_pred, cls_target, threshold=0.5, figsize=(4, 4)):
    """
    cls_pred: predicted hotspot probabilities, shape [H, W]
    cls_target: ground-truth hotspot mask, shape [H, W]
    """

    # Binarize
    y_pred = (cls_pred >= threshold).astype(int).flatten()
    y_true = (cls_target >= threshold).astype(int).flatten()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Nice plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"]
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (Hotspot Classification)")
    plt.tight_layout()
    plt.show()

    return cm


def to_cuda(x):
    """Moves x (tensor or tuple/list of tensors) to CUDA."""
    if isinstance(x, (tuple, list)):
        return tuple(t.cuda(non_blocking=True) for t in x)
    return x.cuda(non_blocking=True)


def extract_regression(pred):
    """Handles multitask output: (reg, cls) -> reg."""
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred


def extract_classification(pred):
    """Handles multitask output: (reg, cls) -> cls."""
    if isinstance(pred, (tuple, list)):
        return pred[1]
    return None  # single-task model → no classification


def extract_regression_target(target):
    """Handles multitask target: (reg, cls) -> reg."""
    if isinstance(target, (tuple, list)):
        return target[0]
    return target


def compute_cls_metrics(cls_pred, cls_target, threshold=0.5):
    """
    Computes accuracy, precision, recall, F1 for 2D maps.
    Expects tensors of shape [H, W] with values in [0,1].
    """

    cls_pred_bin = (cls_pred >= threshold).float()
    cls_target_bin = (cls_target >= threshold).float()

    TP = (cls_pred_bin * cls_target_bin).sum().item()
    FP = (cls_pred_bin * (1 - cls_target_bin)).sum().item()
    FN = ((1 - cls_pred_bin) * cls_target_bin).sum().item()
    TN = ((1 - cls_pred_bin) * (1 - cls_target_bin)).sum().item()

    eps = 1e-6

    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "ACC": acc,
        "PREC": precision,
        "REC": recall,
        "F1": f1,
    }


def test():
    # -----------------------------
    # Parse arguments
    # -----------------------------
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    # Convert eval metrics "A,B,C" → ["A","B","C"]
    if isinstance(arg_dict['eval_metric'], str):
        arg_dict['eval_metric'] = [m.strip() for m in arg_dict['eval_metric'].split(',')]

    # Load arg_file if given
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test']
    arg_dict['test_mode'] = True

    # -----------------------------
    # Dataset
    # -----------------------------
    print('===> Loading datasets')
    dataset = build_dataset(arg_dict)

    # -----------------------------
    # Model
    # -----------------------------
    print('===> Building model')
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()

    # -----------------------------
    # Regression Metrics
    # -----------------------------
    metrics = {k: build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k: 0.0 for k in arg_dict['eval_metric']}

    # -----------------------------
    # Classification Metrics
    # -----------------------------
    cls_metrics_totals = {
        "ACC": 0.0,
        "PREC": 0.0,
        "REC": 0.0,
        "F1": 0.0,
    }

    # -----------------------------
    # Inference loop
    # -----------------------------
    count = 0
    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:

            # Move to device
            if arg_dict['cpu']:
                inputs = feature
                targets = label
            else:
                inputs = to_cuda(feature)
                targets = to_cuda(label)

            # --- Regression target ---
            reg_target = extract_regression_target(targets)

            # --- Forward ---
            prediction = model(inputs)

            # Extract outputs
            reg_pred = extract_regression(prediction)
            cls_pred = extract_classification(prediction)

            # -------------------------
            # Regression metrics
            # -------------------------
            for metric, metric_func in metrics.items():
                value = metric_func(reg_target.cpu(), reg_pred.cpu())
                if value != 1:
                    avg_metrics[metric] += value

            # -------------------------
            # Classification metrics
            # -------------------------
            if cls_pred is not None:
                cls_target = targets[1]  # multitask target layout: (reg, cls)
                cls_stats = compute_cls_metrics(cls_pred.cpu(), cls_target.cpu())

                for k, v in cls_stats.items():
                    cls_metrics_totals[k] += v

            # -------------------------
            # Save outputs
            # -------------------------
            save_dir = osp.join(arg_dict['save_path'], 'test_result')
            os.makedirs(save_dir, exist_ok=True)

            fname = osp.splitext(osp.basename(label_path[0]))[0]

            # save reg
            reg_np = reg_pred.float().detach().cpu().numpy()
            np.save(osp.join(save_dir, f"{fname}.npy"), reg_np)

            # save cls
            if cls_pred is not None:
                cls_np = cls_pred.float().detach().cpu().numpy()
                np.save(osp.join(save_dir, f"{fname}_cls.npy"), cls_np)

            count += 1
            bar.update(1)

    # -----------------------------
    # Print Regression Metrics
    # -----------------------------
    for metric, total in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, total / len(dataset)))

    # -----------------------------
    # Print Classification Metrics
    # -----------------------------
    if cls_pred is not None:
        print("\n===> Classification Metrics (threshold = 0.5)")
        for k, total in cls_metrics_totals.items():
            print("===> Avg. {}: {:.4f}".format(k, total / len(dataset)))

    # -----------------------------
    # Optional ROC
    # -----------------------------
    if arg_dict['plot_roc']:
        roc_metric, _ = build_roc_prc_metric(**arg_dict)
        print("\n===> AUC of ROC: {:.4f}".format(roc_metric))


if __name__ == "__main__":
    test()
