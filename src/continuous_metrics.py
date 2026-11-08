"""與 validated TranAD additional_metrics 完全相同的 continuous-score 三個公式。

獨立入口避免呼叫 Raw/PA wrapper，也不改動既有會被 source fingerprint 核對的檔案。
"""
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def continuous_metrics(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("scores/labels 必須是一對一、有限的 vector")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("ranking metrics 要求 labels 同時包含正常與異常")
    p, r, _ = precision_recall_curve(labels, scores)
    return dict(pr_auc_trapezoidal=float(auc(r, p)),
                average_precision=float(average_precision_score(labels, scores)),
                roc_auc=float(roc_auc_score(labels, scores)))
